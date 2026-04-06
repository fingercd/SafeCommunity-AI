"""
已知事件推理：按滑窗组 12 帧（均匀抽帧），VideoMAE v2 编码 + MIL（异步线程，不阻塞主循环）。
"""
from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lab_anomaly.infer.scoring import load_known_classifier
from lab_anomaly.models.vit_video_encoder import VideoMAEv2Encoder, VideoMAEv2EncoderConfig

logger = logging.getLogger("vit.known_event_runtime")


def _uniform_frame_positions(buffer_len: int, num_frames: int) -> List[int]:
    """在长度为 buffer_len 的缓冲区上下标中均匀取 num_frames 个位置。"""
    if buffer_len <= 0 or num_frames <= 0:
        return []
    if num_frames == 1:
        return [min(buffer_len - 1, 0)]
    hi = buffer_len - 1
    return [min(max(0, int(round(i * hi / (num_frames - 1)))), hi) for i in range(num_frames)]


@dataclass
class VitEventResult:
    pred_label: str
    pred_prob: float
    ranking_score: float


class KnownEventRuntime:
    """
    滑窗：缓冲区存最近 (num_frames) 个**已采样帧**，每次推理从中均匀取 frames_per_clip 帧；
    window_stride 表示相隔多少次入帧触发一次推理。
    """

    _MAX_CONSECUTIVE_ERRORS = 10

    def __init__(
        self,
        known_checkpoint: str | Path,
        device: Optional[str] = None,
        frames_per_clip: int = 16,
        window_stride: int = 4,
        encoder_model_name: Optional[str] = None,
        use_half: bool = True,
    ):
        self.known_checkpoint = Path(known_checkpoint)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.frames_per_clip = int(frames_per_clip)
        self.window_stride = max(1, int(window_stride))

        bundle = load_known_classifier(self.known_checkpoint, device=self.device)
        self.known = bundle
        if bundle.encoder is None:
            ckpt = torch.load(str(self.known_checkpoint), map_location="cpu", weights_only=False)
            name = encoder_model_name or ckpt.get("encoder_cfg", {}).get(
                "model_name", "OpenGVLab/VideoMAEv2-Base"
            )
            enc_cfg = VideoMAEv2EncoderConfig(
                model_name=name,
                num_frames=self.frames_per_clip,
                use_half=bool(use_half and self.device.startswith("cuda")),
            )
            self.encoder = VideoMAEv2Encoder(enc_cfg, device=self.device)
            if "encoder_state" in ckpt:
                self.encoder.load_state_dict(ckpt["encoder_state"], strict=True)
        else:
            self.encoder = bundle.encoder

        self.encoder.eval()
        self.known.model.eval()

        self._buf_max = self.frames_per_clip
        self._buffers: Dict[str, deque] = {}
        self._push_count: Dict[str, int] = {}
        self._last_infer_push: Dict[str, int] = {}
        self._results: Dict[str, VitEventResult] = {}
        self._infer_queue: queue.Queue[Tuple[str, List[np.ndarray], Optional[Dict[str, Any]]]] = queue.Queue(maxsize=32)
        self._vit_worker_thread = threading.Thread(target=self._vit_worker_loop, daemon=True)
        self._vit_worker_thread.start()

    def _run_inference(self, stream_id: str, clip_frames: List[np.ndarray], state: Optional[Dict[str, Any]]) -> None:
        with torch.no_grad():
            emb_t = self.encoder([clip_frames])
        emb_np = emb_t.detach().cpu().numpy().astype(np.float32).reshape(-1)
        x = torch.from_numpy(emb_np).view(1, 1, -1).to(self.known.device)
        mask = torch.ones((1, 1), device=self.known.device, dtype=torch.bool)
        with torch.no_grad():
            logits, details = self.known.model(x, mask=mask, y=None, return_details=True)
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy().reshape(-1)
        pred_i = int(probs.argmax())
        pred_label = str(self.known.idx2label.get(pred_i, str(pred_i)))
        pred_prob = float(probs[pred_i])
        ranking_score = 0.0
        if isinstance(details, dict) and "anomaly_scores" in details:
            ranking_score = float(details["anomaly_scores"].max().item())
        result = VitEventResult(pred_label=pred_label, pred_prob=pred_prob, ranking_score=ranking_score)
        self._results[stream_id] = result
        if state is not None and isinstance(state, dict):
            st = state.get(stream_id)
            if st is not None and hasattr(st, "vit_event"):
                st.vit_event = {
                    "pred_label": result.pred_label,
                    "pred_prob": result.pred_prob,
                    "ranking_score": result.ranking_score,
                }

    def _vit_worker_loop(self) -> None:
        consecutive_errors = 0
        while True:
            try:
                item = self._infer_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            stream_id, clip_frames, state = item
            try:
                self._run_inference(stream_id, clip_frames, state)
                consecutive_errors = 0
            except Exception:
                consecutive_errors += 1
                logger.exception(
                    "ViT inference failed for stream=%s (consecutive_errors=%d)",
                    stream_id, consecutive_errors,
                )
                if consecutive_errors >= self._MAX_CONSECUTIVE_ERRORS:
                    logger.error(
                        "ViT worker pausing 10s after %d consecutive errors",
                        consecutive_errors,
                    )
                    import time
                    time.sleep(10)
                    consecutive_errors = 0

    def _ensure_stream(self, stream_id: str) -> None:
        if stream_id not in self._buffers:
            self._buffers[stream_id] = deque(maxlen=self._buf_max)
            self._push_count[stream_id] = 0
            self._last_infer_push[stream_id] = 0

    def add_frame(self, stream_id: str, frame: np.ndarray, state: Optional[Dict[str, Any]] = None) -> None:
        """
        每传入一帧（BGR），转成 RGB 入队；缓冲区满 frames_per_clip 且满足 stride 后触发推理。
        """
        self._ensure_stream(stream_id)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._buffers[stream_id].append(rgb)
        self._push_count[stream_id] += 1
        buf = self._buffers[stream_id]
        if len(buf) < self._buf_max:
            return
        if (self._push_count[stream_id] - self._last_infer_push[stream_id]) < self.window_stride:
            return
        self._last_infer_push[stream_id] = self._push_count[stream_id]

        pos = _uniform_frame_positions(len(buf), self.frames_per_clip)
        buf_list = list(buf)
        clip_frames = [buf_list[i].copy() for i in pos]
        try:
            self._infer_queue.put_nowait((stream_id, clip_frames, state))
        except queue.Full:
            logger.debug("ViT infer queue full for stream=%s, frame dropped", stream_id)

    def get_result(self, stream_id: str) -> Optional[VitEventResult]:
        return self._results.get(stream_id)
