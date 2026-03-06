"""
ViT 动态事件推理（仅已知分类，不依赖 KMeans/OCSVM 开放集）。
每流维护 clip 缓冲，按窗口步长触发编码 + 已知分类器，返回 pred_label / pred_prob / ranking_score。
推理在独立线程中异步执行，不阻塞主线程。
"""
from __future__ import annotations

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
from lab_anomaly.models.vit_video_encoder import HfVideoEncoder, HfVideoEncoderConfig


def _get_ckpt_embedding_dim(checkpoint_path: str | Path) -> int:
    """从已知分类器 checkpoint 读取 embedding_dim。"""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    cfg_d = ckpt.get("cfg", None)
    if not cfg_d:
        raise RuntimeError("checkpoint 缺少 cfg，无法读取 embedding_dim")
    return int(cfg_d.get("embedding_dim", 768))


def _compute_flow_clip_farneback(clip_frames: list[np.ndarray]) -> np.ndarray:
    """对 clip 内连续帧用 Farneback 计算光流。返回 (T-1, 2, H, W) float32。"""
    if len(clip_frames) < 2:
        return np.zeros((0, 2, 224, 224), dtype=np.float32)
    flows = []
    for i in range(len(clip_frames) - 1):
        g0 = cv2.cvtColor(clip_frames[i], cv2.COLOR_RGB2GRAY)
        g1 = cv2.cvtColor(clip_frames[i + 1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            g0, g1, None, pyr_scale=0.5, levels=2, winsize=15,
            iterations=2, poly_n=5, poly_sigma=1.2, flags=0,
        )
        flow = np.asarray(flow, dtype=np.float32).transpose(2, 0, 1)
        flows.append(flow)
    return np.stack(flows, axis=0)


@dataclass
class VitEventResult:
    """单次 ViT 动态事件结果（仅已知分类）。"""
    pred_label: str
    pred_prob: float
    ranking_score: float


class KnownEventRuntime:
    """
    仅已知分类的 ViT 事件推理：每流维护 clip 缓冲，按 window_stride 触发推理，不加载 OpenSet。
    """

    def __init__(
        self,
        known_checkpoint: str | Path,
        device: Optional[str] = None,
        clip_len: int = 16,
        frame_stride: int = 2,
        window_stride: int = 4,
        encoder_model_name: str = "MCG-NJU/videomae-base",
        use_half: bool = True,
    ):
        self.known_checkpoint = Path(known_checkpoint)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_len = int(clip_len)
        self.frame_stride = max(1, int(frame_stride))
        self.window_stride = max(1, int(window_stride))

        ckpt_embed_dim = _get_ckpt_embedding_dim(self.known_checkpoint)
        use_dual = ckpt_embed_dim == 1536
        enc_cfg = HfVideoEncoderConfig(
            model_name=encoder_model_name,
            use_half=use_half,
            use_dual_stream=use_dual,
            fusion_method="concat",
        )
        self.encoder = HfVideoEncoder(enc_cfg, device=self.device)
        self.encoder.eval()
        if getattr(self.encoder, "embedding_dim", 768) != ckpt_embed_dim:
            raise RuntimeError(
                f"编码器输出维度 {getattr(self.encoder, 'embedding_dim', 768)} "
                f"与 checkpoint embedding_dim {ckpt_embed_dim} 不一致。"
            )
        self.known = load_known_classifier(self.known_checkpoint, device=self.device)

        self._clip_total = (self.clip_len - 1) * self.frame_stride + 1
        self._buffers: Dict[str, deque] = {}
        self._push_count: Dict[str, int] = {}
        self._last_infer_push: Dict[str, int] = {}
        self._results: Dict[str, VitEventResult] = {}
        self._infer_queue: queue.Queue[Tuple[str, List[np.ndarray], Optional[Dict[str, Any]]]] = queue.Queue(maxsize=32)
        self._vit_worker_thread = threading.Thread(target=self._vit_worker_loop, daemon=True)
        self._vit_worker_thread.start()

    def _run_inference(self, stream_id: str, clip_frames: List[np.ndarray], state: Optional[Dict[str, Any]]) -> None:
        """Run encoder + classifier (called from worker thread)."""
        with torch.no_grad():
            if getattr(self.encoder.cfg, "use_dual_stream", False):
                clip_flows = [_compute_flow_clip_farneback(clip_frames)]
                emb_t = self.encoder([clip_frames], clip_flows)
            else:
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
            if stream_id not in state:
                state[stream_id] = {}
            state[stream_id]["vit_event"] = {
                "pred_label": result.pred_label,
                "pred_prob": result.pred_prob,
                "ranking_score": result.ranking_score,
            }

    def _vit_worker_loop(self) -> None:
        """Worker thread: consume infer queue and run ViT inference without blocking main thread."""
        while True:
            try:
                item = self._infer_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            stream_id, clip_frames, state = item
            try:
                self._run_inference(stream_id, clip_frames, state)
            except Exception:
                pass

    def _ensure_stream(self, stream_id: str) -> None:
        if stream_id not in self._buffers:
            self._buffers[stream_id] = deque(maxlen=self._clip_total)
            self._push_count[stream_id] = 0
            self._last_infer_push[stream_id] = 0

    def add_frame(self, stream_id: str, frame: np.ndarray, state: Optional[Dict[str, Any]] = None) -> None:
        """
        将一帧加入该流的缓冲（接受 BGR 或 RGB）；若满足步长且缓冲满则提交到异步推理队列并立即返回。
        frame: (H, W, 3) uint8，BGR（OpenCV 默认）时内部转 RGB 后入缓冲，仅一次转换无多余 copy。
        state: 若提供则 worker 完成后会更新 state[stream_id]["vit_event"]
        """
        self._ensure_stream(stream_id)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._buffers[stream_id].append(rgb)
        self._push_count[stream_id] += 1
        buf = self._buffers[stream_id]
        if len(buf) < self._clip_total:
            return
        if (self._push_count[stream_id] - self._last_infer_push[stream_id]) < self.window_stride:
            return
        self._last_infer_push[stream_id] = self._push_count[stream_id]

        clip_frames = [buf[i * self.frame_stride] for i in range(self.clip_len)]
        clip_copy = [f.copy() for f in clip_frames]
        try:
            self._infer_queue.put_nowait((stream_id, clip_copy, state))
        except queue.Full:
            pass

    def get_result(self, stream_id: str) -> Optional[VitEventResult]:
        """返回该流最近一次 ViT 推理结果，无则返回 None。"""
        return self._results.get(stream_id)
