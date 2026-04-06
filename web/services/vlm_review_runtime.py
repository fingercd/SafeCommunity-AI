"""
当 ViT 判定异常且置信度 >= 阈值时，将同一 clip 送 Agent 复核；限流与冷却。
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np


class VlmReviewRuntime:
    """
    维护每流 last_clip；当 ViT 结果满足阈值时提交到 Agent 队列。
    Worker 线程调用 VLMEngine.analyze_clip 并写回 state[stream_id]["agent_result"]。
    """

    def __init__(
        self,
        engine: Any,
        state_ref: Dict[str, Any],
        vit_threshold: float = 0.6,
        cooldown_sec: float = 15.0,
        max_queue_size: int = 8,
    ):
        self.engine = engine
        self.state_ref = state_ref
        self.vit_threshold = float(vit_threshold)
        self.cooldown_sec = float(cooldown_sec)
        self._last_clip: Dict[str, List[np.ndarray]] = {}
        self._last_fire_ts: Dict[str, float] = {}
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def set_last_clip(self, stream_id: str, clip_frames_rgb: List[np.ndarray]) -> None:
        """Pipeline 在组成与 ViT 相同的 clip 后调用，用于后续触发 Agent 时提交。"""
        with self._lock:
            self._last_clip[stream_id] = [f.copy() for f in clip_frames_rgb]

    def try_submit_if_triggered(
        self,
        stream_id: str,
        vit_result: Optional[Any],
        vit_threshold_override: Optional[float] = None,
    ) -> bool:
        """
        若 ViT 结果为异常且置信度 >= 阈值、冷却已过、且有 last_clip，则提交到 Agent 队列。
        vit_result 可为 dict(pred_label, pred_prob, ranking_score) 或 VitEventResult。
        返回是否已提交。
        """
        threshold = vit_threshold_override if vit_threshold_override is not None else self.vit_threshold
        if vit_result is None:
            return False
        if hasattr(vit_result, "pred_label"):
            pred_label = getattr(vit_result, "pred_label", "")
            pred_prob = getattr(vit_result, "pred_prob", 0.0)
        else:
            pred_label = (vit_result.get("pred_label") or "").strip()
            pred_prob = float(vit_result.get("pred_prob") or 0.0)
        if pred_label.lower() == "normal" or pred_prob < threshold:
            return False
        now = time.time()
        with self._lock:
            if now - self._last_fire_ts.get(stream_id, 0) < self.cooldown_sec:
                return False
            clip = self._last_clip.get(stream_id)
            if not clip:
                return False
            try:
                self._queue.put_nowait((stream_id, clip))
                self._last_fire_ts[stream_id] = now
                return True
            except queue.Full:
                return False

    def _worker_loop(self) -> None:
        while True:
            try:
                stream_id, clip_frames = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                result = self.engine.analyze_clip(clip_frames)
            except Exception as e:
                result = {
                    "is_anomaly": False,
                    "confidence": 0.0,
                    "reasoning": str(e),
                    "error": True,
                }
            with self._lock:
                if stream_id not in self.state_ref:
                    self.state_ref[stream_id] = {}
                self.state_ref[stream_id]["agent_result"] = result
