"""
When ViT detects anomaly above threshold, queue the same clip for VLM review.
Results are stored internally and polled by runtime_manager into pipeline state.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np


class VlmReviewRuntime:
    """
    Maintains per-stream last_clip; when ViT triggers, submits to review queue.
    Worker thread calls VLMEngine.analyze_clip; results stored in _results dict
    for the runtime_manager to poll and write into pipeline StreamState.
    """

    def __init__(
        self,
        engine: Any,
        vit_threshold: float = 0.6,
        cooldown_sec: float = 15.0,
        max_queue_size: int = 8,
    ):
        self.engine = engine
        self.vit_threshold = float(vit_threshold)
        self.cooldown_sec = float(cooldown_sec)
        self._last_clip: Dict[str, List[np.ndarray]] = {}
        self._last_fire_ts: Dict[str, float] = {}
        self._results: Dict[str, Dict[str, Any]] = {}
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def set_last_clip(self, stream_id: str, clip_frames_rgb: List[np.ndarray]) -> None:
        with self._lock:
            self._last_clip[stream_id] = [f.copy() for f in clip_frames_rgb]

    def try_submit_if_triggered(
        self,
        stream_id: str,
        vit_result: Optional[Any],
        vit_threshold_override: Optional[float] = None,
    ) -> bool:
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

    def submit_periodic(self, stream_id: str, clip_frames_rgb: List[np.ndarray]) -> bool:
        """Directly queue a clip for VLM review, bypassing ViT judgment (periodic trigger)."""
        try:
            self._queue.put_nowait((stream_id, [f.copy() for f in clip_frames_rgb]))
            return True
        except queue.Full:
            return False

    def pop_result(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Pop and return the latest VLM result for a stream, or None."""
        with self._lock:
            return self._results.pop(stream_id, None)

    def get_result(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Peek at the latest VLM result without removing it."""
        with self._lock:
            return self._results.get(stream_id)

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
                self._results[stream_id] = result
