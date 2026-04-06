"""
维护每路最新帧和状态，供 MJPEG 与 API 读取。
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np


class FrameStateCache:
    """线程安全：每路 stream_id 对应最新 JPEG 帧、状态、时间戳。"""

    def __init__(self):
        self._lock = threading.RLock()
        self._jpeg: Dict[str, bytes] = {}
        self._ts: Dict[str, float] = {}
        self._status: Dict[str, dict] = {}

    def update(
        self,
        stream_id: str,
        frame_bgr: np.ndarray,
        ts: float,
        state: Optional[Dict[str, Any]] = None,
    ) -> None:
        _, buf = cv2.imencode(".jpg", frame_bgr)
        with self._lock:
            self._jpeg[stream_id] = buf.tobytes()
            self._ts[stream_id] = ts
            if state and stream_id in state:
                st = state[stream_id]
                self._status[stream_id] = {
                    "running": True,
                    "last_frame_ts": ts,
                    "vit_result": st.get("vit_event"),
                    "agent_result": st.get("agent_result"),
                    "error_message": None,
                }
            else:
                self._status[stream_id] = self._status.get(stream_id) or {}
                self._status[stream_id]["running"] = True
                self._status[stream_id]["last_frame_ts"] = ts

    def get_jpeg(self, stream_id: str) -> Optional[bytes]:
        with self._lock:
            return self._jpeg.get(stream_id)

    def get_status(self, stream_id: str) -> dict:
        with self._lock:
            return dict(self._status.get(stream_id, {}))

    def get_all_status(self) -> Dict[str, dict]:
        with self._lock:
            return {k: dict(v) for k, v in self._status.items()}

    def set_error(self, stream_id: str, message: str) -> None:
        with self._lock:
            if stream_id not in self._status:
                self._status[stream_id] = {}
            self._status[stream_id]["error_message"] = message
            self._status[stream_id]["running"] = False

    def set_stopped(self, stream_id: str) -> None:
        with self._lock:
            if stream_id not in self._status:
                self._status[stream_id] = {}
            self._status[stream_id]["running"] = False

    def remove(self, stream_id: str) -> None:
        with self._lock:
            self._jpeg.pop(stream_id, None)
            self._ts.pop(stream_id, None)
            self._status.pop(stream_id, None)

    def stream_ids(self) -> list:
        with self._lock:
            return list(self._jpeg.keys())


# 全局单例，供 pipeline frame_sink 与 MJPEG 路由使用
_frame_state: Optional[FrameStateCache] = None


def get_frame_state() -> FrameStateCache:
    global _frame_state
    if _frame_state is None:
        _frame_state = FrameStateCache()
    return _frame_state
