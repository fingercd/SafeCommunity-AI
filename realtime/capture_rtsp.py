"""
RTSP capture_loop: continuous read, reconnect with backoff on failure,
monotonic timestamp, push to infer queue with drop_old policy.
"""
import time
import threading
from typing import Callable, Optional

import cv2
import numpy as np

from .config import SourceConfig, SystemConfig
from .events import FrameEvent
from .queue_utils import put_event_nonblocking


def open_capture(uri: str, prefer_ffmpeg: bool = True) -> cv2.VideoCapture:
    if prefer_ffmpeg:
        cap = cv2.VideoCapture(uri, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(uri)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def capture_loop(
    source: SourceConfig,
    infer_queue,
    sys_cfg: SystemConfig,
    stop_event: Optional[threading.Event] = None,
    on_error: Optional[Callable[[str, Exception], None]] = None,
) -> None:
    """
    Run in a dedicated thread. Reads frames from RTSP, pushes FrameEvent to infer_queue.
    On read failure: reconnect with backoff (rtsp_reconnect_delay_sec, rtsp_max_reconnect_attempts).
    Uses time.monotonic() for ts. drop_policy from sys_cfg.
    """
    stop = stop_event or threading.Event()
    delay = sys_cfg.rtsp_reconnect_delay_sec
    max_attempts = sys_cfg.rtsp_max_reconnect_attempts
    drop_policy = sys_cfg.drop_policy

    cap = open_capture(source.uri, prefer_ffmpeg=True)
    attempts = 0

    while not stop.is_set():
        ok, frame = cap.read()

        if not ok:
            cap.release()
            if stop.is_set():
                break
            attempts += 1
            if attempts > max_attempts:
                if on_error:
                    on_error(source.uri, RuntimeError("RTSP max reconnect attempts exceeded"))
                break
            time.sleep(delay)
            cap = open_capture(source.uri, prefer_ffmpeg=True)
            continue

        attempts = 0
        ts = time.monotonic()
        ev = FrameEvent(stream_id=source.uri, frame=frame.copy(), ts=ts)
        put_event_nonblocking(infer_queue, ev, drop_policy=drop_policy)

    cap.release()
