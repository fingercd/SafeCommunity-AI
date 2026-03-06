"""
File video capture_loop: read until EOF, then push EndOfStream(stream_id) and exit.
Optional frame rate control for playback.
"""
import time
import threading
from typing import Callable, Optional

import cv2

from .config import SourceConfig
from .events import FrameEvent, EndOfStreamEvent
from .queue_utils import put_event_nonblocking


def open_capture(uri: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(uri)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def capture_loop(
    source: SourceConfig,
    infer_queue,
    drop_policy: str = "drop_old",
    stop_event: Optional[threading.Event] = None,
    target_fps: Optional[float] = None,
) -> None:
    """
    Run in a dedicated thread. Reads frames from file; on EOF pushes EndOfStreamEvent(stream_id) and breaks.
    target_fps: if set, sleep to approximate playback rate (optional).
    """
    stop = stop_event or threading.Event()
    cap = open_capture(source.uri)
    last_ts = None

    while not stop.is_set():
        ok, frame = cap.read()

        if not ok:
            put_event_nonblocking(infer_queue, EndOfStreamEvent(stream_id=source.uri), drop_policy=drop_policy)
            break

        ts = time.monotonic()
        ev = FrameEvent(stream_id=source.uri, frame=frame.copy(), ts=ts)
        put_event_nonblocking(infer_queue, ev, drop_policy=drop_policy)

        if target_fps and target_fps > 0:
            if last_ts is not None:
                elapsed = ts - last_ts
                need = 1.0 / target_fps
                if elapsed < need:
                    time.sleep(need - elapsed)
            last_ts = time.monotonic()

    cap.release()
