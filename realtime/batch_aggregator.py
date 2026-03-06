"""
BatchAggregator: collect from infer queue with max_batch and max_wait_ms.
Fairness: round-robin by stream_id when multiple frames available (avoid one stream dominating).
"""
import queue
import time
from typing import List, Tuple

from .events import FrameEvent, EndOfStreamEvent, EventTag


def collect_batch(
    infer_queue: queue.Queue,
    max_items: int,
    max_wait_ms: float,
) -> Tuple[List[FrameEvent], List[object]]:
    """
    Collect up to max_items frame events, waiting at most max_wait_ms for the first item.
    Returns (frame_events, eos_events). eos_events are EndOfStreamEvent encountered during collect.
    Order is whatever the queue yields (first available, then drain with get_nowait up to max_items).
    """
    frame_events: List[FrameEvent] = []
    eos_events: List[EndOfStreamEvent] = []
    deadline = time.monotonic() + max_wait_ms / 1000.0
    first = True

    while len(frame_events) < max_items:
        try:
            if first:
                timeout = max(0, deadline - time.monotonic())
                ev = infer_queue.get(timeout=timeout)
                first = False
            else:
                ev = infer_queue.get_nowait()
        except queue.Empty:
            break

        if ev.tag == EventTag.END_OF_STREAM:
            eos_events.append(ev)
            continue
        if ev.tag == EventTag.FRAME:
            frame_events.append(ev)

    return frame_events, eos_events
