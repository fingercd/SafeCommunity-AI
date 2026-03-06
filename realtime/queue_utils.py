"""
Infer queue: non-blocking put with drop_old policy (drop oldest frame when full).
EndOfStream events are never dropped.
"""
import queue
from typing import Union
from .events import FrameEvent, EndOfStreamEvent, EventTag

PipelineEvent = Union[FrameEvent, EndOfStreamEvent]


def put_event_nonblocking(
    q: queue.Queue,
    event: PipelineEvent,
    drop_policy: str = "drop_old",
) -> bool:
    """
    Put event into queue. If full and drop_policy=="drop_old", discard oldest
    FRAME event to make room; never drop EndOfStreamEvent.
    Returns True if event was enqueued, False if dropped (only frame can be dropped).
    """
    try:
        q.put_nowait(event)
        return True
    except queue.Full:
        if event.tag == EventTag.END_OF_STREAM:
            # Force make room: remove one frame
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            q.put_nowait(event)
            return True
        if drop_policy != "drop_old":
            return False
        old_to_restore = None
        try:
            old = q.get_nowait()
            if old.tag == EventTag.END_OF_STREAM:
                q.put_nowait(old)
                return False
            old_to_restore = old
        except queue.Empty:
            pass
        try:
            q.put_nowait(event)
            return True
        except queue.Full:
            if old_to_restore is not None:
                try:
                    q.put_nowait(old_to_restore)
                except Exception:
                    pass
            return False
