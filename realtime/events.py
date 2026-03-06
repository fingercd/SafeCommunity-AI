"""
Stream abstraction: frame event (stream_id, frame, ts) and EndOfStream event.
"""
from enum import Enum
from dataclasses import dataclass
from typing import Union
import numpy as np


class EventTag:
    FRAME = "frame"
    END_OF_STREAM = "end_of_stream"


@dataclass
class FrameEvent:
    """One frame from a source; stream_id = source.uri."""
    stream_id: str
    frame: np.ndarray   # HWC BGR (OpenCV)
    ts: float           # monotonic time (seconds)

    @property
    def tag(self) -> str:
        return EventTag.FRAME


@dataclass
class EndOfStreamEvent:
    """Emitted when a file source reaches EOF; triggers flush for that stream."""
    stream_id: str

    @property
    def tag(self) -> str:
        return EventTag.END_OF_STREAM


PipelineEvent = Union[FrameEvent, EndOfStreamEvent]
