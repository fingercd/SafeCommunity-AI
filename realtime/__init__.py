# Real-time multi-stream (RTSP/File) + ByteTrack + ROI dwell + alarm pipeline.
# Use stream_id = source.uri for per-stream isolation.

from .config import (
    SourceConfig,
    StreamConfig,
    TrackerConfig,
    SystemConfig,
    build_stream_config_map,
)
from .events import FrameEvent, EndOfStreamEvent, EventTag
from .types import DetectionResult, Track

__all__ = [
    "SourceConfig",
    "StreamConfig",
    "TrackerConfig",
    "SystemConfig",
    "build_stream_config_map",
    "FrameEvent",
    "EndOfStreamEvent",
    "EventTag",
    "DetectionResult",
    "Track",
]
