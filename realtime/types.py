"""
Detection and Track types for pipeline. YOLO output -> DetectionResult; ByteTrack output -> Track.
"""
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class DetectionResult:
    """Per-frame YOLO output: boxes (xyxy), class_id, score."""
    boxes_xyxy: np.ndarray   # shape (N, 4)
    class_id: np.ndarray     # shape (N,) int
    score: np.ndarray        # shape (N,) float

    @classmethod
    def empty(cls) -> "DetectionResult":
        return cls(
            boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
            class_id=np.zeros(0, dtype=np.int32),
            score=np.zeros(0, dtype=np.float32),
        )


@dataclass
class Track:
    """Single track from ByteTrack: id, bbox, score, class for ROI/alarm."""
    id: int
    bbox_xyxy: np.ndarray    # (4,) x1,y1,x2,y2
    score: float
    class_id: int
    class_name: str
