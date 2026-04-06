"""
可选帧预处理（resize / normalize）。
端到端训练与推理优先使用 HuggingFace VideoMAEImageProcessor，多数情况无需本模块。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class VideoTransformConfig:
    image_size: int = 224
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


def resize_frames(frames: Sequence[np.ndarray], image_size: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for f in frames:
        out.append(cv2.resize(f, (image_size, image_size), interpolation=cv2.INTER_LINEAR))
    return out


def frames_to_torch(frames: Sequence[np.ndarray]) -> torch.Tensor:
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    arr = np.transpose(arr, (0, 3, 1, 2))
    return torch.from_numpy(arr)


def normalize_video(t: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=t.dtype, device=t.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, dtype=t.dtype, device=t.device).view(1, -1, 1, 1)
    return (t - mean_t) / std_t
