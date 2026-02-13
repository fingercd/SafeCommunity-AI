"""
视频帧预处理：resize、转 tensor、ImageNet 风格 normalize。
供 ViT 前处理使用；若用 HF processor 则可能不直接调用。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class VideoTransformConfig:
    """视频预处理配置：图像尺寸、ImageNet 均值方差"""
    image_size: int = 224
    # 参考 ImageNet/ViT 常用均值方差；如使用 HF processor，会覆盖这里
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


def resize_frames(frames: Sequence[np.ndarray], image_size: int) -> list[np.ndarray]:
    """将帧 resize 为 image_size x image_size"""
    out: list[np.ndarray] = []
    for f in frames:
        out.append(cv2.resize(f, (image_size, image_size), interpolation=cv2.INTER_LINEAR))
    return out


def frames_to_torch(frames: Sequence[np.ndarray]) -> torch.Tensor:
    """
    frames: list of HWC RGB uint8
    returns: (T, C, H, W) float32 in [0,1]
    """
    arr = np.stack(frames, axis=0)  # (T,H,W,3)
    arr = arr.astype(np.float32) / 255.0
    arr = np.transpose(arr, (0, 3, 1, 2))  # (T,3,H,W)
    return torch.from_numpy(arr)


def normalize_video(t: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    """按 mean/std 做通道级归一化"""
    """
    t: (T,C,H,W)
    """
    mean_t = torch.tensor(mean, dtype=t.dtype, device=t.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, dtype=t.dtype, device=t.device).view(1, -1, 1, 1)
    return (t - mean_t) / std_t

