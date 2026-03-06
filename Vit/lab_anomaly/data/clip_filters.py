"""
SSIM + 光流幅值二级过滤：去除变化过低的 clips，供 extract_embeddings 在送 ViT 前使用。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def compute_ssim_variation(frames: list[np.ndarray]) -> float:
    """
    计算 clip 内帧间 SSIM 均值；值越大表示变化越小（越静态）。

    Args:
        frames: list of RGB HWC uint8，长度 >= 2

    Returns:
        mean_ssim: 若 > 0.95 则视为静态，建议丢弃
    """
    if len(frames) < 2:
        return 0.0
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError as e:
        raise RuntimeError("需要 scikit-image。请安装：pip install scikit-image") from e

    ssims: list[float] = []
    for i in range(len(frames) - 1):
        a = frames[i]
        b = frames[i + 1]
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        if a.ndim == 3:
            a_gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
            b_gray = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        else:
            a_gray, b_gray = a, b
        s = ssim(a_gray, b_gray, data_range=255)
        ssims.append(float(s))
    return float(np.mean(ssims))


def compute_optical_flow_magnitude(
    frames: list[np.ndarray],
    flows_npz_path: Optional[Path] = None,
    clip_index: int = 0,
    flows_array: Optional[np.ndarray] = None,
) -> float:
    """
    计算光流幅值均值 sqrt(u^2 + v^2)。
    优先从预计算缓存读取；若无缓存则用 Farneback 实时计算。

    Args:
        frames: list of RGB HWC uint8（用于无缓存时实时算光流）
        flows_npz_path: 预计算 flows 的 npz 路径，若存在则从中读 clip_index 对应 clip
        clip_index: 该 clip 在视频内的索引
        flows_array: 若提供则直接使用，避免重复加载 npz（形状 num_clips, T-1, 2, H, W）

    Returns:
        mean_magnitude: 若 < 2.0 则视为运动不足，建议丢弃
    """
    if flows_array is not None:
        if flows_array.ndim == 5 and clip_index < flows_array.shape[0]:
            clip_flow = flows_array[clip_index]
        else:
            clip_flow = flows_array[0] if flows_array.ndim == 5 else flows_array
        if clip_flow.shape[-3] == 2:
            u, v = clip_flow[:, 0, :, :], clip_flow[:, 1, :, :]
        else:
            u, v = clip_flow[..., 0], clip_flow[..., 1]
        mag = np.sqrt(u.astype(np.float64) ** 2 + v.astype(np.float64) ** 2)
        return float(np.mean(mag))
    if flows_npz_path is not None and Path(flows_npz_path).exists():
        try:
            npz = np.load(flows_npz_path, allow_pickle=True)
            flows = np.asarray(npz["flows"], dtype=np.float32)
            if flows.ndim == 5:
                # (num_clips, T-1, 2, H, W)
                if clip_index < flows.shape[0]:
                    clip_flow = flows[clip_index]
                else:
                    clip_flow = flows[0]
            else:
                clip_flow = flows
            # clip_flow: (T-1, 2, H, W) or (T-1, H, W, 2)
            if clip_flow.shape[-3] == 2:
                u, v = clip_flow[:, 0, :, :], clip_flow[:, 1, :, :]
            else:
                u, v = clip_flow[..., 0], clip_flow[..., 1]
            mag = np.sqrt(u.astype(np.float64) ** 2 + v.astype(np.float64) ** 2)
            return float(np.mean(mag))
        except Exception:
            pass

    if len(frames) < 2:
        return 0.0
    mags: list[float] = []
    for i in range(len(frames) - 1):
        gray_a = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray_a, gray_b, None,
            pyr_scale=0.5, levels=2, winsize=15,
            iterations=2, poly_n=5, poly_sigma=1.2, flags=0,
        )
        u, v = flow[..., 0], flow[..., 1]
        mag = np.sqrt(u.astype(np.float64) ** 2 + v.astype(np.float64) ** 2)
        mags.append(float(np.mean(mag)))
    return float(np.mean(mags)) if mags else 0.0
