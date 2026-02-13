"""
视频读取：获取元信息、按帧索引读取帧、均匀 clip 采样索引。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    """视频元信息：路径、fps、帧数、宽高"""
    path: str
    fps: float
    frame_count: int
    width: int
    height: int


def get_video_info(video_path: str | Path) -> VideoInfo:
    """获取视频 fps、帧数、宽高"""
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-3:
        fps = 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return VideoInfo(path=video_path, fps=fps, frame_count=frame_count, width=width, height=height)


def read_frames_by_indices_cv2(
    video_path: str | Path,
    indices: list[int],
    to_rgb: bool = True,
) -> list[np.ndarray]:
    """
    以随机 seek 的方式读取指定帧索引。
    - 返回：list[np.ndarray(H,W,3)]，dtype=uint8
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(max(0, int(idx))))
        ok, frame = cap.read()
        if not ok or frame is None:
            # 读取失败时用最后一帧补齐（或黑帧）
            if frames:
                frame = frames[-1].copy()
            else:
                # 尝试读第一帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
                ok2, frame2 = cap.read()
                if ok2 and frame2 is not None:
                    frame = frame2
                else:
                    frame = np.zeros((224, 224, 3), dtype=np.uint8)
        if to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def uniform_clip_indices(
    frame_count: int,
    clip_len: int,
    frame_stride: int,
    clip_idx: int,
    num_clips: int,
) -> tuple[list[int], int, int]:
    """
    在 [0, frame_count) 内均匀取第 clip_idx 个 clip 的帧索引。

    返回：(indices, start_frame, end_frame_exclusive)
    """
    frame_count = max(0, int(frame_count))
    clip_len = int(clip_len)
    frame_stride = int(frame_stride)
    assert clip_len > 0 and frame_stride > 0
    assert num_clips > 0 and 0 <= clip_idx < num_clips

    clip_span = (clip_len - 1) * frame_stride + 1
    if frame_count <= 0:
        start = 0
    elif frame_count <= clip_span:
        start = 0
    else:
        max_start = frame_count - clip_span
        # 均匀分布到 [0, max_start]
        if num_clips == 1:
            start = max_start // 2
        else:
            start = int(round(clip_idx * max_start / (num_clips - 1)))

    indices = [start + i * frame_stride for i in range(clip_len)]
    # clamp
    indices = [min(max(0, x), max(0, frame_count - 1)) for x in indices]
    end_excl = start + clip_span
    return indices, int(start), int(end_excl)

