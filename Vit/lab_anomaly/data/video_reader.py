"""
视频读取：获取元信息、按帧索引读取帧、按时长切分 clip 并在每段内均匀采样帧索引。
"""
from __future__ import annotations

import math
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
            if frames:
                frame = frames[-1].copy()
            else:
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


def compute_num_clips(
    duration_sec: float,
    interval_sec: float = 8.0,
    max_clips: int = 0,
) -> int:
    """
    按时长决定 clip 个数：0~8s 为 1 个，8~16s 为 2 个，以此类推。
    duration_sec <= 0 时返回 0。
    max_clips > 0 时上限为该值（训练时控显存）。
    """
    if duration_sec <= 0:
        return 0
    n = int(math.ceil(float(duration_sec) / float(interval_sec)))
    n = max(1, n)
    if max_clips and int(max_clips) > 0:
        n = min(n, int(max_clips))
    return n


def count_clips_for_span(
    span_frames: int,
    fps: float,
    interval_sec: float = 8.0,
    max_clips: int = 0,
) -> int:
    """由有效帧区间长度与 fps 计算 clip 数量（与 compute_num_clips 一致）。"""
    span_frames = max(0, int(span_frames))
    if span_frames <= 0:
        return 0
    fps_eff = float(fps) if float(fps) > 1e-6 else 25.0
    duration_sec = span_frames / fps_eff
    return compute_num_clips(duration_sec, interval_sec, max_clips)


def _uniform_indices_in_closed_range(lo: int, hi_inclusive: int, n_samples: int) -> list[int]:
    """在 [lo, hi_inclusive] 上均匀取 n_samples 个整数帧索引（含端点）。"""
    lo, hi_inclusive = int(lo), int(hi_inclusive)
    n_samples = int(n_samples)
    if n_samples <= 0:
        return []
    if hi_inclusive < lo:
        return [lo] * n_samples
    if n_samples == 1:
        return [lo]
    span = hi_inclusive - lo
    out: list[int] = []
    for i in range(n_samples):
        x = lo + int(round(i * span / max(n_samples - 1, 1)))
        x = min(max(x, lo), hi_inclusive)
        out.append(x)
    return out


def duration_based_clip_specs(
    range_start: int,
    span_frames: int,
    fps: float,
    *,
    frames_per_clip: int = 16,
    interval_sec: float = 8.0,
    max_clips: int = 0,
) -> list[tuple[list[int], int, int]]:
    """
    按时长将 [range_start, range_start+span) 切成多个时间段，每段内均匀采样 frames_per_clip 帧。

    返回：[(abs_indices, segment_start_frame, segment_end_exclusive), ...]
    """
    span_frames = max(0, int(span_frames))
    range_start = int(range_start)
    frames_per_clip = max(1, int(frames_per_clip))
    if span_frames <= 0:
        return []

    fps_eff = float(fps) if float(fps) > 1e-6 else 25.0
    duration_sec = span_frames / fps_eff
    num_clips = compute_num_clips(duration_sec, interval_sec, max_clips)
    if num_clips <= 0:
        return []

    specs: list[tuple[list[int], int, int]] = []
    for ci in range(num_clips):
        seg_lo = int(ci * span_frames / num_clips)
        seg_hi_excl = int((ci + 1) * span_frames / num_clips)
        if seg_hi_excl <= seg_lo:
            seg_hi_excl = min(seg_lo + 1, span_frames)
        hi_inclusive = range_start + seg_hi_excl - 1
        lo = range_start + seg_lo
        if hi_inclusive < lo:
            hi_inclusive = lo
        abs_idx = _uniform_indices_in_closed_range(lo, hi_inclusive, frames_per_clip)
        specs.append((abs_idx, lo, range_start + seg_hi_excl))
    return specs
