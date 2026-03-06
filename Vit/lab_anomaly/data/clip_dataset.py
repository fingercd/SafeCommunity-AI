"""
视频 Clip 数据集：以 video_labels.csv 为源，按 clip_len / frame_stride 均匀采样 clip，
每个 clip 为一组连续帧，供 ViT 编码或 embedding 提取使用。
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from lab_anomaly.data.video_labels import VideoLabelRow, read_video_labels_csv
from lab_anomaly.data.video_reader import get_video_info, read_frames_by_indices_cv2, uniform_clip_indices


@dataclass(frozen=True)
class ClipSample:
    """单个 clip 采样结果：视频元信息 + 帧列表"""
    video_id: str
    video_path: str
    label: str
    clip_index: int
    start_frame: int
    end_frame_exclusive: int
    frames: list[np.ndarray]  # list of RGB HWC uint8


def _parse_float_or_none(x: str) -> Optional[float]:
    """解析秒数等浮点值"""
    x = "" if x is None else str(x).strip()
    if not x:
        return None
    try:
        return float(x)
    except ValueError:
        return None


class VideoClipDataset:
    """
    以 video_labels.csv 为源，按视频均匀采样 N 个 clips（每个 clip_len 帧，步长 frame_stride）。

    这里返回原始 RGB uint8 帧列表，便于：
    - 交给 HF processor（推荐）
    - 或使用本仓库 transforms 做 resize/normalize

    额外支持：当 CSV 中提供 start_time/end_time（秒）时，只在该时间段对应的帧区间内采样。
    """

    def __init__(
        self,
        dataset_root: str | Path,
        labels_csv: str | Path,
        clip_len: int = 16,
        frame_stride: int = 2,
        num_clips_per_video: int = 8,
        shuffle_clips: bool = False,
        video_filter: Optional[Callable[[VideoLabelRow], bool]] = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.labels_csv = Path(labels_csv)
        self.clip_len = int(clip_len)
        self.frame_stride = int(frame_stride)
        self.num_clips_per_video = int(num_clips_per_video)
        self.shuffle_clips = bool(shuffle_clips)

        rows = read_video_labels_csv(self.labels_csv)
        if video_filter is not None:
            rows = [r for r in rows if video_filter(r)]
        self.rows = rows

        # 预构建 (video_idx, clip_idx)
        pairs = []
        for vi, _ in enumerate(self.rows):
            for ci in range(self.num_clips_per_video):
                pairs.append((vi, ci))
        if self.shuffle_clips:
            random.shuffle(pairs)
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def _abs_video_path(self, video_path: str) -> Path:
        """将相对路径转为基于 dataset_root 的绝对路径"""
        p = Path(video_path)
        if p.is_absolute():
            return p
        return (self.dataset_root / p).resolve()

    def _time_range_to_frame_range(self, row: VideoLabelRow, fps: float, frame_count: int) -> tuple[int, int]:
        """
        把 row.start_time/end_time（秒）转换为 [start_frame, end_frame_exclusive)。
        - 若字段缺失/非法，则退化为全视频范围。
        """
        frame_count = max(0, int(frame_count))
        if frame_count <= 0:
            return 0, 0

        st = _parse_float_or_none(row.start_time)
        et = _parse_float_or_none(row.end_time)
        if st is None and et is None:
            return 0, frame_count

        fps = float(fps or 0.0)
        if fps <= 1e-3:
            fps = 25.0

        start = 0 if st is None else int(max(0.0, st) * fps)
        end_excl = frame_count if et is None else int(max(0.0, et) * fps)
        start = max(0, min(start, frame_count))
        end_excl = max(0, min(end_excl, frame_count))
        if end_excl <= start:
            return 0, frame_count
        return start, end_excl

    def get_video_clips(self, video_idx: int) -> list[ClipSample]:
        """
        返回指定视频的全部 clips（长度 = num_clips_per_video）。
        适合“每视频输出一个 embedding 文件”的缓存模式。
        """
        row = self.rows[int(video_idx)]
        abs_path = self._abs_video_path(row.video_path)
        info = get_video_info(abs_path)
        range_start, range_end_excl = self._time_range_to_frame_range(row, fps=info.fps, frame_count=info.frame_count)
        span = max(0, range_end_excl - range_start)

        clips: list[ClipSample] = []
        for ci in range(self.num_clips_per_video):
            indices_local, start_local, end_local_excl = uniform_clip_indices(
                frame_count=span,
                clip_len=self.clip_len,
                frame_stride=self.frame_stride,
                clip_idx=ci,
                num_clips=self.num_clips_per_video,
            )
            indices = [range_start + x for x in indices_local]
            frames = read_frames_by_indices_cv2(abs_path, indices=indices, to_rgb=True)
            clips.append(
                ClipSample(
                    video_id=row.video_id,
                    video_path=row.video_path,
                    label=row.label,
                    clip_index=ci,
                    start_frame=range_start + int(start_local),
                    end_frame_exclusive=range_start + int(end_local_excl),
                    frames=frames,
                )
            )
        return clips

    def __getitem__(self, idx: int) -> ClipSample:
        vi, ci = self.pairs[idx]
        row = self.rows[vi]
        abs_path = self._abs_video_path(row.video_path)
        info = get_video_info(abs_path)
        range_start, range_end_excl = self._time_range_to_frame_range(row, fps=info.fps, frame_count=info.frame_count)
        span = max(0, range_end_excl - range_start)
        indices_local, start_local, end_local_excl = uniform_clip_indices(
            frame_count=span,
            clip_len=self.clip_len,
            frame_stride=self.frame_stride,
            clip_idx=ci,
            num_clips=self.num_clips_per_video,
        )
        indices = [range_start + x for x in indices_local]
        frames = read_frames_by_indices_cv2(abs_path, indices=indices, to_rgb=True)
        return ClipSample(
            video_id=row.video_id,
            video_path=row.video_path,
            label=row.label,
            clip_index=ci,
            start_frame=range_start + int(start_local),
            end_frame_exclusive=range_start + int(end_local_excl),
            frames=frames,
        )

