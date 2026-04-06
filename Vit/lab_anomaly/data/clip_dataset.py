"""
视频 Clip 数据集：以 video_labels.csv 为源，按时长动态切 clip（每约 8s 一段），
每段内均匀采样 frames_per_clip 帧，供 VideoMAE 编码使用。

支持两种模式：
- 预切片（推荐训练）：preclip_root 指向离线目录（manifest.json），只读 .npz，**不打开原视频、不按视频重新算切片**。
- 现读现切：preclip_root 为空时，从原视频 decode（一般仅调试用；训练请先用 precompute_clips.py）。
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from lab_anomaly.data.preclip_manifest import (
    assert_params_match,
    build_params_blob,
    load_manifest,
    npz_load_frames_list,
)
from lab_anomaly.data.video_labels import VideoLabelRow, read_video_labels_csv
from lab_anomaly.data.video_reader import (
    duration_based_clip_specs,
    get_video_info,
    read_frames_by_indices_cv2,
)


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


def _project_root_dir() -> Path:
    """
    项目根目录：同时包含 lab_anomaly 和 lab_dataset 的那一层。
    相对路径统一基于这里解析，不依赖 PyCharm 的工作目录。
    """
    return Path(__file__).resolve().parent.parent.parent


def _resolve_path_under_project(path_value: str | Path) -> Path:
    path_value = Path(path_value)
    if path_value.is_absolute():
        return path_value
    return (_project_root_dir() / path_value).resolve()


def _parse_float_or_none(x: str) -> Optional[float]:
    """解析秒数等浮点值"""
    x = "" if x is None else str(x).strip()
    if not x:
        return None
    try:
        return float(x)
    except ValueError:
        return None


def time_range_to_frame_range(row: VideoLabelRow, fps: float, frame_count: int) -> tuple[int, int]:
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


def clip_specs_for_row(
    row: VideoLabelRow,
    dataset_root: Path,
    *,
    frames_per_clip: int,
    interval_sec: float,
    max_clips_per_video: int,
) -> tuple[list[tuple[list[int], int, int]], Optional[Path]]:
    """
    与 VideoClipDataset 内部一致的切片规划；供预切片 worker 复用。
    返回 (specs, abs_video_path)；无法打开视频时 specs 为空，abs_path 仍可能返回。
    """
    dataset_root = Path(dataset_root)
    p = Path(row.video_path)
    if p.is_absolute():
        abs_path = p.resolve()
    else:
        abs_path = (dataset_root / p).resolve()
    try:
        info = get_video_info(abs_path)
    except Exception:
        return [], abs_path

    range_start, range_end_excl = time_range_to_frame_range(row, info.fps, info.frame_count)
    span = max(0, range_end_excl - range_start)
    specs = duration_based_clip_specs(
        range_start,
        span,
        info.fps,
        frames_per_clip=int(frames_per_clip),
        interval_sec=float(interval_sec),
        max_clips=int(max_clips_per_video),
    )
    return specs, abs_path


def _abs_video_path(row: VideoLabelRow, dataset_root: Path) -> Path:
    p = Path(row.video_path)
    if p.is_absolute():
        return p.resolve()
    return (Path(dataset_root) / p).resolve()


class VideoClipDataset:
    """
    以 video_labels.csv 为源：
    - 若提供 preclip_root：clip 列表完全由 manifest 的 rel_paths 决定，只从 .npz 读帧（训练推荐，不再读原视频）。
    - 否则：按视频时长算 clip 数，从原视频 decode（仅调试/未预切片时使用）。
    返回原始 RGB uint8 帧列表，交给 HF VideoMAE processor。
    """

    def __init__(
        self,
        dataset_root: str | Path,
        labels_csv: str | Path,
        frames_per_clip: int = 16,
        interval_sec: float = 8.0,
        max_clips_per_video: int = 16,
        shuffle_clips: bool = False,
        video_filter: Optional[Callable[[VideoLabelRow], bool]] = None,
        preclip_root: str | Path | None = None,
        exclude_unknown_for_manifest: bool = False,
        normal_label_for_manifest: str = "normal",
    ):
        self.dataset_root = _resolve_path_under_project(dataset_root)
        self.labels_csv = _resolve_path_under_project(labels_csv)
        self.frames_per_clip = int(frames_per_clip)
        self.interval_sec = float(interval_sec)
        self.max_clips_per_video = int(max_clips_per_video)
        self.shuffle_clips = bool(shuffle_clips)

        rows = read_video_labels_csv(self.labels_csv)
        if video_filter is not None:
            rows = [r for r in rows if video_filter(r)]
        self.rows = rows

        self._preclip_root: Optional[Path] = None
        self._preclip_rel_paths: Optional[list[list[str]]] = None

        self._clip_specs_per_video: list[list[tuple[list[int], int, int]]] = []
        pairs: list[tuple[int, int]] = []

        use_preclip = preclip_root is not None and str(preclip_root).strip() != ""

        if use_preclip:
            self._preclip_root = Path(preclip_root).resolve()
            manifest = load_manifest(self._preclip_root)
            mp = manifest["params"]
            expected = build_params_blob(
                labels_csv=self.labels_csv,
                dataset_root=self.dataset_root,
                frames_per_clip=self.frames_per_clip,
                interval_sec=self.interval_sec,
                max_clips_per_video=self.max_clips_per_video,
                exclude_unknown=bool(exclude_unknown_for_manifest),
                normal_label=str(normal_label_for_manifest),
            )
            assert_params_match(mp, expected)

            vids = manifest["videos"]
            if len(vids) != len(self.rows):
                raise RuntimeError(
                    f"预切片 manifest 视频条数 {len(vids)} 与当前标签行数 {len(self.rows)} 不一致。"
                )
            # 仅用 manifest：clip 数量与顺序以 rel_paths 为准，不读原视频、不重新 duration 切片
            for vi, ent in enumerate(vids):
                rels = list(ent.get("rel_paths") or [])
                self._clip_specs_per_video.append([([], 0, 0) for _ in range(len(rels))])
                for ci in range(len(rels)):
                    pairs.append((vi, ci))
            self._preclip_rel_paths = [list(ent.get("rel_paths") or []) for ent in vids]
        else:
            for vi, row in enumerate(self.rows):
                specs, _ = clip_specs_for_row(
                    row,
                    self.dataset_root,
                    frames_per_clip=self.frames_per_clip,
                    interval_sec=self.interval_sec,
                    max_clips_per_video=self.max_clips_per_video,
                )
                self._clip_specs_per_video.append(specs)
                for ci in range(len(specs)):
                    pairs.append((vi, ci))

        if self.shuffle_clips:
            random.shuffle(pairs)
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def num_clips_for_video(self, video_idx: int) -> int:
        return len(self._clip_specs_per_video[int(video_idx)])

    def _decode_clip_frames(
        self,
        video_idx: int,
        clip_index: int,
        indices: list[int],
    ) -> list[np.ndarray]:
        if self._preclip_rel_paths is not None and self._preclip_root is not None:
            rel = self._preclip_rel_paths[int(video_idx)][int(clip_index)]
            full = self._preclip_root / rel
            return npz_load_frames_list(full)
        row = self.rows[int(video_idx)]
        abs_path = _abs_video_path(row, self.dataset_root)
        return read_frames_by_indices_cv2(abs_path, indices=indices, to_rgb=True)

    def get_video_clips(self, video_idx: int) -> list[ClipSample]:
        """返回指定视频的全部 clips（按时长切分后的全部段）。"""
        row = self.rows[int(video_idx)]
        specs = self._clip_specs_per_video[int(video_idx)]
        clips: list[ClipSample] = []
        for ci, (indices, start_f, end_excl) in enumerate(specs):
            frames = self._decode_clip_frames(int(video_idx), ci, indices)
            clips.append(
                ClipSample(
                    video_id=row.video_id,
                    video_path=row.video_path,
                    label=row.label,
                    clip_index=ci,
                    start_frame=start_f,
                    end_frame_exclusive=end_excl,
                    frames=frames,
                )
            )
        return clips

    def __getitem__(self, idx: int) -> ClipSample:
        vi, ci = self.pairs[idx]
        row = self.rows[vi]
        indices, start_f, end_excl = self._clip_specs_per_video[vi][ci]
        frames = self._decode_clip_frames(vi, ci, indices)
        return ClipSample(
            video_id=row.video_id,
            video_path=row.video_path,
            label=row.label,
            clip_index=ci,
            start_frame=start_f,
            end_frame_exclusive=end_excl,
            frames=frames,
        )
