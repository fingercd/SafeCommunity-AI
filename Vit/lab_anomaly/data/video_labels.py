"""
视频标签 CSV 读写：video_labels.csv 定义视频路径、label（normal/异常类）、可选时间范围等。
"""
from __future__ import annotations

import csv
import dataclasses
from pathlib import Path
from typing import Iterable, Optional


CSV_HEADER = [
    "video_id",  # 唯一ID（建议：相对路径或 hash）
    "video_path",  # 视频路径（建议相对 lab_dataset 目录）
    "label",  # 类别：normal 或某个已知异常类别名（字符串）
    "camera_id",  # 摄像头/场景 ID（可空）
    "start_time",  # 可选：起始时间（秒，浮点/整数）
    "end_time",  # 可选：结束时间（秒，浮点/整数）
    "note",  # 可选：备注
]


@dataclasses.dataclass(frozen=True)
class VideoLabelRow:
    """CSV 单行对应的视频标签记录"""
    video_id: str
    video_path: str
    label: str
    camera_id: str = ""
    start_time: str = ""
    end_time: str = ""
    note: str = ""

    @staticmethod
    def from_dict(d: dict[str, str]) -> "VideoLabelRow":
        """从 CSV 行 dict 构造，缺列补空字符串"""
        # 允许缺列，统一补空
        def g(k: str) -> str:
            v = d.get(k, "")
            return "" if v is None else str(v)

        return VideoLabelRow(
            video_id=g("video_id"),
            video_path=g("video_path"),
            label=g("label"),
            camera_id=g("camera_id"),
            start_time=g("start_time"),
            end_time=g("end_time"),
            note=g("note"),
        )

    def to_dict(self) -> dict[str, str]:
        return {k: getattr(self, k) for k in CSV_HEADER}


def read_video_labels_csv(csv_path: str | Path) -> list[VideoLabelRow]:
    """读取 video_labels.csv，返回 VideoLabelRow 列表"""
    csv_path = Path(csv_path)
    rows: list[VideoLabelRow] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(VideoLabelRow.from_dict(r))
    return rows


def write_video_labels_csv(csv_path: str | Path, rows: Iterable[VideoLabelRow]) -> None:
    """将 VideoLabelRow 列表写回 CSV"""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def make_video_id_from_relpath(relpath: str) -> str:
    """用相对路径作为 video_id，保证稳定可复现"""
    # 默认用相对路径作为 ID，保证稳定可复现
    return relpath.replace("\\", "/")


def guess_camera_id_from_path(video_path: str) -> str:
    """从路径中推测 camera_id（约定 raw_videos/{label}/{camera_id}/... 时取二级目录）"""
    # 约定：raw_videos/一级目录/二级目录/xxx.mp4 时 camera_id=二级目录（如 normal/01 -> 01）
    # 若只有 raw_videos/一级目录/xxx.mp4 则返回空
    p = Path(video_path)
    parts = [x for x in p.parts if x]
    for folder in ("raw_videos", "raw_video"):
        try:
            raw_idx = parts.index(folder)
            if raw_idx + 2 < len(parts):
                return parts[raw_idx + 2]
            return ""
        except ValueError:
            continue
    return ""

