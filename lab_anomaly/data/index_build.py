"""
扫描 lab_dataset/raw_videos 下的视频，生成/更新 video_labels.csv。

标签规则：raw_videos 下的一级文件夹名作为 label（normal、Abuse、Arrest 等）；
若一级文件夹下还有子文件夹（如 normal/01/、normal/02/），则二级文件夹名作为 camera_id。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from lab_anomaly.data.video_labels import (
    VideoLabelRow,
    make_video_id_from_relpath,
    read_video_labels_csv,
    write_video_labels_csv,
)


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def scan_videos(root_dir: Path) -> list[Path]:
    """递归扫描目录下所有视频文件（mp4/avi/mov/mkv/webm/m4v）"""
    videos: list[Path] = []
    for base, _, files in os.walk(root_dir):
        for fn in files:
            p = Path(base) / fn
            if p.suffix.lower() in VIDEO_EXTS:
                videos.append(p)
    videos.sort()
    return videos


def label_and_camera_from_path(video_path: Path, videos_root: Path, default_label: str = "unknown") -> tuple[str, str]:
    """
    根据视频路径相对 videos_root 的层级得到 label 和 camera_id。
    - label = 一级文件夹名（raw_videos 下的第一层目录：normal、Abuse、Arrest 等）
    - camera_id = 二级文件夹名（若存在，如 normal/01 中的 01）
    """
    try:
        rel = video_path.relative_to(videos_root)
    except ValueError:
        return default_label, ""
    parts = rel.parts
    if len(parts) == 0:
        return default_label, ""
    label = parts[0]
    camera_id = parts[1] if len(parts) > 1 else ""
    return label, camera_id


def build_or_update_csv(
    dataset_root: Path,
    videos_root: Path,
    out_csv: Path,
    default_label: str,
) -> None:
    """扫描 videos_root 下视频，按文件夹结构生成 label/camera_id，写入 out_csv。"""
    dataset_root = dataset_root.resolve()
    videos_root = videos_root.resolve()
    out_csv = out_csv.resolve()

    if not videos_root.is_dir():
        raise FileNotFoundError(
            f"视频目录不存在或不是文件夹，请检查路径（工作目录不同会导致相对路径错误）：\n  {videos_root}"
        )

    existing = {}
    if out_csv.exists():
        for r in read_video_labels_csv(out_csv):
            existing[r.video_id] = r

    rows: list[VideoLabelRow] = []
    videos = scan_videos(videos_root)
    if not videos and not existing:
        raise FileNotFoundError(
            f"在以下目录未找到任何视频文件（支持 .mp4/.avi/.mov/.mkv/.webm/.m4v）：\n  {videos_root}\n"
            "请确认：1) 路径正确（脚本已按项目根解析相对路径）；2) 该目录下确有视频文件。"
        )
    for vp in videos:
        try:
            rel = vp.relative_to(dataset_root).as_posix()
        except ValueError:
            rel = str(vp)
        video_id = make_video_id_from_relpath(rel)

        # 一律按当前路径的文件夹结构生成 label 和 camera_id（不沿用旧 CSV）
        label, camera_id = label_and_camera_from_path(vp, videos_root, default_label)

        prev = existing.get(video_id)
        rows.append(
            VideoLabelRow(
                video_id=video_id,
                video_path=rel,
                label=label,
                camera_id=camera_id,
                start_time=prev.start_time if prev else "",
                end_time=prev.end_time if prev else "",
                note=prev.note if prev else "",
            )
        )

    # 保留 CSV 中扫描不到但用户保留的条目
    for vid, r in existing.items():
        if vid not in {x.video_id for x in rows}:
            rows.append(r)

    write_video_labels_csv(out_csv, rows)
    labels_seen = sorted({r.label for r in rows})
    print(f"[OK] wrote {len(rows)} rows to: {out_csv}")
    print(f"[OK] labels from folder names: {labels_seen}")


def main() -> None:
    ap = argparse.ArgumentParser(description="扫描 lab_dataset 并生成/更新 video_labels.csv（视频级标签）")
    ap.add_argument(
        "--dataset_root",
        type=str,
        default=str(Path("lab_dataset")),
        help="数据集根目录（建议：lab_dataset）",
    )
    ap.add_argument(
        "--videos_root",
        type=str,
        default=str(Path("lab_dataset") / "raw_videos"),
        help="原始视频目录（默认：lab_dataset/raw_videos）",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default=str(Path("lab_dataset") / "labels" / "video_labels.csv"),
        help="输出 CSV 路径",
    )
    ap.add_argument(
        "--default_label",
        type=str,
        default="unknown",
        help="新扫描到的视频默认 label",
    )
    args = ap.parse_args()

    # 相对路径按项目根解析，避免从 PyCharm 非项目根运行时找不到 lab_dataset
    _project_root = Path(__file__).resolve().parent.parent.parent
    dataset_root = Path(args.dataset_root)
    videos_root = Path(args.videos_root)
    out_csv = Path(args.out_csv)
    if not dataset_root.is_absolute():
        dataset_root = _project_root / dataset_root
    if not videos_root.is_absolute():
        videos_root = _project_root / videos_root
    if not out_csv.is_absolute():
        out_csv = _project_root / out_csv

    build_or_update_csv(
        dataset_root=dataset_root,
        videos_root=videos_root,
        out_csv=out_csv,
        default_label=args.default_label,
    )


if __name__ == "__main__":
    main()

