"""
将已落盘的 clip 帧（如 data/processed/ecva_clips）按单帧像素乘积上限等比例缩小并覆盖写回。
与 data/prepare_clips.resize_max_pixel_area 逻辑一致；不修改 train.json / 划分。

用法（工作目录 = 项目根 vlm）：
  python scripts/downscale_clip_frames.py
  python scripts/downscale_clip_frames.py --root data/processed/ecva_clips --max_pixel_area 518400 --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.prepare_clips import resize_max_pixel_area  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="将 clip 目录下单帧 JPG 的 w*h 限制在 max_pixel_area 内（大于则等比缩小至乘积严格小于上限）"
    )
    ap.add_argument(
        "--root",
        type=str,
        default="data/processed/ecva_clips",
        help="clips 根目录（其下每个子目录含 frame_XX.jpg）",
    )
    ap.add_argument(
        "--max_pixel_area",
        type=int,
        default=518400,
        help="像素乘积上限（默认 720*720），<=0 则退出",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="仅统计将缩放的文件数，不写盘",
    )
    ap.add_argument(
        "--jpeg_quality",
        type=int,
        default=95,
        help="cv2.imwrite JPEG 质量 0-100",
    )
    args = ap.parse_args()

    if args.max_pixel_area <= 0:
        print("max_pixel_area 须为正整数")
        raise SystemExit(2)

    root = Path(args.root)
    if not root.is_absolute():
        root = _PROJECT_ROOT / root
    if not root.is_dir():
        print(f"目录不存在: {root}")
        raise SystemExit(1)

    paths: list[Path] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        jpgs = sorted(sub.glob("frame_*.jpg"))
        if not jpgs:
            jpgs = sorted(sub.glob("*.jpg"))
        paths.extend(jpgs)

    if not paths:
        print(f"未找到 frame_*.jpg: {root}")
        raise SystemExit(0)

    n_change = 0
    n_skip = 0
    for p in tqdm(paths, desc="Frames", unit="file"):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            tqdm.write(f"[skip] 无法读取: {p}")
            continue
        h, w = img.shape[:2]
        if w * h <= args.max_pixel_area:
            n_skip += 1
            continue
        n_change += 1
        if args.dry_run:
            continue
        out = resize_max_pixel_area(img, args.max_pixel_area)
        cv2.imwrite(
            str(p),
            out,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
        )

    print(
        f"完成: 总文件 {len(paths)}, 需缩放 {n_change}, 已跳过(已够小) {n_skip}, "
        f"dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
