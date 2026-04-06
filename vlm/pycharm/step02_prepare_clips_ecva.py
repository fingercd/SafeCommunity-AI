"""
步骤 2：按 ECVA 时间段切帧 + 从 ucf_normal_root 抽 Normal clip
PyCharm：工作目录 = 项目根 vlm。clip_len / max_pixel_area 由 configs/default.yaml 中 clips 段与 prepare_clips --config 一致。
若 decord 报错，将 USE_OPENCV_ONLY 改为 True。
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
_CFG = _ROOT / "configs" / "default.yaml"

# 解码失败时改为 True（强制 OpenCV）
USE_OPENCV_ONLY = False


def main() -> None:
    with _CFG.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    clips = cfg.get("clips") or {}
    clip_len = int(clips.get("clip_len", 16))

    script = _ROOT / "data" / "prepare_clips.py"
    argv = [
        str(script),
        "--mode",
        "ecva",
        "--config",
        str(_CFG),
        "--clip_len",
        str(clip_len),
    ]
    if USE_OPENCV_ONLY:
        argv.append("--use_cv2")
    sys.argv = argv
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
