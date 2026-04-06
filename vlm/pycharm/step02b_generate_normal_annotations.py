"""
步骤 2b（可选）：对 ecva_clips_metadata 中 label_binary==normal 的 clip 用基座 VLM 离线生成 structured_output，
写入 data/processed/annotations_normal.jsonl；随后 step03 build_dataset 会合并进 train/val/test。

PyCharm：工作目录 = 项目根 vlm。需 GPU；OOM 时在本文件末尾把 BATCH_SIZE 改为 1。
依赖：已完成 prepare_clips（ecva）。
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CFG = _ROOT / "configs" / "default.yaml"

# 视频 VLM 推理显存敏感，默认 1
BATCH_SIZE = 1


def main() -> None:
    script = _ROOT / "data" / "generate_annotations.py"
    sys.argv = [
        str(script),
        "--config",
        str(_CFG),
        "--only_normal",
        "--batch_size",
        str(BATCH_SIZE),
    ]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
