"""
步骤 2c（可选）：对 clips 元数据中的全部 clip（正常 + 异常）用基座 VLM 生成 ECVA 风格英文 structured_output，
写入 data/processed/annotations.jsonl（见 configs/default.yaml 的 data.annotations_path）。

PyCharm：工作目录 = 项目根 vlm。需 GPU；OOM 时将 BATCH_SIZE 改为 1。
依赖：已完成 prepare_clips（得到 ecva_clips_metadata.jsonl 或 clips_metadata.jsonl）。

与 step02b 的区别：step02b 仅 --only_normal 写 annotations_normal.jsonl；本步骤不写 --only_normal，
用于新数据集闭环（异常也需要模型生成解释）。
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CFG = _ROOT / "configs" / "default.yaml"

# 视频 VLM 推理显存敏感，默认 1；显存充足可改大
BATCH_SIZE = 1


def main() -> None:
    script = _ROOT / "data" / "generate_annotations.py"
    sys.argv = [
        str(script),
        "--config",
        str(_CFG),
        "--batch_size",
        str(BATCH_SIZE),
    ]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
