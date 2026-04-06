"""
步骤 3：ecva_clips_metadata.jsonl → train.json / val.json / test.json（按 video_id 划分）
PyCharm：工作目录 = 项目根 vlm。
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CFG = _ROOT / "configs" / "default.yaml"


def main() -> None:
    script = _ROOT / "data" / "build_dataset.py"
    sys.argv = [
        str(script),
        "--source",
        "ecva",
        "--config",
        str(_CFG),
    ]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
