"""
步骤 1：ECVA Excel → ecva_metadata.jsonl
PyCharm：本配置以 configs/default.yaml 为准；工作目录 = 项目根目录 vlm。
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CFG = _ROOT / "configs" / "default.yaml"


def main() -> None:
    script = _ROOT / "data" / "parse_ecva.py"
    sys.argv = [str(script), "--config", str(_CFG)]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
