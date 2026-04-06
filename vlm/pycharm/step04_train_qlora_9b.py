"""
步骤 4：QLoRA 微调 9B（paths/超参全部来自 configs/default.yaml）
PyCharm：工作目录 = 项目根 vlm。需 GPU + 足够显存。
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CFG = _ROOT / "configs" / "default.yaml"


def main() -> None:
    script = _ROOT / "train" / "train_qlora.py"
    sys.argv = [str(script), "--config", str(_CFG)]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
