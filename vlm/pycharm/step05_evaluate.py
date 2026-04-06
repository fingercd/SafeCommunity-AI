"""
步骤 5：在 test.json 上只评测微调后的模型（LoRA 目录 = 基座 + adapter）。
PyCharm：工作目录 = 项目根 vlm。跑前需完成 step04。

下面常量可在 PyCharm 里直接改，无需命令行。
预测结果：与报告同目录下 {报告名}_predictions.jsonl / .json（由 evaluate.py 生成）。
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CFG = _ROOT / "configs" / "default.yaml"
# 含 adapter_config.json 的目录（一般为训练输出的 final 或某个 checkpoint）
_ADAPTER = _ROOT / "outputs" / "qlora" / "checkpoint-100"
# 单模型评测报告路径（不设 --dual_eval，只跑微调）
_REPORT = _ROOT / "outputs" / "finetuned_report.json"

# 每多少条 clip 写一次完整 predictions.json 快照（jsonl 每条追加）
_PREDICTIONS_FLUSH_EVERY = 10
# True：续跑断点；False：加 --no_resume 清空该报告对应的 predictions 断点
_RESUME = True


def main() -> None:
    script = _ROOT / "train" / "evaluate.py"
    sys.argv = [
        str(script),
        "--config",
        str(_CFG),
        "--model_path",
        str(_ADAPTER),
        "--output_report",
        str(_REPORT),
        "--predictions_flush_every",
        str(_PREDICTIONS_FLUSH_EVERY),
    ]
    if not _RESUME:
        sys.argv.append("--no_resume")
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
