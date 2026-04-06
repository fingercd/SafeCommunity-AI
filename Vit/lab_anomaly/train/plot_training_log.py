# -*- coding: utf-8 -*-
"""
读取旧版文本训练日志，解析 epoch / loss / acc 等并绘图。
端到端训练请直接使用 train_end2end 写出的 history.json。

用法:
  python -m lab_anomaly.train.plot_training_log  < 训练日志.txt
  python -m lab_anomaly.train.plot_training_log  --log path/to/log.txt [--out dir]
  python -m lab_anomaly.train.plot_training_log  --log path/to/log.txt --out . --no-show

支持格式示例:
  epoch 991 | train_loss=0.5301 rank=0.2231 acc=0.798 | val_loss=0.9968 acc=0.662  auc_bin=0.838  acc_bin@0.5=0.753
    confusion (rows=gt, cols=pred): ['normal', 'steal', 'violent conflict']
      [182, 45, 18]
      [26, 62, 24]
      [6, 18, 24]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _HAS_PLOT = True
except ImportError:
    _HAS_PLOT = False


def parse_log(lines: List[str]) -> Dict[str, Any]:
    """解析训练日志，返回 epochs, records, confusion_labels, confusion_per_epoch, per_class_acc_per_epoch。"""
    records: List[Dict[str, float]] = []
    confusion_labels: Optional[List[str]] = None
    confusion_per_epoch: Dict[int, List[List[int]]] = {}
    per_class_acc_per_epoch: Dict[int, Dict[str, float]] = {}

    # epoch 991 | train_loss=0.5301 rank=0.2231 acc=0.798 | val_loss=0.9968 acc=0.662  auc_bin=0.838  acc_bin@0.5=0.753
    re_epoch = re.compile(
        r"epoch\s+(\d+)\s+\|\s+train_loss=([\d.]+)\s+rank=([\d.]+)\s+acc=([\d.]+)\s+\|\s+val_loss=([\d.]+)\s+acc=([\d.]+)"
        r"(?:\s+auc_bin=([\d.]+))?(?:\s+acc_bin@0\.5=([\d.]+))?"
    )
    # per_class_acc: {'normal': 0.74..., 'steal': ..., 'violent conflict': ...}
    re_per_class = re.compile(r"per_class_acc:\s*(\{[^}]+\})")
    # confusion (rows=gt, cols=pred): ['normal', 'steal', 'violent conflict']
    re_confusion_labels = re.compile(r"confusion\s+\(rows=gt,\s*cols=pred\):\s*(\[[^\]]+\])")
    # [182, 45, 18]
    re_confusion_row = re.compile(r"^\s*\[([\d,\s]+)\]")

    i = 0
    while i < len(lines):
        line = lines[i]
        m = re_epoch.search(line)
        if m:
            epoch = int(m.group(1))
            rec = {
                "train_loss": float(m.group(2)),
                "train_rank": float(m.group(3)),
                "train_acc": float(m.group(4)),
                "val_loss": float(m.group(5)),
                "val_acc": float(m.group(6)),
            }
            if m.group(7) is not None:
                rec["val_auc_binary"] = float(m.group(7))
            if m.group(8) is not None:
                rec["val_acc_binary_0.5"] = float(m.group(8))
            records.append(rec)

            # 下一行可能是 per_class_acc
            j = i + 1
            if j < len(lines) and "per_class_acc" in lines[j]:
                try:
                    # 提取 {'normal': 0.74, ...} 并用 eval 解析（仅数字和键名）
                    inner = re_per_class.search(lines[j])
                    if inner:
                        s = inner.group(1).replace("'", '"')
                        d = eval(s)
                        per_class_acc_per_epoch[epoch] = {k: float(v) for k, v in d.items()}
                except Exception:
                    pass
                j += 1

            # 接下来可能是 confusion (rows=gt, cols=pred): [...]
            if j < len(lines) and "confusion" in lines[j]:
                lab = re_confusion_labels.search(lines[j])
                if lab:
                    if confusion_labels is None:
                        confusion_labels = eval(lab.group(1).replace("'", '"'))
                    j += 1
                    rows = []
                    while j < len(lines):
                        row_m = re_confusion_row.match(lines[j])
                        if row_m:
                            row = [int(x.strip()) for x in row_m.group(1).split(",")]
                            rows.append(row)
                            j += 1
                        else:
                            break
                    if rows:
                        confusion_per_epoch[epoch] = rows
                i = j - 1
            else:
                i = j - 1
        i += 1

    return {
        "records": records,
        "confusion_labels": confusion_labels or [],
        "confusion_per_epoch": confusion_per_epoch,
        "per_class_acc_per_epoch": per_class_acc_per_epoch,
    }


def plot_results(data: Dict[str, Any], out_dir: Path, show: bool = False) -> None:
    """根据解析结果绘制 loss / acc / auc / 混淆矩阵 等图并保存到 out_dir。"""
    if not _HAS_PLOT:
        print("[WARN] 未安装 matplotlib，无法绘图。请安装: pip install matplotlib numpy")
        return

    records = data["records"]
    if not records:
        print("[WARN] 未解析到任何 epoch 记录，跳过绘图。")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(len(records)))

    # 1. Train/Val Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [r["train_loss"] for r in records], label="train_loss")
    plt.plot(epochs, [r["val_loss"] for r in records], label="val_loss")
    plt.xlabel("epoch")
    plt.title("Train/Val Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_loss.png", dpi=120)
    if show:
        plt.show()
    plt.close()

    # 2. Train/Val Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [r["train_acc"] for r in records], label="train_acc")
    plt.plot(epochs, [r["val_acc"] for r in records], label="val_acc")
    if any("val_auc_binary" in r for r in records):
        plt.plot(epochs, [r.get("val_auc_binary", float("nan")) for r in records], label="val_auc_binary")
    if any("val_acc_binary_0.5" in r for r in records):
        plt.plot(epochs, [r.get("val_acc_binary_0.5", float("nan")) for r in records], label="val_acc_bin@0.5")
    plt.xlabel("epoch")
    plt.title("Accuracy / AUC")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_acc.png", dpi=120)
    if show:
        plt.show()
    plt.close()

    # 3. Train rank loss + val acc + val auc
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [r["train_rank"] for r in records], label="train_rank")
    plt.plot(epochs, [r["val_acc"] for r in records], label="val_acc")
    if any("val_auc_binary" in r for r in records):
        plt.plot(epochs, [r.get("val_auc_binary", float("nan")) for r in records], label="val_auc_binary")
    plt.xlabel("epoch")
    plt.title("Rank loss, Val Acc, Val AUC")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_rank_acc_auc.png", dpi=120)
    if show:
        plt.show()
    plt.close()

    # 4. 最后一个 epoch 的混淆矩阵
    confusion_per_epoch = data["confusion_per_epoch"]
    confusion_labels = data["confusion_labels"]
    if confusion_per_epoch and confusion_labels:
        last_epoch = max(confusion_per_epoch.keys())
        cm = np.array(confusion_per_epoch[last_epoch])
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(confusion_labels)))
        ax.set_yticks(range(len(confusion_labels)))
        ax.set_xticklabels(confusion_labels)
        ax.set_yticklabels(confusion_labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white")
        plt.colorbar(im, ax=ax, label="count")
        plt.title(f"Confusion Matrix (epoch {last_epoch})")
        plt.tight_layout()
        plt.savefig(out_dir / "plot_confusion_last.png", dpi=120)
        if show:
            plt.show()
        plt.close()

    # 5. 若有 per_class_acc，画各类别准确率（取有记录的 epoch）
    per_class = data["per_class_acc_per_epoch"]
    if per_class:
        ep_list = sorted(per_class.keys())
        classes = list(next(iter(per_class.values())).keys())
        plt.figure(figsize=(8, 5))
        for cls in classes:
            accs = [per_class[ep].get(cls, float("nan")) for ep in ep_list]
            plt.plot(ep_list, accs, label=cls)
        plt.xlabel("epoch")
        plt.title("Per-class Accuracy (every 5 epochs)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "plot_per_class_acc.png", dpi=120)
        if show:
            plt.show()
        plt.close()

    print(f"[OK] 图表已保存至: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="解析 known classifier 训练日志并绘图")
    ap.add_argument("--log", type=str, default=None, help="日志文件路径；不指定则从 stdin 读取")
    ap.add_argument("--out", type=str, default=None, help="图片输出目录，默认与日志同目录或当前目录")
    ap.add_argument("--no-show", action="store_true", help="不弹出显示窗口，仅保存文件（建议加上以免卡住）")
    args = ap.parse_args()

    print("plot_training_log: 开始运行...", flush=True)

    if args.log:
        log_path = Path(args.log).resolve()
        if not log_path.exists():
            print(f"[ERROR] 文件不存在: {log_path}", flush=True)
            sys.exit(1)
        print(f"  读取日志: {log_path}", flush=True)
        text = log_path.read_text(encoding="utf-8", errors="replace")
        out_dir = args.out or str(log_path.parent)
    else:
        print("  未指定 --log，正在从标准输入读取（输入结束后按 Ctrl+Z 回车 或 管道传入）。建议: --log 你的日志.txt", flush=True)
        text = sys.stdin.read()
        out_dir = args.out or "."

    lines = text.strip().splitlines()
    print(f"  共 {len(lines)} 行，解析中...", flush=True)
    data = parse_log(lines)
    n = len(data["records"])
    print(f"[OK] 解析到 {n} 个 epoch", flush=True)
    if data["confusion_labels"]:
        print(f"     类别: {data['confusion_labels']}")
    if data["per_class_acc_per_epoch"]:
        print(f"     per_class_acc 记录: {len(data['per_class_acc_per_epoch'])} 个 epoch")

    print("  绘图并保存（--no-show 不弹窗）...", flush=True)
    plot_results(data, Path(out_dir), show=not args.no_show)


if __name__ == "__main__":
    main()
