"""
已知异常 MIL 分类器训练（基于 clip embeddings 缓存）

用途：当有 normal + 抢劫等异常标签时，在 clip embeddings 上训练 MIL 头，
     学习「正常 vs 抢劫」等多类分类；支持 attn / topk 两种聚合方式。

可选 MIL 排序损失（借鉴 CVPR 2018 论文）：
  --use_anomaly_branch 启用异常分数分支 + MIL 排序损失
  排序损失与 CE 分类损失联合训练：loss = loss_ce + lambda_rank * loss_rank
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lab_anomaly.models.mil_head import MILClassifier, MILHeadConfig

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None
from lab_anomaly.models.ranking_loss import mil_ranking_loss

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """逐行解析 JSONL 文件"""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class EmbeddingMILVideoDataset(Dataset):
    """
    以 extract_embeddings.py 的输出（embeddings_meta.jsonl + npy）为输入，
    返回：每个视频一个样本：clips embeddings (N,D) + mask (N,)
    """

    def __init__(
        self,
        embeddings_dir: str | Path,
        expected_num_clips: int = 0,
        include_labels: Optional[set[str]] = None,
        exclude_labels: Optional[set[str]] = None,
        use_rgb_only: bool = False,
        label_mapping: Optional[dict[str, str]] = None,
    ):
        self.embeddings_dir = Path(embeddings_dir)
        # 仅用 RGB 训练：为 True 时从双流 1536 维中只取前 768 维（RGB 分支），丢弃光流 768 维
        self.use_rgb_only = bool(use_rgb_only)
        self.label_mapping = label_mapping or {}

        def _map_label(l: str) -> str:
            return self.label_mapping.get(l, l)

        meta_path = self.embeddings_dir / "embeddings_meta.jsonl"
        if not meta_path.exists():
            raise FileNotFoundError(f"cannot find embeddings meta: {meta_path}")

        def resolve_path(path_str: str) -> str:
            """非绝对路径则相对于 embeddings_dir 解析，便于换 cwd 仍能加载。"""
            p = Path(path_str)
            return str((self.embeddings_dir / p).resolve()) if not p.is_absolute() else path_str

        rows = read_jsonl(meta_path)

        # 支持两种 embedding 缓存格式：
        # - npy_per_clip：jsonl 一条记录对应一个 clip
        # - npz_per_video：jsonl 一条记录对应一个视频（npz 内包含全部 clips embeddings）
        by_video_npy: dict[str, list[dict[str, Any]]] = {}
        by_video_npz: dict[str, dict[str, Any]] = {}

        for r in rows:
            label = _map_label(str(r.get("label", "")))
            if include_labels is not None and label not in include_labels:
                continue
            if exclude_labels is not None and label in exclude_labels:
                continue

            fmt = str(r.get("save_format", "npy_per_clip"))
            vid = str(r["video_id"])
            if fmt == "npz_per_video":
                # 同一视频如既有 npz 又有 npy，优先使用 npy（更细粒度，也便于追溯 clip）
                if vid not in by_video_npz:
                    by_video_npz[vid] = r
            else:
                by_video_npy.setdefault(vid, []).append(r)

        items: list[dict[str, Any]] = []

        # 先收集 npy_per_clip 视频
        for vid, clips in by_video_npy.items():
            clips = sorted(clips, key=lambda x: int(x.get("clip_index", 0)))
            labels = [_map_label(str(x.get("label", ""))) for x in clips]
            label = labels[0] if labels else ""
            if len(set(labels)) > 1:
                label = max(set(labels), key=lambda k: labels.count(k))
            items.append(
                {
                    "video_id": vid,
                    "label": label,
                    "save_format": "npy_per_clip",
                    "clip_paths": [resolve_path(str(x["embedding_path"])) for x in clips],
                    "clip_indices": [int(x.get("clip_index", i)) for i, x in enumerate(clips)],
                    "num_clips": int(len(clips)),
                }
            )

        # 再补充只有 npz_per_video 的视频（排除已被 npy 覆盖的）
        for vid, r in by_video_npz.items():
            if vid in by_video_npy:
                continue
            items.append(
                {
                    "video_id": vid,
                    "label": _map_label(str(r.get("label", ""))),
                    "save_format": "npz_per_video",
                    "npz_path": resolve_path(str(r["embedding_path"])),
                    "num_clips": int(r.get("num_clips", 0) or 0),
                }
            )

        # 固定每个视频 clip 数量，便于 batch
        if expected_num_clips <= 0:
            expected_num_clips = int(max((int(x.get("num_clips", 0) or 0) for x in items), default=0))
        self.expected_num_clips = int(expected_num_clips)
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        it = self.items[idx]
        label = str(it["label"])
        vid = str(it["video_id"])
        fmt = str(it.get("save_format", "npy_per_clip"))

        embs: list[np.ndarray] = []
        if fmt == "npz_per_video":
            npz_path = str(it["npz_path"])
            npz = np.load(npz_path, allow_pickle=True)
            E = np.asarray(npz["embeddings"], dtype=np.float32)  # (N,D)
            if E.ndim != 2 or E.shape[0] == 0:
                raise RuntimeError(f"video {vid} has invalid npz embeddings: {npz_path}")
            for i in range(min(E.shape[0], self.expected_num_clips)):
                embs.append(E[i])
        else:
            paths = list(it.get("clip_paths", []))
            for p in paths[: self.expected_num_clips]:
                arr = np.load(p).astype(np.float32)
                embs.append(arr)

        if not embs:
            raise RuntimeError(f"video {vid} has no embeddings")

        d = int(embs[0].shape[-1])
        n_valid = int(len(embs))
        n = self.expected_num_clips
        x = np.zeros((n, d), dtype=np.float32)
        mask = np.zeros((n,), dtype=np.bool_)
        x[:n_valid] = np.stack(embs, axis=0)
        mask[:n_valid] = True

        # 只用 RGB 训练：双流 embedding 为 [RGB_768, Flow_768] concat，取前 768 维即 RGB，丢弃光流
        if self.use_rgb_only and d >= 768:
            x = x[:, :768].copy()
            d = 768

        return {
            "video_id": vid,
            "label": label,
            "x": torch.from_numpy(x),  # (N,D)，D=768 当 use_rgb_only 且原为 1536 维
            "mask": torch.from_numpy(mask),  # (N,)
        }


def split_train_val(items: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """按比例划分训练集/验证集索引"""
    idxs = list(range(len(items)))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)
    n_val = int(round(len(idxs) * float(val_ratio)))
    val = idxs[:n_val]
    train = idxs[n_val:]
    return train, val


class BalancedBatchSampler(torch.utils.data.Sampler[list[int]]):
    """
    保证每个 batch 内至少 1 个 normal bag + 1 个 anomaly bag，避免 ranking loss 长期为 0。
    - train_idx: 训练集对应的全局样本下标（Subset 的 indice 即 0..len(train_idx)-1）
    - items: ds.items，用于取 label
    - epoch_ref: 可选，[current_epoch] 的列表；若提供则每轮迭代使用 seed + epoch_ref[0]，便于复用同一 DataLoader
    """

    def __init__(
        self,
        train_idx: list[int],
        items: list[dict[str, Any]],
        normal_label: str,
        batch_size: int,
        drop_last: bool = False,
        seed: int = 0,
        epoch_ref: Optional[list[int]] = None,
    ):
        self.train_idx = train_idx
        self.items = items
        self.normal_label = normal_label
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch_ref = epoch_ref  # 可变引用，每 epoch 更新后 __iter__ 用 seed + epoch_ref[0]
        # 按 label 分为 normal / anomaly（均为 Subset 内下标 0..len(train_idx)-1）
        self.normal_idx: list[int] = []
        self.anomaly_idx: list[int] = []
        for i in range(len(train_idx)):
            global_idx = train_idx[i]
            if str(items[global_idx].get("label", "")) == normal_label:
                self.normal_idx.append(i)
            else:
                self.anomaly_idx.append(i)

    def __iter__(self):
        effective_seed = self.seed + (self.epoch_ref[0] if self.epoch_ref else 0)
        rnd = random.Random(effective_seed)
        normal = list(self.normal_idx)
        anomaly = list(self.anomaly_idx)
        rnd.shuffle(normal)
        rnd.shuffle(anomaly)
        batches: list[list[int]] = []
        while normal or anomaly:
            batch: list[int] = []
            if normal and anomaly:
                batch.append(normal.pop())
                batch.append(anomaly.pop())
            elif normal:
                batch.append(normal.pop())
            else:
                batch.append(anomaly.pop())
            while len(batch) < self.batch_size and (normal or anomaly):
                if normal and anomaly:
                    batch.append(normal.pop() if rnd.random() < 0.5 else anomaly.pop())
                elif normal:
                    batch.append(normal.pop())
                else:
                    batch.append(anomaly.pop())
            if self.drop_last and len(batch) < self.batch_size:
                break
            rnd.shuffle(batch)
            batches.append(batch)
        rnd.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        n = len(self.normal_idx) + len(self.anomaly_idx)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """DataLoader 的 collate_fn：将多个样本 stack 成 batch"""
    video_ids = [b["video_id"] for b in batch]
    labels = [b["label"] for b in batch]
    x = torch.stack([b["x"] for b in batch], dim=0)  # (B,N,D)
    mask = torch.stack([b["mask"] for b in batch], dim=0)  # (B,N)
    return {"video_id": video_ids, "label": labels, "x": x, "mask": mask}


@torch.no_grad()
def evaluate(
    model: MILClassifier,
    loader: DataLoader,
    label2idx: dict[str, int],
    device: str,
    eval_with_gt_topk: bool = False,
    ce_weight: Optional[torch.Tensor] = None,
    normal_idx: Optional[int] = None,
    idx2label: Optional[dict[int, str]] = None,
) -> dict[str, Any]:
    """在验证集上计算 val_loss、val_acc、二分类 AUC（normal vs others）及 per-class acc/混淆矩阵。"""
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    all_y: list[int] = []
    all_pred: list[int] = []
    all_p_abn: list[float] = []  # 流式收集二分类概率，避免囤积全部 logits
    ce = nn.CrossEntropyLoss(weight=ce_weight)
    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        y = torch.tensor([label2idx[l] for l in batch["label"]], device=device, dtype=torch.long)
        use_gt_for_topk = eval_with_gt_topk and model.pooling == "topk"
        logits, _ = model(x, mask=mask, y=y if use_gt_for_topk else None)
        loss = ce(logits, y)
        pred = logits.argmax(dim=-1)
        total += int(y.numel())
        correct += int((pred == y).sum().item())
        loss_sum += float(loss.item()) * int(y.numel())
        all_y.extend(y.cpu().tolist())
        all_pred.extend(pred.cpu().tolist())
        if normal_idx is not None and roc_auc_score is not None:
            probs = F.softmax(logits, dim=-1)
            p_abn_batch = (1.0 - probs[:, normal_idx]).cpu().tolist()
            all_p_abn.extend(p_abn_batch)
    n = len(all_y)
    result: dict[str, Any] = {
        "val_loss": loss_sum / max(1, total),
        "val_acc": correct / max(1, total),
    }
    # 二分类 AUC：normal vs others（用流式收集的 p_abn，无需囤积 logits）
    if normal_idx is not None and n > 0 and roc_auc_score is not None and len(all_p_abn) == n:
        y_bin = np.array([1 if yi != normal_idx else 0 for yi in all_y], dtype=np.int64)
        p_abn = np.array(all_p_abn, dtype=np.float64)
        if np.unique(y_bin).size == 2:
            result["val_auc_binary"] = float(roc_auc_score(y_bin, p_abn))
        else:
            result["val_auc_binary"] = float("nan")
        pred_bin = (p_abn >= 0.5).astype(np.int64)
        result["val_acc_binary_0.5"] = float(np.mean(pred_bin == y_bin))
    # per-class 准确率与混淆矩阵
    if n > 0 and idx2label is not None:
        classes = sorted(idx2label.keys())
        num_classes = len(classes)
        correct_per = {c: 0 for c in classes}
        total_per = {c: 0 for c in classes}
        cm = [[0] * num_classes for _ in range(num_classes)]
        for yi, pi in zip(all_y, all_pred):
            total_per[yi] = total_per.get(yi, 0) + 1
            if yi == pi:
                correct_per[yi] = correct_per.get(yi, 0) + 1
            cm[yi][pi] = cm[yi][pi] + 1
        result["per_class_acc"] = {
            idx2label[c]: (correct_per[c] / max(1, total_per[c])) for c in classes
        }
        result["confusion_matrix"] = cm
        result["confusion_labels"] = [idx2label[c] for c in classes]
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="已知异常：视频级标签 MIL 聚合分类训练（基于 clip embeddings 缓存）")
    ap.add_argument("--embeddings_dir", type=str, default = r"C:\Users\Administrator\Desktop\Vit\lab_dataset\derived\embeddings")
    ap.add_argument("--out_dir", type=str, default= r"C:\Users\Administrator\Desktop\Vit\lab_dataset\derived\known_classifier")
    ap.add_argument("--exclude_unknown", action="store_true", help="排除 label=unknown 的样本")

    ap.add_argument("--pooling", type=str, default="attn", choices=["attn", "topk"])
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--attn_hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.65)
    ap.add_argument("--no_layernorm", action="store_true", help="默认启用 LayerNorm；加此参数可关闭")

    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--target_acc", type=float, default=0.8, help="验证准确率达到此值后提前停止（0=不提前停）")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4, help="默认 2e-4，过小易欠拟合")
    ap.add_argument("--weight_decay", type=float, default= 5e-2)
    ap.add_argument("--val_ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=420)
    ap.add_argument("--device", type=str, default="", help="cuda/cpu（默认自动）")
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--expected_num_clips", type=int, default=0, help="每视频 clip 数（0=自动推断最大值）")
    ap.add_argument("--resume", type=str, default="", help="从 checkpoint.pt 恢复")
    ap.add_argument("--save_every", type=int, default=1, help="每 N 个 epoch 保存一次 checkpoint_last.pt（best 仍按 val 提升即存）")

    # --- MIL 排序损失参数（借鉴 CVPR 2018 论文） ---
    ap.add_argument(
        "--use_anomaly_branch",
        action="store_true",
        default=True,
        help="启用异常分数分支 + MIL 排序损失：为每个 clip 预测 0~1 异常分数，与排序/稀疏/平滑损失联合训练，使模型更关注异常片段而非背景。不开启时仅做视频级分类。",
    )
    ap.add_argument(
        "--lambda_rank",
        type=float,
        default=0.7,
        help="MIL 排序损失权重：正 bag（异常视频）最高 clip 分数应大于负 bag（正常视频）最高分数，该项在总 loss 中的系数。",
    )
    ap.add_argument(
        "--lambda_sparse",
        type=float,
        default=8e-5,
        help="稀疏约束权重：正 bag 内各 clip 异常分数之和尽量小，鼓励异常只出现在少数片段（论文建议 8e-5）。",
    )
    ap.add_argument(
        "--lambda_smooth",
        type=float,
        default=8e-5,
        help="时间平滑约束权重：相邻 clip 的异常分数差平方和，使分数在时间上连续、减少抖动（论文建议 8e-5）。",
    )
    ap.add_argument(
        "--anomaly_hidden",
        type=int,
        default=256,
        help="异常分数分支隐层维度：embedding -> 512 -> 32 -> 1(Sigmoid)，与 CVPR 2018 结构一致。",
    )
    ap.add_argument(
        "--normal_label",
        type=str,
        default="normal",
        help="正常类标签名：用于区分正/负 bag（负=正常视频、正=异常视频），排序损失与平衡采样依赖此项；须与数据集中标签一致。",
    )
    ap.add_argument(
        "--eval_with_gt_topk",
        action="store_true",
        help="验证时 topk 用 GT 选 clip：仅在 pooling=topk 时生效，验证按真实类别选 top-k clip 聚合。用于诊断：若开启后 val_acc 明显升则说明训练/推理选 clip 方式不一致。",
    )
    ap.add_argument(
        "--topk_consistent",
        action="store_true",
        help="训练与推理一致：topk 训练时也不传 GT，按 max logit 选 clip，与推理一致。若 val 接近随机可加此参数或改用 pooling attn。",
    )
    ap.add_argument(
        "--class_weight",
        action="store_true",
        default=True,
        help="按类别样本数对 CE 加权：样本少的类权重大，缓解类别不平衡；类别分布悬殊时建议开启。",
    )
    ap.add_argument(
        "--use_rgb_only",
        action="store_true",
        help="只用 RGB 分支：从双流 1536 维 embedding 中只取前 768 维（RGB）训练，丢弃光流维；不重新提 embedding。",
    )


    # --- 可选：基座训练结束后跑伪标签迭代（同一 ds/val_idx） ---
    ap.add_argument(
        "--run_pseudo_iter",
        action="store_true",
        default=True,
        help="基座训练结束后用 checkpoint_best.pt 自动跑伪标签迭代：KMeans+覆盖度×分类分数选 top/bottom clip 作伪标签，与视频级 MIL 联合训练，提升 clip 级判别；结果写入 --pseudo_out_dir。",
    )
    ap.add_argument(
        "--pseudo_out_dir",
        type=str,
        default="",
        help="伪标签迭代输出目录；为空时使用 out_dir/pseudo_iter。迭代 checkpoint 与 pseudo_stats/ metrics 写在此目录。",
    )
    ap.add_argument(
        "--pseudo_iters",
        type=int,
        default=3,
        help="伪标签迭代轮数：每轮重新用当前模型生成伪标签（top-k/bottom-k），再训练若干 epoch。",
    )
    ap.add_argument(
        "--pseudo_epochs_per_iter",
        type=int,
        default=3,
        help="每轮伪标签迭代内训练的 epoch 数；每轮先建伪标签再训练该 epoch 数。",
    )
    ap.add_argument(
        "--pseudo_kmeans_k",
        type=int,
        default=32,
        help="伪标签生成时对同类 clip embedding 做 KMeans 的簇数，用于覆盖度等权重。",
    )
    ap.add_argument(
        "--pseudo_top_k",
        type=int,
        default=2,
        help="每视频选伪正样本的 clip 数：按覆盖度×分类分数×离簇心权重排序后取 top-k 作为该类正样本。",
    )
    ap.add_argument(
        "--pseudo_bottom_k",
        type=int,
        default=2,
        help="每视频选伪负样本的 clip 数：分数最低的 bottom-k；默认作为对目标类的负约束（压低该类概率），加 --pseudo_no_negative_constraint 则标为 normal。",
    )
    ap.add_argument(
        "--pseudo_lambda",
        type=float,
        default=0.5,
        help="伪标签 clip 损失在总 loss 中的权重：loss = loss_video + pseudo_lambda * loss_pseudo。",
    )
    ap.add_argument(
        "--pseudo_lr",
        type=float,
        default=1e-4,
        help="伪标签迭代阶段的优化器学习率。",
    )
    ap.add_argument(
        "--pseudo_weight_decay",
        type=float,
        default=1e-2,
        help="伪标签迭代阶段的 AdamW 权重衰减。",
    )
    ap.add_argument(
        "--pseudo_num_workers",
        type=int,
        default=0,
        help="伪标签迭代的 DataLoader num_workers；为 0 时使用与主训练相同的 num_workers。",
    )
    ap.add_argument(
        "--pseudo_no_negative_constraint",
        action="store_true",
        help="加此参数则 bottom-k 标为 normal 类；不加则对目标类做负约束（训练时压低该类概率），默认推荐负约束。",
    )

    args = ap.parse_args()

    device = args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude = {"unknown"} if args.exclude_unknown else set()
    exclude.add("Vandalism")  # 固定排除 Vandalism
    exclude.add("RoadAccidents")  # 固定排除 RoadAccidents，做 3 类分类
    # use_rgb_only=True 时在数据集内只取 embedding 前 768 维（RGB），与双流 1536 维兼容
    # Robbery 归入 steal，做 3 类分类：normal, steal, violent conflict
    label_mapping = {"Robbery": "steal"}
    ds = EmbeddingMILVideoDataset(
        embeddings_dir=args.embeddings_dir,
        expected_num_clips=int(args.expected_num_clips),
        exclude_labels=exclude if exclude else None,
        use_rgb_only=getattr(args, "use_rgb_only", False),
        label_mapping=label_mapping,
    )
    if label_mapping:
        print(f"[LABEL_MAPPING] {label_mapping}（Robbery 归入 steal，3 类分类）")
    if getattr(args, "use_rgb_only", False):
        print("[RGB_ONLY] 仅用前 768 维（RGB）训练，光流维已丢弃")
    if len(ds) <= 1:
        raise RuntimeError("dataset too small; please check embeddings_dir and labels.")

    # label space
    labels = sorted({it["label"] for it in ds.items})
    if "normal" in labels:
        labels = ["normal"] + [x for x in labels if x != "normal"]
    label2idx = {l: i for i, l in enumerate(labels)}
    idx2label = {i: l for l, i in label2idx.items()}

    # 类别分布（便于不平衡时启用 class_weight / 调 batch、topk）
    label_counts = Counter(it["label"] for it in ds.items)
    counts_list = [label_counts.get(l, 0) for l in labels]
    n_min, n_max = min(counts_list), max(counts_list)
    print(f"[CLASS_DIST] {dict(zip(labels, counts_list))}")
    if n_max > 0 and n_min > 0 and (n_max / n_min) > 6 and not getattr(args, "class_weight", False):
        print("[IMBALANCE] 建议加 --class_weight 并按分布考虑调大 batch_size 或 topk；伪标签迭代时可调 lambda_pseudo")

    # infer embedding dim
    sample0 = ds[0]["x"]
    embedding_dim = int(sample0.shape[-1])

    cfg = MILHeadConfig(
        embedding_dim=embedding_dim,
        num_classes=len(labels),
        pooling=args.pooling,
        topk=int(args.topk),
        attn_hidden=int(args.attn_hidden),
        dropout=float(args.dropout),
        use_layernorm=not bool(args.no_layernorm),
        use_anomaly_branch=bool(args.use_anomaly_branch),
        anomaly_hidden=int(args.anomaly_hidden),
    )
    model = MILClassifier(cfg).to(device)

    # 确定 normal 标签在 label space 中的 index（用于正负 bag 区分）
    normal_label = str(args.normal_label).strip()
    normal_idx = label2idx.get(normal_label, -1)
    if args.use_anomaly_branch and normal_idx < 0:
        print(f"[WARN] normal_label={normal_label!r} not found in labels {labels}; "
              f"ranking loss will be skipped (no negative bags).")

    start_epoch = 0
    best_val = -1.0
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val_acc", -1.0))
        # 允许从旧 ckpt 恢复 label2idx（如需要）
        if "label2idx" in ckpt:
            label2idx = {str(k): int(v) for k, v in ckpt["label2idx"].items()}
            idx2label = {int(k): str(v) for k, v in ckpt["idx2label"].items()}
        print(f"[OK] resume from {args.resume}, start_epoch={start_epoch}")

    # split
    train_idx, val_idx = split_train_val(ds.items, val_ratio=float(args.val_ratio), seed=int(args.seed))
    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)
    batch_size = int(args.batch_size)
    num_workers = int(args.num_workers)
    pin_mem = device.startswith("cuda")

    # 平衡采样：保证每 batch 至少 1 normal + 1 anomaly，避免 ranking loss 长期为 0
    normal_label = str(args.normal_label).strip()
    n_normal_train = sum(1 for i in train_idx if str(ds.items[i].get("label", "")) == normal_label)
    n_anomaly_train = len(train_idx) - n_normal_train
    use_balanced_sampler = (
        bool(args.use_anomaly_branch) and normal_idx >= 0 and n_normal_train > 0 and n_anomaly_train > 0
    )
    if use_balanced_sampler:
        print(f"[BALANCED] train: {n_normal_train} normal, {n_anomaly_train} anomaly; "
              "each batch has at least 1 normal + 1 anomaly bag.")
        # 只创建一次 DataLoader；每 epoch 通过 epoch_ref 更新 seed，避免每轮重建多进程
        _epoch_ref: list[int] = [0]
        _balanced_sampler = BalancedBatchSampler(
            train_idx=train_idx,
            items=ds.items,
            normal_label=normal_label,
            batch_size=batch_size,
            drop_last=False,
            seed=int(args.seed),
            epoch_ref=_epoch_ref,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=_balanced_sampler,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=pin_mem,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=pin_mem,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=pin_mem,
    )

    # 可选：按类别样本数对 CE 加权
    ce_weight: Optional[torch.Tensor] = None
    if getattr(args, "class_weight", False):
        counts = Counter(it["label"] for it in ds.items)
        n_samples = len(ds.items)
        n_classes = len(labels)
        weights = [n_samples / (n_classes * max(counts.get(l, 1), 1)) for l in labels]
        ce_weight = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f"[CLASS_WEIGHT] per-class weight: {dict(zip(labels, [round(w, 4) for w in weights]))}")
    ce = nn.CrossEntropyLoss(weight=ce_weight)
    use_rank = bool(args.use_anomaly_branch) and normal_idx >= 0
    if use_rank:
        print(f"[RANK] MIL ranking loss enabled: lambda_rank={args.lambda_rank}, "
              f"lambda_sparse={args.lambda_sparse}, lambda_smooth={args.lambda_smooth}")

    if model.pooling == "topk":
        if getattr(args, "topk_consistent", False):
            print("[TOPK_CONSISTENT] 训练与推理均按 max logit 选 topk，不做 GT 选 clip")
        else:
            print("[TOPK] 训练用 GT 选 topk、验证用 max 选 topk；若 val 接近随机可加 --topk_consistent 或改用 --pooling attn")

    # 训练前打印完整超参数表
    _param_rows = [
        ("embeddings_dir", str(args.embeddings_dir)),
        ("out_dir", str(args.out_dir)),
        ("exclude_unknown", getattr(args, "exclude_unknown", False)),
        ("pooling", args.pooling),
        ("topk", args.topk),
        ("attn_hidden", args.attn_hidden),
        ("dropout", args.dropout),
        ("no_layernorm", getattr(args, "no_layernorm", False)),
        ("epochs", args.epochs),
        ("target_acc", args.target_acc),
        ("batch_size", args.batch_size),
        ("lr", args.lr),
        ("weight_decay", args.weight_decay),
        ("val_ratio", args.val_ratio),
        ("seed", args.seed),
        ("device", device),
        ("num_workers", args.num_workers),
        ("expected_num_clips", args.expected_num_clips),
        ("save_every", getattr(args, "save_every", 1)),
        ("resume", args.resume or "(none)"),
        ("use_anomaly_branch", getattr(args, "use_anomaly_branch", False)),
        ("lambda_rank", getattr(args, "lambda_rank", 1.0)),
        ("lambda_sparse", getattr(args, "lambda_sparse", 8e-5)),
        ("lambda_smooth", getattr(args, "lambda_smooth", 8e-5)),
        ("anomaly_hidden", getattr(args, "anomaly_hidden", 512)),
        ("normal_label", args.normal_label),
        ("eval_with_gt_topk", getattr(args, "eval_with_gt_topk", False)),
        ("topk_consistent", getattr(args, "topk_consistent", False)),
        ("class_weight", getattr(args, "class_weight", False)),
        ("use_rgb_only", getattr(args, "use_rgb_only", False)),
        ("num_classes", len(labels)),
        ("labels", str(labels)),
        ("embedding_dim", embedding_dim),
        ("train_size", len(train_ds)),
        ("val_size", len(val_ds)),
    ]
    _w = max(len(k) for k, _ in _param_rows)
    print("\n" + "=" * (_w + 2 + 50))
    print(f"{'Parameter':<{_w}} | {'Value'}")
    print("-" * (_w + 2 + 50))
    for _k, _v in _param_rows:
        print(f"{_k:<{_w}} | {_v}")
    print("=" * (_w + 2 + 50) + "\n")

    print(f"[START] train={len(train_ds)} val={len(val_ds)} batch_size={batch_size} num_workers={num_workers}")
    history: list[dict[str, float]] = []
    for epoch in range(start_epoch, int(args.epochs)):
        if use_balanced_sampler:
            _epoch_ref[0] = epoch  # 同一 DataLoader 复用，仅改 sampler 的 effective_seed
        model.train()
        loss_sum = 0.0
        rank_loss_sum = 0.0
        total = 0
        correct = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            y = torch.tensor([label2idx[l] for l in batch["label"]], device=device, dtype=torch.long)

            opt.zero_grad(set_to_none=True)
            # topk_consistent=True 时训练也不传 y，与推理一致（用 max 选 clip）
            use_gt_topk_train = (
                model.pooling == "topk"
                and not getattr(args, "topk_consistent", False)
            )
            logits, details = model(x, mask=mask, y=y if use_gt_topk_train else None, return_details=True)
            loss_ce = ce(logits, y)

            # --- MIL 排序损失 ---
            loss_rank = torch.tensor(0.0, device=device)
            if use_rank and "anomaly_scores" in details:
                anomaly_scores = details["anomaly_scores"]  # (B, N)
                is_normal = (y == normal_idx)
                is_anomaly = ~is_normal
                if is_anomaly.any() and is_normal.any():
                    loss_rank = mil_ranking_loss(
                        scores_pos=anomaly_scores[is_anomaly],
                        scores_neg=anomaly_scores[is_normal],
                        mask_pos=mask[is_anomaly],
                        mask_neg=mask[is_normal],
                        lambda_sparse=float(args.lambda_sparse),
                        lambda_smooth=float(args.lambda_smooth),
                    )

            loss = loss_ce + float(args.lambda_rank) * loss_rank
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                total += int(y.numel())
                correct += int((pred == y).sum().item())
                loss_sum += float(loss_ce.item()) * int(y.numel())
                rank_loss_sum += float(loss_rank.item()) * int(y.numel())

        train_loss = loss_sum / max(1, total)
        train_rank = rank_loss_sum / max(1, total)
        train_acc = correct / max(1, total)

        metrics = evaluate(
            model, val_loader, label2idx=label2idx, device=device,
            eval_with_gt_topk=getattr(args, "eval_with_gt_topk", False),
            ce_weight=ce_weight,
            normal_idx=normal_idx if normal_idx >= 0 else None,
            idx2label=idx2label,
        )
        rank_info = f" rank={train_rank:.4f}" if use_rank else ""
        log_parts = [
            f"epoch {epoch:03d} | train_loss={train_loss:.4f}{rank_info} acc={train_acc:.3f}",
            f"| val_loss={metrics['val_loss']:.4f} acc={metrics['val_acc']:.3f}",
        ]
        if "val_auc_binary" in metrics and not np.isnan(metrics["val_auc_binary"]):
            log_parts.append(f" auc_bin={metrics['val_auc_binary']:.3f}")
        if "val_acc_binary_0.5" in metrics:
            log_parts.append(f" acc_bin@0.5={metrics['val_acc_binary_0.5']:.3f}")
        # 记录本 epoch 指标用于绘图与最终汇总
        rec = {
            "train_loss": train_loss,
            "train_rank": train_rank,
            "train_acc": train_acc,
            "val_loss": metrics["val_loss"],
            "val_acc": metrics["val_acc"],
        }
        if "val_auc_binary" in metrics and not np.isnan(metrics["val_auc_binary"]):
            rec["val_auc_binary"] = float(metrics["val_auc_binary"])
        if "val_acc_binary_0.5" in metrics:
            rec["val_acc_binary_0.5"] = float(metrics["val_acc_binary_0.5"])
        history.append(rec)
        print(" ".join(log_parts))
        if "per_class_acc" in metrics and epoch % 5 == 0:
            print("  per_class_acc:", metrics["per_class_acc"])
        if "confusion_matrix" in metrics:
            print("  confusion (rows=gt, cols=pred):", metrics["confusion_labels"])
            for row in metrics["confusion_matrix"]:
                print("   ", row)

        # save last（每 save_every 个 epoch 或最后一个 epoch）
        save_every = int(getattr(args, "save_every", 1))
        if save_every <= 0 or epoch % save_every == 0 or epoch == int(args.epochs) - 1:
            last_path = out_dir / "checkpoint_last.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "cfg": asdict(cfg),
                    "label2idx": label2idx,
                    "idx2label": idx2label,
                    "best_val_acc": best_val,
                },
                last_path,
            )

        if metrics["val_acc"] > best_val:
            best_val = float(metrics["val_acc"])
            best_path = out_dir / "checkpoint_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "cfg": asdict(cfg),
                    "label2idx": label2idx,
                    "idx2label": idx2label,
                    "best_val_acc": best_val,
                },
                best_path,
            )
            print(f"[OK] new best saved: {best_path}  best_val_acc={best_val:.3f}")

        # 每 6 个 epoch 额外保存一份（保留 best 路径不变）
        if epoch % 6 == 0:
            epoch_path = out_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "cfg": asdict(cfg),
                    "label2idx": label2idx,
                    "idx2label": idx2label,
                    "best_val_acc": best_val,
                },
                epoch_path,
            )
            print(f"[OK] epoch snapshot: {epoch_path}")

        # 达到目标准确率则提前停止（提前停时也保存 last 便于 resume）
        if args.target_acc > 0 and metrics["val_acc"] >= args.target_acc:
            last_path = out_dir / "checkpoint_last.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "cfg": asdict(cfg),
                    "label2idx": label2idx,
                    "idx2label": idx2label,
                    "best_val_acc": best_val,
                },
                last_path,
            )
            print(f"[EARLY STOP] val_acc={metrics['val_acc']:.3f} >= target {args.target_acc:.2f}, stop at epoch {epoch}.")
            break

    # 训练结束后汇总并打印训练集/验证集准确率
    if history:
        last = history[-1]
        print(f"[FINAL] train_acc={last['train_acc']:.4f} val_acc={last['val_acc']:.4f}")
        best_rec = max(history, key=lambda r: r["val_acc"])
        print(f"[FINAL] best_val_acc={best_rec['val_acc']:.4f} at epoch {history.index(best_rec)}")

    # 训练结束后画三张曲线图并保存
    if history and _HAS_MATPLOTLIB:
        try:
            epochs = list(range(len(history)))
            train_losses = [r["train_loss"] for r in history]
            val_losses = [r["val_loss"] for r in history]
            train_accs = [r["train_acc"] for r in history]
            val_accs = [r["val_acc"] for r in history]
            train_ranks = [r["train_rank"] for r in history]
            val_aucs = [r.get("val_auc_binary", float("nan")) for r in history]

            # 图 1：两个 loss
            plt.figure()
            plt.plot(epochs, train_losses, label="train_loss")
            plt.plot(epochs, val_losses, label="val_loss")
            plt.xlabel("epoch")
            plt.title("Train/Val Loss")
            plt.legend()
            plt.savefig(out_dir / "plot_loss.png")
            plt.close()

            # 图 2：两种准确率
            plt.figure()
            plt.plot(epochs, train_accs, label="train_acc")
            plt.plot(epochs, val_accs, label="val_acc")
            plt.xlabel("epoch")
            plt.title("Train/Val Accuracy")
            plt.legend()
            plt.savefig(out_dir / "plot_acc.png")
            plt.close()

            # 图 3：rank、val_acc、val_auc_binary
            plt.figure()
            plt.plot(epochs, train_ranks, label="train_rank")
            plt.plot(epochs, val_accs, label="val_acc")
            plt.plot(epochs, val_aucs, label="val_auc_binary")
            plt.xlabel("epoch")
            plt.title("Rank loss, Val Acc, Val AUC")
            plt.legend()
            plt.savefig(out_dir / "plot_rank_acc_auc.png")
            plt.close()

            print(f"[OK] 曲线已保存至 {out_dir}: plot_loss.png, plot_acc.png, plot_rank_acc_auc.png")
        except Exception as e:
            print(f"[WARN] 绘图失败，已跳过: {e}")
    elif history and not _HAS_MATPLOTLIB:
        print("[WARN] 未安装 matplotlib，跳过训练曲线绘图。可安装: pip install matplotlib")

    # export a small json for convenience
    (out_dir / "labels.json").write_text(json.dumps({"label2idx": label2idx, "idx2label": idx2label}, ensure_ascii=False, indent=2), encoding="utf-8")

    if getattr(args, "run_pseudo_iter", False):
        best_path = out_dir / "checkpoint_best.pt"
        if not best_path.exists():
            print(f"[WARN] --run_pseudo_iter 已开启但未找到 {best_path}，跳过伪标签迭代")
        else:
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            pseudo_out = Path(args.pseudo_out_dir) if getattr(args, "pseudo_out_dir", "") else out_dir / "pseudo_iter"
            from lab_anomaly.train.pseudo_label_iter import run_pseudo_iterations

            run_pseudo_iterations(
                ds=ds,
                model=model,
                label2idx=label2idx,
                idx2label=idx2label,
                cfg=cfg,
                device=device,
                out_dir=pseudo_out,
                val_idx=val_idx,
                iters=int(args.pseudo_iters),
                epochs_per_iter=int(args.pseudo_epochs_per_iter),
                kmeans_k=int(args.pseudo_kmeans_k),
                top_k=int(args.pseudo_top_k),
                bottom_k=int(args.pseudo_bottom_k),
                lambda_pseudo=float(args.pseudo_lambda),
                seed=int(args.seed),
                batch_size=int(args.batch_size),
                lr=float(args.pseudo_lr),
                weight_decay=float(args.pseudo_weight_decay),
                num_workers=int(args.pseudo_num_workers) or int(args.num_workers),
                negative_constraint=not getattr(args, "pseudo_no_negative_constraint", False),
                exclude_labels=exclude,
            )
            print(f"[OK] 伪标签迭代完成，输出目录: {pseudo_out}")

    print(f"[OK] done. outputs in: {out_dir}")


if __name__ == "__main__":
    main()

