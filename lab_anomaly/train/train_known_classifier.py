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
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lab_anomaly.models.mil_head import MILClassifier, MILHeadConfig
from lab_anomaly.models.ranking_loss import mil_ranking_loss


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
    ):
        self.embeddings_dir = Path(embeddings_dir)
        meta_path = self.embeddings_dir / "embeddings_meta.jsonl"
        if not meta_path.exists():
            raise FileNotFoundError(f"cannot find embeddings meta: {meta_path}")

        rows = read_jsonl(meta_path)

        # 支持两种 embedding 缓存格式：
        # - npy_per_clip：jsonl 一条记录对应一个 clip
        # - npz_per_video：jsonl 一条记录对应一个视频（npz 内包含全部 clips embeddings）
        by_video_npy: dict[str, list[dict[str, Any]]] = {}
        by_video_npz: dict[str, dict[str, Any]] = {}

        for r in rows:
            label = str(r.get("label", ""))
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
            label = str(clips[0].get("label", ""))
            labels = [str(x.get("label", "")) for x in clips]
            if len(set(labels)) > 1:
                label = max(set(labels), key=lambda k: labels.count(k))
            items.append(
                {
                    "video_id": vid,
                    "label": label,
                    "save_format": "npy_per_clip",
                    "clip_paths": [str(x["embedding_path"]) for x in clips],
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
                    "label": str(r.get("label", "")),
                    "save_format": "npz_per_video",
                    "npz_path": str(r["embedding_path"]),
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

        return {
            "video_id": vid,
            "label": label,
            "x": torch.from_numpy(x),  # (N,D)
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
) -> dict[str, float]:
    """在验证集上计算 val_loss 与 val_acc"""
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()
    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        y = torch.tensor([label2idx[l] for l in batch["label"]], device=device, dtype=torch.long)
        logits, _ = model(x, mask=mask, y=y if model.pooling == "topk" else None)
        loss = ce(logits, y)
        pred = logits.argmax(dim=-1)
        total += int(y.numel())
        correct += int((pred == y).sum().item())
        loss_sum += float(loss.item()) * int(y.numel())
    return {
        "val_loss": loss_sum / max(1, total),
        "val_acc": correct / max(1, total),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="已知异常：视频级标签 MIL 聚合分类训练（基于 clip embeddings 缓存）")
    ap.add_argument("--embeddings_dir", type=str, default = r"C:\Users\Administrator\Desktop\Vit\lab_dataset\derived\embeddings")
    ap.add_argument("--out_dir", type=str, default= r"C:\Users\Administrator\Desktop\Vit\lab_dataset\derived\known_classifier")
    ap.add_argument("--exclude_unknown", action="store_true", help="排除 label=unknown 的样本")

    ap.add_argument("--pooling", type=str, default="attn", choices=["attn", "topk"])
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--attn_hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no_layernorm", action="store_true", help="默认启用 LayerNorm；加此参数可关闭")

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="", help="cuda/cpu（默认自动）")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--expected_num_clips", type=int, default=0, help="每视频 clip 数（0=自动推断最大值）")
    ap.add_argument("--resume", type=str, default="", help="从 checkpoint.pt 恢复")

    # --- MIL 排序损失参数（借鉴 CVPR 2018 论文） ---
    ap.add_argument("--use_anomaly_branch", action="store_true", help="启用异常分数分支 + MIL 排序损失")
    ap.add_argument("--lambda_rank", type=float, default=1.0, help="排序损失在总 loss 中的权重")
    ap.add_argument("--lambda_sparse", type=float, default=8e-5, help="稀疏约束权重（论文建议 8e-5）")
    ap.add_argument("--lambda_smooth", type=float, default=8e-5, help="时间平滑约束权重（论文建议 8e-5）")
    ap.add_argument("--anomaly_hidden", type=int, default=512, help="异常分数分支隐层维度")
    ap.add_argument("--normal_label", type=str, default="normal", help="normal 类别的标签名，用于区分正负 bag")

    args = ap.parse_args()

    device = args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude = {"unknown"} if args.exclude_unknown else None
    ds = EmbeddingMILVideoDataset(
        embeddings_dir=args.embeddings_dir,
        expected_num_clips=int(args.expected_num_clips),
        exclude_labels=exclude,
    )
    if len(ds) <= 1:
        raise RuntimeError("dataset too small; please check embeddings_dir and labels.")

    # label space
    labels = sorted({it["label"] for it in ds.items})
    if "normal" in labels:
        labels = ["normal"] + [x for x in labels if x != "normal"]
    label2idx = {l: i for i, l in enumerate(labels)}
    idx2label = {i: l for l, i in label2idx.items()}

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
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=collate,
        pin_memory=device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate,
        pin_memory=device.startswith("cuda"),
    )

    ce = nn.CrossEntropyLoss()
    use_rank = bool(args.use_anomaly_branch) and normal_idx >= 0
    if use_rank:
        print(f"[RANK] MIL ranking loss enabled: lambda_rank={args.lambda_rank}, "
              f"lambda_sparse={args.lambda_sparse}, lambda_smooth={args.lambda_smooth}")

    for epoch in range(start_epoch, int(args.epochs)):
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
            logits, details = model(x, mask=mask, y=y if model.pooling == "topk" else None, return_details=True)
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

        metrics = evaluate(model, val_loader, label2idx=label2idx, device=device)
        rank_info = f" rank={train_rank:.4f}" if use_rank else ""
        print(
            f"epoch {epoch:03d} | train_loss={train_loss:.4f}{rank_info} acc={train_acc:.3f} "
            f"| val_loss={metrics['val_loss']:.4f} acc={metrics['val_acc']:.3f}"
        )

        # save last
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

    # export a small json for convenience
    (out_dir / "labels.json").write_text(json.dumps({"label2idx": label2idx, "idx2label": idx2label}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] done. outputs in: {out_dir}")


if __name__ == "__main__":
    main()

