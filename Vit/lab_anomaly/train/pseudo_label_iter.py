"""
伪标签迭代训练：用 KMeans 替代 FINCH，以覆盖度×分类分数选 top-k / bottom-k clip 作为伪标签，
与视频级 MIL 训练联合优化，提升弱监督场景下的 clip 级判别能力。
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
from torch.utils.data import DataLoader, Dataset, Subset

from lab_anomaly.models.mil_head import MILClassifier, MILHeadConfig

# 每轮迭代后评估 val_acc / binary AUC（与 train_known_classifier 一致）
try:
    from lab_anomaly.train.train_known_classifier import evaluate, split_train_val
except Exception:  # noqa: S110
    evaluate = None  # type: ignore[assignment]
    split_train_val = None


def require_sklearn():
    """检查 scikit-learn 是否可用"""
    try:
        from sklearn.cluster import KMeans  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("缺少 scikit-learn 依赖。请安装：pip install scikit-learn") from e


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
        exclude_labels: Optional[set[str]] = None,
        label_mapping: Optional[dict[str, str]] = None,
    ):
        self.embeddings_dir = Path(embeddings_dir)
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

        by_video_npy: dict[str, list[dict[str, Any]]] = {}
        by_video_npz: dict[str, dict[str, Any]] = {}
        for r in rows:
            label = _map_label(str(r.get("label", "")))
            if exclude_labels is not None and label in exclude_labels:
                continue
            fmt = str(r.get("save_format", "npy_per_clip"))
            vid = str(r["video_id"])
            if fmt == "npz_per_video":
                if vid not in by_video_npz:
                    by_video_npz[vid] = r
            else:
                by_video_npy.setdefault(vid, []).append(r)

        items: list[dict[str, Any]] = []
        for vid, clips in by_video_npy.items():
            clips = sorted(clips, key=lambda x: int(x.get("clip_index", 0)))
            label = _map_label(str(clips[0].get("label", "")))
            labels = [_map_label(str(x.get("label", ""))) for x in clips]
            if len(set(labels)) > 1:
                label = max(set(labels), key=lambda k: labels.count(k))
            items.append(
                {
                    "video_id": vid,
                    "label": label,
                    "save_format": "npy_per_clip",
                    "clip_paths": [resolve_path(str(x["embedding_path"])) for x in clips],
                    "num_clips": int(len(clips)),
                }
            )
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

        if expected_num_clips <= 0:
            expected_num_clips = int(max((int(x.get("num_clips", 0) or 0) for x in items), default=0))
        self.expected_num_clips = int(expected_num_clips)
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        it = self.items[idx]
        vid = it["video_id"]
        label = it["label"]
        fmt = str(it.get("save_format", "npy_per_clip"))

        embs: list[np.ndarray] = []
        if fmt == "npz_per_video":
            npz_path = str(it["npz_path"])
            npz = np.load(npz_path, allow_pickle=True)
            E = np.asarray(npz["embeddings"], dtype=np.float32)
            if E.ndim != 2 or E.shape[0] == 0:
                raise RuntimeError(f"video {vid} has invalid npz embeddings: {npz_path}")
            for i in range(min(E.shape[0], self.expected_num_clips)):
                embs.append(E[i])
        else:
            paths = list(it.get("clip_paths", []))
            for p in paths[: self.expected_num_clips]:
                embs.append(np.load(p).astype(np.float32))
        if not embs:
            raise RuntimeError(f"video {vid} has no embeddings")

        d = int(embs[0].shape[-1])
        n_valid = len(embs)
        n = self.expected_num_clips
        x = np.zeros((n, d), dtype=np.float32)
        mask = np.zeros((n,), dtype=np.bool_)
        x[:n_valid] = np.stack(embs, axis=0)
        mask[:n_valid] = True
        return {
            "video_id": vid,
            "label": label,
            "x": torch.from_numpy(x),
            "mask": torch.from_numpy(mask),
        }


def collate_videos(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """DataLoader collate_fn：将视频样本 stack 成 batch"""
    return {
        "video_id": [b["video_id"] for b in batch],
        "label": [b["label"] for b in batch],
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "mask": torch.stack([b["mask"] for b in batch], dim=0),
    }


class PseudoClipDataset(Dataset):
    """伪标签 clip 数据集：x=embedding，y=类别索引；is_neg=True 表示负约束（压低 y 类概率）"""
    def __init__(self, x: np.ndarray, y: np.ndarray, is_neg: Optional[np.ndarray] = None):
        assert x.ndim == 2 and y.ndim == 1 and x.shape[0] == y.shape[0]
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.is_neg = torch.from_numpy(is_neg.astype(np.uint8)) if is_neg is not None else torch.zeros(x.shape[0], dtype=torch.uint8)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx], self.is_neg[idx]


def build_pseudo_clips(
    ds: EmbeddingMILVideoDataset,
    model: MILClassifier,
    label2idx: dict[str, int],
    device: str,
    *,
    kmeans_k: int,
    top_k: int,
    bottom_k: int,
    seed: int,
    exclude_labels: set[str],
    max_clips_per_class: int = 0,
    negative_as_constraint: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    覆盖度普遍性分数 × 分类分数 × 离簇中心权重：
      - 对同一类视频的所有 clip embedding 做 KMeans，覆盖度 + 离中心越近权重越大
      - pseudo_score = 覆盖度 * clip_score * w_center
      - 对每条视频：top-k 作为正（该类）；bottom-k 为负样本：
        - negative_as_constraint=True：对目标类的负约束（不标 normal，训练时压低该类概率）
        - False：bottom-k 标为 normal（原逻辑）
    返回 (Xp, yp, is_neg, stats)，is_neg[i]=True 表示该 clip 为负约束（目标类为 yp[i]）。
    """
    require_sklearn()
    from sklearn.cluster import KMeans

    model.eval()
    rng = np.random.default_rng(int(seed))

    # 组织每个类别的视频索引
    class_to_vidx: dict[str, list[int]] = {}
    for i, it in enumerate(ds.items):
        lbl = str(it["label"])
        if lbl in exclude_labels:
            continue
        class_to_vidx.setdefault(lbl, []).append(i)

    normal_name = "normal"
    if normal_name not in label2idx:
        raise RuntimeError("label2idx missing 'normal' class; please include normal in training labels")

    pseudo_x: list[np.ndarray] = []
    pseudo_y: list[int] = []
    pseudo_is_neg: list[bool] = []
    stats: dict[str, Any] = {"classes": {}}

    with torch.no_grad():
        for cls, vidxs in class_to_vidx.items():
            if cls == normal_name:
                continue
            if cls not in label2idx:
                continue

            all_embs = []
            all_vid = []

            for vi in vidxs:
                sample = ds[vi]
                x = sample["x"].numpy()
                mask = sample["mask"].numpy().astype(bool)
                valid = x[mask]
                if valid.size == 0:
                    continue
                if max_clips_per_class and len(all_embs) >= max_clips_per_class:
                    break
                for ci in range(valid.shape[0]):
                    all_embs.append(valid[ci])
                    all_vid.append(sample["video_id"])

            if not all_embs:
                continue
            E = np.stack(all_embs, axis=0).astype(np.float32)  # (M,D)

            # KMeans
            try:
                km = KMeans(n_clusters=int(kmeans_k), random_state=int(seed), n_init="auto")
            except TypeError:  # pragma: no cover
                km = KMeans(n_clusters=int(kmeans_k), random_state=int(seed), n_init=10)
            clus = km.fit_predict(E)
            centers = km.cluster_centers_  # (K, D)

            # 覆盖度：unique video / total videos
            total_videos = len(set(all_vid))
            cover = {}
            for ci in range(int(kmeans_k)):
                vids = {all_vid[j] for j in np.where(clus == ci)[0]}
                cover[ci] = (len(vids) / max(1, total_videos))

            # 离簇中心权重：簇内到 centroid 距离做 min-max 归一化，w_center = 1 - norm_dist（越近越大）
            w_center = np.ones((E.shape[0],), dtype=np.float32)
            for ci in range(int(kmeans_k)):
                idx = np.where(clus == ci)[0]
                if len(idx) == 0:
                    continue
                centroid = centers[ci]
                dist = np.linalg.norm(E[idx] - centroid, axis=1).astype(np.float32)
                d_min, d_max = dist.min(), dist.max()
                if d_max > d_min:
                    norm_dist = (dist - d_min) / (d_max - d_min)
                else:
                    norm_dist = np.zeros_like(dist)
                w_center[idx] = 1.0 - norm_dist

            # clip 分类分数（clip-level 近似）
            emb_t = torch.from_numpy(E).to(device)
            emb_t = model.drop(model.norm(emb_t))
            if getattr(model, "pooling", "") == "topk" and hasattr(model, "clip_classifier"):
                logits = model.clip_classifier(emb_t)
            else:
                logits = model.classifier(emb_t)  # type: ignore[attr-defined]
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
            cls_idx = int(label2idx[cls])
            clip_score = probs[:, cls_idx]

            # 融合：覆盖度 × 分类分数 × 离中心权重（论文 universality × 预测分数）
            coverage_arr = np.array([cover[int(clus[i])] for i in range(E.shape[0])], dtype=np.float32)
            pseudo_score = coverage_arr * clip_score * w_center

            # 按 video 选 top/bottom
            by_video: dict[str, list[int]] = {}
            for i, vid in enumerate(all_vid):
                by_video.setdefault(vid, []).append(i)

            pos_cnt = 0
            neg_cnt = 0
            for vid, idxs in by_video.items():
                if not idxs:
                    continue
                # 打乱打破平局
                idxs = [idxs[i] for i in rng.permutation(len(idxs)).tolist()]
                s = pseudo_score[idxs]

                kpos = min(int(top_k), len(idxs))
                kneg = min(int(bottom_k), len(idxs))
                top_rel = np.argsort(-s)[:kpos]
                bot_rel = np.argsort(s)[:kneg]

                for r in top_rel.tolist():
                    pseudo_x.append(E[idxs[r]])
                    pseudo_y.append(cls_idx)
                    pseudo_is_neg.append(False)
                    pos_cnt += 1
                for r in bot_rel.tolist():
                    pseudo_x.append(E[idxs[r]])
                    # 负约束：目标类为 cls_idx，训练时压低该类概率；否则标为 normal
                    pseudo_y.append(cls_idx if negative_as_constraint else int(label2idx[normal_name]))
                    pseudo_is_neg.append(negative_as_constraint)
                    neg_cnt += 1

            stats["classes"][cls] = {
                "videos": int(len(vidxs)),
                "total_videos_seen": int(total_videos),
                "clips": int(E.shape[0]),
                "pos_added": int(pos_cnt),
                "neg_added": int(neg_cnt),
                "coverage_mean": float(np.mean(list(cover.values()))),
            }

    if not pseudo_x:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=bool), stats
    Xp = np.stack(pseudo_x, axis=0).astype(np.float32)
    yp = np.array(pseudo_y, dtype=np.int64)
    is_neg = np.array(pseudo_is_neg, dtype=bool)
    return Xp, yp, is_neg, stats


def load_checkpoint(path: Path, device: str) -> tuple[MILClassifier, dict[str, int], dict[int, str], MILHeadConfig]:
    """加载 train_known_classifier 的 checkpoint，返回模型、标签映射和配置"""
    ckpt = torch.load(str(path), map_location="cpu")
    cfg = MILHeadConfig(**ckpt["cfg"])
    model = MILClassifier(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    label2idx = {str(k): int(v) for k, v in ckpt.get("label2idx", {}).items()}
    idx2label = {int(k): str(v) for k, v in ckpt.get("idx2label", {}).items()}
    model.eval()
    return model, label2idx, idx2label, cfg


def run_pseudo_iterations(
    ds: Dataset,
    model: MILClassifier,
    label2idx: dict[str, int],
    idx2label: dict[int, str],
    cfg: MILHeadConfig,
    device: str,
    *,
    out_dir: Path,
    val_idx: list[int],
    iters: int = 3,
    epochs_per_iter: int = 2,
    kmeans_k: int = 32,
    top_k: int = 2,
    bottom_k: int = 2,
    lambda_pseudo: float = 0.5,
    seed: int = 42,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    num_workers: int = 0,
    negative_constraint: bool = True,
    exclude_labels: Optional[set[str]] = None,
) -> None:
    """
    可复用的伪标签迭代循环：不加载 checkpoint、不建 ds，由调用方传入。
    ds 需有 .items 且 __getitem__ 返回 dict(x, mask, label, video_id)。
    """
    exclude_labels = exclude_labels or set()
    exclude_labels = set(exclude_labels) | {"unknown"}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    normal_idx = label2idx.get("normal", -1)

    val_loader = None
    if val_idx:
        val_ds = Subset(ds, val_idx)
        val_loader = DataLoader(
            val_ds,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            collate_fn=collate_videos,
            pin_memory=device.startswith("cuda"),
        )

    iter_metrics_log: list[dict[str, Any]] = []
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    ce = nn.CrossEntropyLoss()

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    for it in range(int(iters)):
        Xp, yp, is_neg, stats = build_pseudo_clips(
            ds,
            model,
            label2idx=label2idx,
            device=device,
            kmeans_k=int(kmeans_k),
            top_k=int(top_k),
            bottom_k=int(bottom_k),
            seed=int(seed) + it,
            exclude_labels=exclude_labels,
            negative_as_constraint=negative_constraint,
        )

        (out_dir / f"pseudo_stats_iter_{it:02d}.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        n_neg = int(np.sum(is_neg))
        print(f"[iter {it}] pseudo clips: {int(yp.shape[0])} (pos={yp.shape[0]-n_neg}, neg_constraint={n_neg})")
        if yp.shape[0] == 0:
            print("[WARN] no pseudo clips generated; stop early.")
            break

        pseudo_loader = DataLoader(
            PseudoClipDataset(Xp, yp, is_neg),
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=int(num_workers),
            drop_last=False,
            pin_memory=device.startswith("cuda"),
        )
        video_loader = DataLoader(
            ds,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=int(num_workers),
            collate_fn=collate_videos,
            drop_last=False,
            pin_memory=device.startswith("cuda"),
        )

        for ep in range(int(epochs_per_iter)):
            model.train()
            pseudo_it = iter(pseudo_loader)
            video_it = iter(video_loader)
            steps = max(len(pseudo_loader), len(video_loader))
            loss_sum = 0.0
            for _ in range(steps):
                try:
                    xb, yb, is_neg_b = next(pseudo_it)
                except StopIteration:
                    pseudo_it = iter(pseudo_loader)
                    xb, yb, is_neg_b = next(pseudo_it)
                try:
                    vb = next(video_it)
                except StopIteration:
                    video_it = iter(video_loader)
                    vb = next(video_it)

                xb = xb.to(device)
                yb = yb.to(device)
                vx = vb["x"].to(device)
                vmask = vb["mask"].to(device)
                vy = torch.tensor([label2idx[l] for l in vb["label"]], device=device, dtype=torch.long)

                opt.zero_grad(set_to_none=True)
                vlogits, _ = model(vx, mask=vmask, y=vy if model.pooling == "topk" else None)
                loss_video = ce(vlogits, vy)

                xb2 = model.drop(model.norm(xb))
                if getattr(model, "pooling", "") == "topk" and hasattr(model, "clip_classifier"):
                    plogits = model.clip_classifier(xb2)
                else:
                    plogits = model.classifier(xb2)  # type: ignore[attr-defined]
                pos_mask = ~(is_neg_b.bool())
                loss_pseudo = torch.tensor(0.0, device=device)
                if pos_mask.any():
                    loss_pseudo = loss_pseudo + ce(plogits[pos_mask], yb[pos_mask])
                if (~pos_mask).any():
                    neg_mask = ~pos_mask
                    probs = F.softmax(plogits[neg_mask], dim=-1)
                    p_target = probs.gather(1, yb[neg_mask].unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0 - 1e-6)
                    loss_pseudo = loss_pseudo + (-torch.log(1.0 - p_target).mean())

                loss = loss_video + float(lambda_pseudo) * loss_pseudo
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt.step()
                loss_sum += float(loss.item())

            print(f"[iter {it}] epoch {ep:02d} loss={loss_sum / max(1, steps):.4f}")

        ckpt_path = out_dir / f"checkpoint_iter_{it:02d}.pt"
        torch.save(
            {
                "iter": it,
                "model_state": model.state_dict(),
                "cfg": asdict(cfg),
                "label2idx": label2idx,
                "idx2label": idx2label,
            },
            ckpt_path,
        )
        print(f"[OK] saved: {ckpt_path}")

        if val_loader is not None and evaluate is not None:
            metrics = evaluate(
                model,
                val_loader,
                label2idx=label2idx,
                device=device,
                eval_with_gt_topk=False,
                ce_weight=None,
                normal_idx=normal_idx if normal_idx >= 0 else None,
                idx2label=idx2label,
            )
            rec = {"iter": it, "val_acc": metrics["val_acc"], "val_loss": metrics["val_loss"]}
            if "val_auc_binary" in metrics:
                rec["val_auc_binary"] = metrics["val_auc_binary"]
            if "val_acc_binary_0.5" in metrics:
                rec["val_acc_binary_0.5"] = metrics["val_acc_binary_0.5"]
            iter_metrics_log.append(rec)
            print(f"[iter {it}] val_acc={metrics['val_acc']:.3f} val_loss={metrics['val_loss']:.4f}", end="")
            if "val_auc_binary" in metrics:
                auc = metrics["val_auc_binary"]
                print(f" val_auc_binary={auc:.3f}" if not (isinstance(auc, float) and np.isnan(auc)) else " val_auc_binary=nan", end="")
            print()

    if iter_metrics_log:
        (out_dir / "pseudo_metrics_iters.json").write_text(
            json.dumps(iter_metrics_log, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[OK] metrics log: {out_dir / 'pseudo_metrics_iters.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="伪标签迭代训练：覆盖度×分类分数（KMeans 替代 FINCH 思路）")
    ap.add_argument("--embeddings_dir", type=str, default="lab_dataset/derived/embeddings")
    ap.add_argument("--ckpt", type=str, required=True, help="train_known_classifier 的 checkpoint_best.pt 或 checkpoint_last.pt")
    ap.add_argument("--out_dir", type=str, default="lab_dataset/derived/pseudo_iter")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--epochs_per_iter", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--lambda_pseudo", type=float, default=0.5, help="伪标签 clip 损失权重")
    ap.add_argument("--no_negative_constraint", action="store_false", dest="negative_constraint", default=True, help="加此参数则 bottom-k 标为 normal（旧逻辑）；默认做对目标类的负约束")
    ap.add_argument("--kmeans_k", type=int, default=32)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--bottom_k", type=int, default=2)
    ap.add_argument("--expected_num_clips", type=int, default=0)
    ap.add_argument("--exclude_unknown", action="store_true")
    ap.add_argument("--label_mapping", type=str, default="Robbery:steal", help="标签映射，格式 key:value 多个用逗号分隔，如 Robbery:steal。需与 checkpoint 训练时一致")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例，用于每轮迭代后计算 val_acc 与 binary AUC")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="", help="cuda/cpu（默认自动）")
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    device = args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, label2idx, idx2label, cfg = load_checkpoint(Path(args.ckpt), device=device)

    exclude = {"unknown"} if args.exclude_unknown else set()
    label_mapping: dict[str, str] = {}
    for part in (args.label_mapping or "").strip().split(","):
        part = part.strip()
        if ":" in part:
            k, v = part.split(":", 1)
            label_mapping[k.strip()] = v.strip()
    if label_mapping:
        print(f"[LABEL_MAPPING] {label_mapping}（需与 checkpoint 训练时一致）")
    ds = EmbeddingMILVideoDataset(
        embeddings_dir=args.embeddings_dir,
        expected_num_clips=int(args.expected_num_clips),
        exclude_labels=exclude if exclude else None,
        label_mapping=label_mapping if label_mapping else None,
    )
    if len(ds) < 2:
        raise RuntimeError("dataset too small")

    val_ratio = float(getattr(args, "val_ratio", 0.2))
    if split_train_val:
        _train_idx, val_idx = split_train_val(ds.items, val_ratio=val_ratio, seed=int(args.seed))
    else:
        val_idx = list(range(min(len(ds) // 5, len(ds) - 1))) if len(ds) > 1 else []

    run_pseudo_iterations(
        ds=ds,
        model=model,
        label2idx=label2idx,
        idx2label=idx2label,
        cfg=cfg,
        device=device,
        out_dir=out_dir,
        val_idx=val_idx,
        iters=int(args.iters),
        epochs_per_iter=int(args.epochs_per_iter),
        kmeans_k=int(args.kmeans_k),
        top_k=int(args.top_k),
        bottom_k=int(args.bottom_k),
        lambda_pseudo=float(args.lambda_pseudo),
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        num_workers=int(args.num_workers),
        negative_constraint=getattr(args, "negative_constraint", True),
        exclude_labels=exclude,
    )

    print(f"[OK] done. outputs in: {out_dir}")


if __name__ == "__main__":
    main()

