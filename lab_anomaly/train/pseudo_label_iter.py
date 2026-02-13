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
from torch.utils.data import DataLoader, Dataset

from lab_anomaly.models.mil_head import MILClassifier, MILHeadConfig


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
    ):
        self.embeddings_dir = Path(embeddings_dir)
        meta_path = self.embeddings_dir / "embeddings_meta.jsonl"
        if not meta_path.exists():
            raise FileNotFoundError(f"cannot find embeddings meta: {meta_path}")

        rows = read_jsonl(meta_path)

        by_video_npy: dict[str, list[dict[str, Any]]] = {}
        by_video_npz: dict[str, dict[str, Any]] = {}
        for r in rows:
            label = str(r.get("label", ""))
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
                    "num_clips": int(len(clips)),
                }
            )
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
    """伪标签 clip 数据集：x=embedding，y=类别索引"""
    def __init__(self, x: np.ndarray, y: np.ndarray):
        assert x.ndim == 2 and y.ndim == 1 and x.shape[0] == y.shape[0]
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


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
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    覆盖度普遍性分数 × 分类分数：
      - 对同一类视频的所有 clip embedding 做 KMeans
      - 覆盖度 = cluster 内 unique video 数 / 该类视频总数
      - clip 分数 = 分类器对该类的概率（clip-level 近似）
      - pseudo_score = 覆盖度 * clip_score
      - 对每条视频：top-k 作为正（该类），bottom-k 作为负（normal）
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

            # 覆盖度：unique video / total videos
            total_videos = len(set(all_vid))
            cover = {}
            for ci in range(int(kmeans_k)):
                vids = {all_vid[j] for j in np.where(clus == ci)[0]}
                cover[ci] = (len(vids) / max(1, total_videos))

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

            pseudo_score = np.array([cover[int(clus[i])] for i in range(E.shape[0])], dtype=np.float32) * clip_score

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
                    pos_cnt += 1
                for r in bot_rel.tolist():
                    pseudo_x.append(E[idxs[r]])
                    pseudo_y.append(int(label2idx[normal_name]))
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
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64), stats
    Xp = np.stack(pseudo_x, axis=0).astype(np.float32)
    yp = np.array(pseudo_y, dtype=np.int64)
    return Xp, yp, stats


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
    ap.add_argument("--kmeans_k", type=int, default=32)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--bottom_k", type=int, default=2)
    ap.add_argument("--expected_num_clips", type=int, default=0)
    ap.add_argument("--exclude_unknown", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="", help="cuda/cpu（默认自动）")
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    device = args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, label2idx, idx2label, cfg = load_checkpoint(Path(args.ckpt), device=device)
    model.train()

    exclude = {"unknown"} if args.exclude_unknown else set()
    ds = EmbeddingMILVideoDataset(
        embeddings_dir=args.embeddings_dir,
        expected_num_clips=int(args.expected_num_clips),
        exclude_labels=exclude if exclude else None,
    )
    if len(ds) < 2:
        raise RuntimeError("dataset too small")

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    ce = nn.CrossEntropyLoss()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    for it in range(int(args.iters)):
        Xp, yp, stats = build_pseudo_clips(
            ds,
            model,
            label2idx=label2idx,
            device=device,
            kmeans_k=int(args.kmeans_k),
            top_k=int(args.top_k),
            bottom_k=int(args.bottom_k),
            seed=int(args.seed) + it,
            exclude_labels=exclude | {"unknown"},
        )

        (out_dir / f"pseudo_stats_iter_{it:02d}.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[iter {it}] pseudo clips: {int(yp.shape[0])}")
        if yp.shape[0] == 0:
            print("[WARN] no pseudo clips generated; stop early.")
            break

        pseudo_loader = DataLoader(
            PseudoClipDataset(Xp, yp),
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
            drop_last=False,
            pin_memory=device.startswith("cuda"),
        )
        video_loader = DataLoader(
            ds,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
            collate_fn=collate_videos,
            drop_last=False,
            pin_memory=device.startswith("cuda"),
        )

        for ep in range(int(args.epochs_per_iter)):
            model.train()
            pseudo_it = iter(pseudo_loader)
            video_it = iter(video_loader)
            steps = max(len(pseudo_loader), len(video_loader))

            loss_sum = 0.0
            for _ in range(steps):
                try:
                    xb, yb = next(pseudo_it)
                except StopIteration:
                    pseudo_it = iter(pseudo_loader)
                    xb, yb = next(pseudo_it)

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
                loss_pseudo = ce(plogits, yb)

                loss = loss_video + float(args.lambda_pseudo) * loss_pseudo
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

    print(f"[OK] done. outputs in: {out_dir}")


if __name__ == "__main__":
    main()

