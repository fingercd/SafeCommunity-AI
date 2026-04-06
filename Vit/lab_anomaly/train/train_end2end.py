"""
端到端训练：VideoMAE v2 + MIL 头（二分类：normal vs anomaly），三阶段渐进解冻。
在 PyCharm 中直接运行本文件；所有路径与超参在下方 CONFIG 中修改。
"""
from __future__ import annotations
import contextlib
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from lab_anomaly.data.clip_dataset import VideoClipDataset
from lab_anomaly.models.mil_head import MILClassifier, MILHeadConfig
from lab_anomaly.models.ranking_loss import mil_ranking_loss
from lab_anomaly.models.vit_video_encoder import VideoMAEv2Encoder, VideoMAEv2EncoderConfig

try:
    from sklearn.metrics import confusion_matrix, roc_auc_score
except ImportError:
    roc_auc_score = None
    confusion_matrix = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# ============ 可调：路径与训练参数（主要改这里）=================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CONFIG: dict[str, Any] = {
    "dataset_root": str(PROJECT_ROOT / "lab_dataset"),
    "labels_csv":   str(PROJECT_ROOT / "lab_dataset" / "labels" / "video_labels.csv"),
    # 预切片目录（内含 manifest.json）。训练只读已切好的 .npz，不再从视频现切。请先运行 lab_anomaly/tools/precompute_clips.py。
    "preclip_root": str(PROJECT_ROOT / "lab_dataset" / "derived" / "preclips"),
    "out_dir": str(PROJECT_ROOT / "lab_dataset" / "derived" / "end2end_classifier"),
    "yaml_config": str(PROJECT_ROOT / "lab_anomaly" / "configs" / "train_end2end.yaml"),
    "exclude_unknown": True,
    "normal_label": "normal",
    "frames_per_clip": 16,
    "interval_sec": 8.0,
    "max_clips_per_video": 16,
    "encoder_model_name": "OpenGVLab/VideoMAEv2-Base",
    "encoder_use_half": True,
    # VideoMAE 编码时分块大小；各 stage 可单独写 micro_batch_clips 覆盖本值
    "micro_batch_clips": 40,
    "grad_accum_steps": 2,
    "clip_encode_eval_mode_when_frozen": True,
    "stages": [
        {"name": "head_only", "epochs": 30, "lr": 1e-3, "unfreeze_blocks": 0, "weight_decay": 0.01},
        {"name": "unfreeze_2", "epochs": 20, "lr": 5e-5, "unfreeze_blocks": 2, "weight_decay": 0.01},
        {"name": "unfreeze_4", "epochs": 15, "lr": 2e-5, "unfreeze_blocks": 4, "weight_decay": 0.01},
    ],
    "batch_size": 20,
    "val_ratio": 0.2,
    "seed": 420,
    "num_workers": 20,
    "target_acc": 0.0,
    "mil_pooling": "attn",
    "mil_topk": 3,
    "mil_attn_hidden": 256,
    "mil_dropout": 0.35,
    "mil_use_layernorm": True,
    "use_anomaly_branch": True,
    "lambda_rank": 0.7,
    "lambda_sparse": 8e-5,
    "lambda_smooth": 8e-5,
    "anomaly_hidden": 256,
    "class_weight": True,
    "device": "",
    # 训练时进度：False=每个 batch 单独一行，loss 都能看到；True=同一行刷新（像进度条）
    "progress_one_line_refresh": False,
    # 训练结束后是否保存验证集混淆矩阵（需 matplotlib）
    "save_val_confusion_matrix": True,
    # SwanLab（pip install swanlab；未安装则跳过）
    "swanlab_enabled": False,
    "swanlab_project": "lab_anomaly_end2end",
    "swanlab_experiment": "",
    "swanlab_description": "",
    "swanlab_mode": "",
    "swanlab_log_every_n_batches": 1,
}

IDX2LABEL = {0: "normal", 1: "anomaly"}
LABEL2IDX = {"normal": 0, "anomaly": 1}

_swanlab_active = False


def _load_yaml_merge(cfg: dict[str, Any], path: Path) -> None:
    if not path.is_file():
        return
    try:
        import yaml
    except ImportError:
        return
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if isinstance(data, dict):
        cfg.update(data)


def _resolve_paths_under_project(cfg: dict[str, Any], project_root: Path) -> None:
    """把 YAML 里的相对路径统一解析到项目根目录，避免受 PyCharm 工作目录影响。"""
    root = project_root.resolve()
    for key in ("dataset_root", "labels_csv", "out_dir", "preclip_root"):
        value = cfg.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        p = Path(text)
        if not p.is_absolute():
            cfg[key] = str((root / p).resolve())


def split_train_val(n: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    idxs = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)
    n_val = int(round(n * float(val_ratio)))
    val = idxs[:n_val]
    train = idxs[n_val:]
    return train, val


class VideoMILBinaryDataset(Dataset):
    """每个样本 = 一个视频的全部 clip（原始帧），二分类标签。"""

    def __init__(
        self,
        clip_ds: VideoClipDataset,
        video_indices: list[int],
        normal_label: str,
    ):
        self.clip_ds = clip_ds
        self.video_indices = list(video_indices)
        self.normal_label = str(normal_label).strip().lower()

    def __len__(self) -> int:
        return len(self.video_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        vi = int(self.video_indices[idx])
        row = self.clip_ds.rows[vi]
        clips = self.clip_ds.get_video_clips(vi)
        lbl = str(row.label).strip().lower()
        y = 0 if lbl == self.normal_label else 1
        frames_nested = [c.frames for c in clips]
        return {
            "video_id": row.video_id,
            "clips": frames_nested,
            "y": y,
            "num_clips": len(frames_nested),
        }


def collate_video_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_n = max(int(b["num_clips"]) for b in batch)
    B = len(batch)
    mask = torch.zeros((B, max_n), dtype=torch.bool)
    clips_nested: list[list[list[np.ndarray]]] = []
    ys: list[int] = []
    vids: list[str] = []
    for i, b in enumerate(batch):
        vids.append(str(b["video_id"]))
        ys.append(int(b["y"]))
        n = int(b["num_clips"])
        clips_nested.append(b["clips"])
        mask[i, :n] = True
    labels_str = [IDX2LABEL[y] for y in ys]
    return {
        "video_id": vids,
        "clips_nested": clips_nested,
        "mask": mask,
        "label": labels_str,
        "y": torch.tensor(ys, dtype=torch.long),
    }


class BalancedVideoBatchSampler(torch.utils.data.Sampler[list[int]]):
    """保证每个 batch 至少 1 个 normal + 1 个 anomaly（用于 ranking loss）。"""

    def __init__(
        self,
        train_vis: list[int],
        clip_ds: VideoClipDataset,
        normal_label: str,
        batch_size: int,
        seed: int,
        epoch_ref: Optional[list[int]] = None,
    ):
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch_ref = epoch_ref
        normal_label = str(normal_label).strip().lower()

        self.normal_idx: list[int] = []
        self.anom_idx: list[int] = []
        # train_vis[j] 为全局视频下标；j 为 Dataset 下标
        for j, vi in enumerate(train_vis):
            lbl = str(clip_ds.rows[vi].label).strip().lower()
            if lbl == normal_label:
                self.normal_idx.append(j)
            else:
                self.anom_idx.append(j)

    def __iter__(self):
        eff = self.seed + (self.epoch_ref[0] if self.epoch_ref else 0)
        rnd = random.Random(eff)
        n_list = list(self.normal_idx)
        a_list = list(self.anom_idx)
        rnd.shuffle(n_list)
        rnd.shuffle(a_list)
        batches: list[list[int]] = []
        while n_list or a_list:
            batch: list[int] = []
            if n_list and a_list:
                batch.append(n_list.pop())
                batch.append(a_list.pop())
            elif n_list:
                batch.append(n_list.pop())
            else:
                batch.append(a_list.pop())
            while len(batch) < self.batch_size and (n_list or a_list):
                if n_list and a_list:
                    batch.append(n_list.pop() if rnd.random() < 0.5 else a_list.pop())
                elif n_list:
                    batch.append(n_list.pop())
                else:
                    batch.append(a_list.pop())
            rnd.shuffle(batch)
            batches.append(batch)
        rnd.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return (len(self.normal_idx) + len(self.anom_idx) + self.batch_size - 1) // self.batch_size


@torch.no_grad()
def evaluate(
    encoder: VideoMAEv2Encoder,
    mil: MILClassifier,
    loader: DataLoader,
    device: str,
    weight: Optional[torch.Tensor],
    backbone_frozen: bool,
    micro_bs: int,
    clip_eval_mode_when_frozen: bool,
    return_predictions: bool = False,
    use_amp: bool = False,
) -> dict[str, Any]:
    encoder.backbone.eval()
    mil.eval()
    ce = nn.CrossEntropyLoss(weight=weight)
    total = correct = 0
    loss_sum = 0.0
    all_y: list[int] = []
    all_pred: list[int] = []
    all_p_abn: list[float] = []
    normal_idx = LABEL2IDX["normal"]

    amp_ctx: Any = contextlib.nullcontext()
    if use_amp and device.startswith("cuda"):
        amp_ctx = torch.amp.autocast("cuda")

    for batch in loader:
        mask = batch["mask"].to(device)
        y = batch["y"].to(device)
        with amp_ctx:
            x = _encode_nested_clips(
                encoder,
                batch["clips_nested"],
                mask,
                device,
                backbone_frozen,
                micro_bs,
                clip_eval_mode_when_frozen,
            )
            logits, _ = mil(x, mask=mask, y=None)
            loss = ce(logits, y)
        pred = logits.argmax(dim=-1)
        total += int(y.numel())
        correct += int((pred == y).sum().item())
        loss_sum += float(loss.item()) * int(y.numel())
        all_y.extend(y.cpu().tolist())
        all_pred.extend(pred.cpu().tolist())
        if roc_auc_score is not None:
            probs = F.softmax(logits, dim=-1)
            all_p_abn.extend((1.0 - probs[:, normal_idx]).cpu().tolist())

    out: dict[str, Any] = {
        "val_loss": loss_sum / max(1, total),
        "val_acc": correct / max(1, total),
    }
    n = len(all_y)
    if roc_auc_score is not None and n > 0 and len(all_p_abn) == n:
        y_bin = np.array([1 if yi != normal_idx else 0 for yi in all_y], dtype=np.int64)
        p_abn = np.array(all_p_abn, dtype=np.float64)
        if np.unique(y_bin).size == 2:
            out["val_auc_binary"] = float(roc_auc_score(y_bin, p_abn))
        else:
            out["val_auc_binary"] = float("nan")
    if return_predictions:
        out["y_true"] = list(all_y)
        out["y_pred"] = list(all_pred)
    return out


def _encode_nested_clips(
    encoder: VideoMAEv2Encoder,
    clips_nested: list[list[list[np.ndarray]]],
    mask: torch.Tensor,
    device: str,
    backbone_frozen: bool,
    micro_bs: int,
    clip_eval_mode_when_frozen: bool,
) -> torch.Tensor:
    B, max_n = mask.shape
    D = encoder.embedding_dim
    flat: list[list[np.ndarray]] = []
    for b in range(B):
        for t in range(max_n):
            if mask[b, t]:
                flat.append(clips_nested[b][t])
    if not flat:
        return torch.zeros(B, max_n, D, device=device)

    if backbone_frozen and clip_eval_mode_when_frozen:
        encoder.backbone.eval()

    chunks: list[torch.Tensor] = []
    for i in range(0, len(flat), max(1, micro_bs)):
        chunk = flat[i : i + max(1, micro_bs)]
        if backbone_frozen:
            with torch.no_grad():
                out = encoder(chunk)
        else:
            out = encoder(chunk)
        chunks.append(out)
    e_flat = torch.cat(chunks, dim=0)
    dtype = e_flat.dtype
    x = torch.zeros(B, max_n, D, device=device, dtype=dtype)
    k = 0
    for b in range(B):
        for t in range(max_n):
            if mask[b, t]:
                x[b, t] = e_flat[k]
                k += 1
    return x


def _training_step(
    encoder: VideoMAEv2Encoder,
    mil: MILClassifier,
    batch: dict[str, Any],
    device: str,
    opt: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    ce: nn.Module,
    backbone_frozen: bool,
    micro_bs: int,
    clip_eval_mode_when_frozen: bool,
    use_rank: bool,
    normal_idx: int,
    lambda_rank: float,
    lambda_sparse: float,
    lambda_smooth: float,
) -> tuple[float, float, torch.Tensor]:
    mask = batch["mask"].to(device)
    y = batch["y"].to(device)
    opt.zero_grad(set_to_none=True)

    def _forward_with_x(x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return mil(x, mask=mask, y=None, return_details=True)

    if scaler is not None:
        with torch.amp.autocast("cuda"):
            x = _encode_nested_clips(
                encoder,
                batch["clips_nested"],
                mask,
                device,
                backbone_frozen,
                micro_bs,
                clip_eval_mode_when_frozen,
            )
            logits, details = _forward_with_x(x)
            loss_ce = ce(logits, y)
            loss_rank = torch.tensor(0.0, device=device)
            if use_rank and "anomaly_scores" in details:
                s = details["anomaly_scores"]
                is_normal = y == normal_idx
                is_anom = ~is_normal
                if is_anom.any() and is_normal.any():
                    loss_rank = mil_ranking_loss(
                        s[is_anom],
                        s[is_normal],
                        mask_pos=mask[is_anom],
                        mask_neg=mask[is_normal],
                        lambda_sparse=lambda_sparse,
                        lambda_smooth=lambda_smooth,
                    )
            loss = loss_ce + lambda_rank * loss_rank
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        logits_train = logits.detach()
    else:
        x = _encode_nested_clips(
            encoder,
            batch["clips_nested"],
            mask,
            device,
            backbone_frozen,
            micro_bs,
            clip_eval_mode_when_frozen,
        )
        logits, details = _forward_with_x(x)
        loss_ce = ce(logits, y)
        loss_rank = torch.tensor(0.0, device=device)
        if use_rank and "anomaly_scores" in details:
            s = details["anomaly_scores"]
            is_normal = y == normal_idx
            is_anom = ~is_normal
            if is_anom.any() and is_normal.any():
                loss_rank = mil_ranking_loss(
                    s[is_anom],
                    s[is_normal],
                    mask_pos=mask[is_anom],
                    mask_neg=mask[is_normal],
                    lambda_sparse=lambda_sparse,
                    lambda_smooth=lambda_smooth,
                )
        loss = loss_ce + lambda_rank * loss_rank
        loss.backward()
        opt.step()
        logits_train = logits.detach()

    return float(loss_ce.item()), float(loss_rank.item()), logits_train


def _p(msg: str) -> None:
    print(msg, flush=True)
    sys.stdout.flush()


def _swanlab_try_init(cfg: dict[str, Any]) -> None:
    global _swanlab_active
    _swanlab_active = False
    if not bool(cfg.get("swanlab_enabled")):
        return
    try:
        import swanlab
    except ImportError:
        _p("[swanlab] 已开启 swanlab_enabled，但未安装。请在当前环境安装：pip install swanlab")
        return
    init_kw: dict[str, Any] = {
        "project": (str(cfg.get("swanlab_project") or "lab_anomaly_end2end").strip() or "lab_anomaly_end2end"),
        "config": _swanlab_config_snapshot(cfg),
    }
    exp = str(cfg.get("swanlab_experiment") or "").strip()
    if exp:
        init_kw["experiment_name"] = exp
    desc = str(cfg.get("swanlab_description") or "").strip()
    if desc:
        init_kw["description"] = desc
    mode = str(cfg.get("swanlab_mode") or "").strip()
    if mode:
        init_kw["mode"] = mode
    try:
        swanlab.init(**init_kw)
        _swanlab_active = True
        _p(f"[swanlab] 已初始化 project={init_kw['project']!r}")
    except Exception as e:  # pragma: no cover
        _p(f"[swanlab] 初始化失败（将不记录曲线）: {e}")


def _swanlab_config_snapshot(cfg: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "batch_size",
        "micro_batch_clips",
        "encoder_use_half",
        "lambda_rank",
        "mil_pooling",
        "encoder_model_name",
    )
    snap: dict[str, Any] = {k: cfg.get(k) for k in keys}
    stages = cfg.get("stages")
    if isinstance(stages, list):
        snap["stages"] = [
            {
                "name": s.get("name"),
                "epochs": s.get("epochs"),
                "unfreeze_blocks": s.get("unfreeze_blocks"),
                "micro_batch_clips": s.get("micro_batch_clips"),
                "lr": s.get("lr"),
            }
            for s in stages
            if isinstance(s, dict)
        ]
    return snap


def _swanlab_log(data: dict[str, Any], step: Optional[int] = None) -> None:
    if not _swanlab_active:
        return
    try:
        import swanlab
    except ImportError:
        return
    clean: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, float) and not math.isfinite(v):
            continue
        if isinstance(v, (bool, int, float, str)):
            clean[k] = v
        elif v is None:
            continue
        else:
            try:
                clean[k] = float(v)
            except (TypeError, ValueError):
                continue
    if not clean:
        return
    try:
        swanlab.log(clean, step=step)
    except Exception:
        pass


def _fmt_eta(sec: float) -> str:
    if not np.isfinite(sec) or sec < 0:
        return "计算中"
    sec = int(round(sec))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}小时{m:02d}分"
    if m > 0:
        return f"{m}分{s:02d}秒"
    return f"{s}秒"


def _progress_print(line: str, *, same_line: bool) -> None:
    if same_line:
        pad = max(0, 140 - len(line))
        sys.stdout.write("\r" + line + " " * min(pad, 100))
    else:
        print(line, flush=True)
    sys.stdout.flush()


def _plot_confusion_matrix_png(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str],
    path: Path,
) -> None:
    if not _HAS_MPL or not y_true:
        return
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    k = len(label_names)
    if confusion_matrix is not None:
        cm = confusion_matrix(yt, yp, labels=list(range(k)))
    else:
        cm = np.zeros((k, k), dtype=np.int64)
        for t, p in zip(yt, yp):
            if 0 <= int(t) < k and 0 <= int(p) < k:
                cm[int(t), int(p)] += 1
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=label_names,
        yticklabels=label_names,
        ylabel="真实标签",
        xlabel="模型预测",
        title="验证集混淆矩阵（训练结束时的权重）",
    )
    vmax = max(float(cm.max()), 1.0)
    thresh = vmax / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(int(cm[i, j]), "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = dict(CONFIG)
    yaml_path = Path(str(cfg["yaml_config"]))
    _load_yaml_merge(cfg, yaml_path)
    _resolve_paths_under_project(cfg, PROJECT_ROOT)

    device = (cfg.get("device") or "").strip() or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(str(cfg["out_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    _swanlab_try_init(cfg)

    random.seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))

    exclude: set[str] = set()
    if cfg.get("exclude_unknown"):
        exclude.add("unknown")

    def _vf(row: Any) -> bool:
        if exclude and str(row.label).strip() in exclude:
            return False
        return True

    preclip = str(cfg.get("preclip_root") or "").strip()
    if not preclip:
        raise RuntimeError(
            "训练只使用已预切好的 clip（.npz）。请在 CONFIG 里设置 preclip_root 为包含 manifest.json 的目录，"
            "并先运行 lab_anomaly/tools/precompute_clips.py。不要把 preclip_root 留空。"
        )
    preclip_path = Path(preclip)
    if not (preclip_path / "manifest.json").is_file():
        raise RuntimeError(
            f"找不到预切片索引 {preclip_path / 'manifest.json'}。请先运行 lab_anomaly/tools/precompute_clips.py，"
            f"或把 preclip_root 改成正确的预切片输出目录。"
        )

    clip_ds = VideoClipDataset(
        dataset_root=cfg["dataset_root"],
        labels_csv=cfg["labels_csv"],
        frames_per_clip=int(cfg["frames_per_clip"]),
        interval_sec=float(cfg["interval_sec"]),
        max_clips_per_video=int(cfg["max_clips_per_video"]),
        shuffle_clips=False,
        video_filter=_vf,
        preclip_root=preclip,
        exclude_unknown_for_manifest=bool(cfg.get("exclude_unknown")),
        normal_label_for_manifest=str(cfg["normal_label"]),
    )

    all_vi = [i for i in range(len(clip_ds.rows)) if clip_ds.num_clips_for_video(i) > 0]
    if len(all_vi) < 2:
        raise RuntimeError("有效视频过少，请检查 labels_csv 与视频路径")

    train_rel, val_rel = split_train_val(len(all_vi), float(cfg["val_ratio"]), int(cfg["seed"]))
    train_vis = [all_vi[i] for i in train_rel]
    val_vis = [all_vi[i] for i in val_rel]

    train_ds = VideoMILBinaryDataset(clip_ds, train_vis, str(cfg["normal_label"]))
    val_ds = VideoMILBinaryDataset(clip_ds, val_vis, str(cfg["normal_label"]))
    bs = int(cfg["batch_size"])
    nw = int(cfg["num_workers"])
    pin = device.startswith("cuda")

    n_norm_tr = sum(
        1
        for vi in train_vis
        if str(clip_ds.rows[vi].label).strip().lower() == str(cfg["normal_label"]).strip().lower()
    )
    n_anom_tr = len(train_vis) - n_norm_tr
    use_balanced = bool(cfg.get("use_anomaly_branch", True)) and n_norm_tr > 0 and n_anom_tr > 0
    epoch_ref = [0]
    if use_balanced:
        sampler = BalancedVideoBatchSampler(
            train_vis,
            clip_ds,
            str(cfg["normal_label"]),
            bs,
            int(cfg["seed"]),
            epoch_ref,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=sampler,
            num_workers=nw,
            collate_fn=collate_video_batch,
            pin_memory=pin,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=bs,
            shuffle=True,
            num_workers=nw,
            collate_fn=collate_video_batch,
            pin_memory=pin,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        collate_fn=collate_video_batch,
        pin_memory=pin,
    )

    enc_cfg = VideoMAEv2EncoderConfig(
        model_name=str(cfg["encoder_model_name"]),
        num_frames=int(cfg["frames_per_clip"]),
        use_half=bool(cfg["encoder_use_half"]) and device.startswith("cuda"),
    )
    encoder = VideoMAEv2Encoder(enc_cfg, device=device)

    mil_cfg = MILHeadConfig(
        embedding_dim=encoder.embedding_dim,
        num_classes=2,
        pooling=str(cfg["mil_pooling"]),
        topk=int(cfg["mil_topk"]),
        attn_hidden=int(cfg["mil_attn_hidden"]),
        dropout=float(cfg["mil_dropout"]),
        use_layernorm=bool(cfg["mil_use_layernorm"]),
        use_anomaly_branch=bool(cfg["use_anomaly_branch"]),
        anomaly_hidden=int(cfg["anomaly_hidden"]),
    )
    mil = MILClassifier(mil_cfg).to(device)

    ce_weight: Optional[torch.Tensor] = None
    if cfg.get("class_weight"):
        n0 = sum(1 for vi in all_vi if str(clip_ds.rows[vi].label).strip().lower() == str(cfg["normal_label"]).strip().lower())
        n1 = len(all_vi) - n0
        w0 = len(all_vi) / (2 * max(n0, 1))
        w1 = len(all_vi) / (2 * max(n1, 1))
        ce_weight = torch.tensor([w0, w1], dtype=torch.float32, device=device)

    normal_idx = LABEL2IDX["normal"]
    use_rank = bool(cfg.get("use_anomaly_branch")) and normal_idx >= 0
    clip_eval_frozen = bool(cfg["clip_encode_eval_mode_when_frozen"])

    use_amp = bool(cfg["encoder_use_half"]) and device.startswith("cuda")
    scaler: Optional[torch.amp.GradScaler] = None
    if use_amp:
        scaler = torch.amp.GradScaler("cuda")

    history: list[dict[str, Any]] = []
    best_val = -1.0
    global_epoch = 0
    stop_training = False
    plan_total_epochs = sum(int(s["epochs"]) for s in cfg["stages"])
    last_full_epoch_sec: Optional[float] = None
    lambda_rank_f = float(cfg["lambda_rank"])
    prog_refresh = bool(cfg.get("progress_one_line_refresh", False))
    sw_train_step = 0

    for stage_index, stage in enumerate(cfg["stages"]):
        if stop_training:
            break
        name = str(stage["name"])
        ep_n = int(stage["epochs"])
        lr = float(stage["lr"])
        wd = float(stage["weight_decay"])
        unfreeze = int(stage["unfreeze_blocks"])
        micro_bs = max(1, int(stage.get("micro_batch_clips", cfg["micro_batch_clips"])))

        encoder.unfreeze_last_n_blocks(unfreeze)
        if unfreeze > 0:
            encoder.backbone.train()
        params = [p for p in encoder.parameters() if p.requires_grad] + list(mil.parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        ce = nn.CrossEntropyLoss(weight=ce_weight)
        backbone_frozen = unfreeze == 0

        _p(
            f"\n=== stage={name} epochs={ep_n} lr={lr} unfreeze_blocks={unfreeze} "
            f"micro_batch_clips={micro_bs} "
            f"trainable={encoder.trainable_param_count() + sum(p.numel() for p in mil.parameters() if p.requires_grad)} ==="
        )

        for ep_in_stage in range(ep_n):
            if use_balanced:
                epoch_ref[0] = global_epoch
            mil.train()
            if not backbone_frozen:
                encoder.backbone.train()
            elif not clip_eval_frozen:
                encoder.backbone.train()

            loss_ce_sum = 0.0
            loss_r_sum = 0.0
            n_steps = 0
            correct = 0
            total = 0

            t_epoch_wall0 = time.perf_counter()
            n_batches = len(train_loader)
            lr_now = float(opt.param_groups[0]["lr"])

            for batch_idx, batch in enumerate(train_loader):
                y = batch["y"].to(device)
                lce, lrk, logits = _training_step(
                    encoder,
                    mil,
                    batch,
                    device,
                    opt,
                    scaler,
                    ce,
                    backbone_frozen,
                    micro_bs,
                    clip_eval_frozen,
                    use_rank,
                    normal_idx,
                    float(cfg["lambda_rank"]),
                    float(cfg["lambda_sparse"]),
                    float(cfg["lambda_smooth"]),
                )
                pred = logits.argmax(dim=-1)
                total += int(y.numel())
                correct += int((pred == y).sum().item())
                loss_ce_sum += lce
                loss_r_sum += lrk
                n_steps += 1
                ltot = lce + lambda_rank_f * lrk
                lr_now = float(opt.param_groups[0]["lr"])

                sw_train_step += 1
                sw_every = int(cfg.get("swanlab_log_every_n_batches", 1))
                if _swanlab_active and sw_every > 0 and (sw_train_step % sw_every == 0):
                    _swanlab_log(
                        {
                            "train_batch/loss_total": ltot,
                            "train_batch/loss_ce": lce,
                            "train_batch/loss_rank": lrk,
                            "train_batch/lr": lr_now,
                            "train_batch/clips": float(int(batch["mask"].sum().item())),
                            "train_batch/micro_batch_clips": float(micro_bs),
                            "train_batch/stage_unfreeze_blocks": float(unfreeze),
                            "train_batch/global_epoch": float(global_epoch),
                            "train_batch/stage_index": float(stage_index),
                        },
                        step=sw_train_step,
                    )

                done = batch_idx + 1
                elapsed = time.perf_counter() - t_epoch_wall0
                pred_full_epoch = elapsed / max(1, done) * max(1, n_batches)
                eta_epoch = max(0.0, pred_full_epoch - elapsed)
                avg_ep = last_full_epoch_sec if last_full_epoch_sec is not None else pred_full_epoch
                rem_full_epochs = max(0, plan_total_epochs - global_epoch - 1)
                eta_all = eta_epoch + rem_full_epochs * avg_ep

                line = (
                    f"[batch {done}/{n_batches}] loss_total={ltot:.4f} loss_ce={lce:.4f} loss_rank={lrk:.4f} | "
                    f"lr={lr_now:.2e} | clips={int(batch['mask'].sum().item())} | "
                    f"阶段={name} | 总第{global_epoch + 1}/{plan_total_epochs}轮 | 本阶段{ep_in_stage + 1}/{ep_n}轮 | "
                    f"本轮还剩{_fmt_eta(eta_epoch)} | 全部还剩{_fmt_eta(eta_all)}"
                )
                _progress_print(line, same_line=prog_refresh)

            if prog_refresh:
                sys.stdout.write("\n")
                sys.stdout.flush()

            last_full_epoch_sec = time.perf_counter() - t_epoch_wall0

            metrics = evaluate(
                encoder,
                mil,
                val_loader,
                device,
                ce_weight,
                backbone_frozen,
                micro_bs,
                clip_eval_frozen,
                use_amp=use_amp,
            )
            tr_acc = correct / max(1, total)
            avg_ce = loss_ce_sum / max(1, n_steps)
            avg_rk = loss_r_sum / max(1, n_steps)
            rec = {
                "epoch": global_epoch,
                "stage": name,
                "train_acc": tr_acc,
                "train_loss_ce": avg_ce,
                "train_loss_rank": avg_rk,
                "train_loss_total": avg_ce + lambda_rank_f * avg_rk,
                **metrics,
            }
            history.append(rec)
            auc = rec.get("val_auc_binary", float("nan"))
            lr_ep = float(opt.param_groups[0]["lr"])
            _p(
                f"*** epoch {global_epoch + 1}/{plan_total_epochs} [{name}] lr={lr_ep:.2e} | "
                f"train_ce={avg_ce:.4f} train_rk={avg_rk:.4f} train_tot={avg_ce + lambda_rank_f * avg_rk:.4f} | "
                f"train_acc={tr_acc:.4f} val_loss={metrics['val_loss']:.4f} val_acc={metrics['val_acc']:.4f} auc={auc}"
            )

            if _swanlab_active:
                ep_log: dict[str, Any] = {
                    "epoch/train_loss_ce": avg_ce,
                    "epoch/train_loss_rank": avg_rk,
                    "epoch/train_loss_total": avg_ce + lambda_rank_f * avg_rk,
                    "epoch/train_acc": tr_acc,
                    "epoch/val_loss": float(metrics["val_loss"]),
                    "epoch/val_acc": float(metrics["val_acc"]),
                    "epoch/lr": lr_ep,
                    "epoch/micro_batch_clips": float(micro_bs),
                    "epoch/stage_index": float(stage_index),
                }
                try:
                    auc_f = float(auc)
                    if math.isfinite(auc_f):
                        ep_log["epoch/val_auc"] = auc_f
                except (TypeError, ValueError):
                    pass
                _swanlab_log(ep_log, step=global_epoch)

            if metrics["val_acc"] > best_val:
                best_val = float(metrics["val_acc"])
                ckpt = {
                    "encoder_cfg": asdict(enc_cfg),
                    "encoder_state": encoder.state_dict(),
                    "mil_state": mil.state_dict(),
                    "mil_cfg": asdict(mil_cfg),
                    "label2idx": LABEL2IDX,
                    "idx2label": {str(k): v for k, v in IDX2LABEL.items()},
                    "best_val_acc": best_val,
                    "stage": name,
                    "epoch": global_epoch,
                }
                torch.save(ckpt, out_dir / "checkpoint_best.pt")
                _p(f"  saved best: {best_val:.4f}")

            torch.save(
                {
                    "encoder_cfg": asdict(enc_cfg),
                    "encoder_state": encoder.state_dict(),
                    "mil_state": mil.state_dict(),
                    "mil_cfg": asdict(mil_cfg),
                    "label2idx": LABEL2IDX,
                    "idx2label": {str(k): v for k, v in IDX2LABEL.items()},
                    "epoch": global_epoch,
                    "stage": name,
                },
                out_dir / "checkpoint_last.pt",
            )

            if float(cfg.get("target_acc", 0)) > 0 and metrics["val_acc"] >= float(cfg["target_acc"]):
                _p("early stop: target_acc reached")
                stop_training = True
                break
            global_epoch += 1

    (out_dir / "labels.json").write_text(
        json.dumps({"label2idx": LABEL2IDX, "idx2label": IDX2LABEL}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if history and _HAS_MPL:
        ep = list(range(len(history)))
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        ax0, ax1 = axes
        ax0.plot(ep, [h["train_loss_ce"] for h in history], label="训练 CE", marker="o", markersize=2)
        ax0.plot(ep, [h["val_loss"] for h in history], label="验证 loss", marker="s", markersize=2)
        ax0.plot(ep, [h["train_loss_total"] for h in history], label="训练 total(ce+λ·rank)", marker="^", markersize=2)
        ax0.set_ylabel("loss")
        ax0.legend(loc="best", fontsize=8)
        ax0.grid(True, alpha=0.3)
        ax0.set_title("损失曲线")
        ax1.plot(ep, [h["train_loss_rank"] for h in history], label="训练 rank", color="tab:purple", marker="d", markersize=2)
        ax1.set_xlabel("epoch（从0开始）")
        ax1.set_ylabel("rank loss")
        ax1.legend(loc="best", fontsize=8)
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "plot_loss_curves.png", dpi=150)
        plt.close(fig)

        plt.figure(figsize=(8, 4))
        plt.plot(ep, [h["train_acc"] for h in history], label="train_acc")
        plt.plot(ep, [h["val_acc"] for h in history], label="val_acc")
        plt.xlabel("epoch")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(out_dir / "plot_acc.png", dpi=150)
        plt.close()

    if cfg.get("save_val_confusion_matrix", True):
        final_ev = evaluate(
            encoder,
            mil,
            val_loader,
            device,
            ce_weight,
            backbone_frozen,
            micro_bs,
            clip_eval_frozen,
            return_predictions=True,
            use_amp=use_amp,
        )
        y_t = final_ev.get("y_true") or []
        y_p = final_ev.get("y_pred") or []
        if y_t and y_p and _HAS_MPL:
            names = [IDX2LABEL[i] for i in sorted(IDX2LABEL.keys())]
            _plot_confusion_matrix_png(y_t, y_p, names, out_dir / "confusion_matrix_val.png")
            _p(f"[图] 验证集混淆矩阵已保存: {out_dir / 'confusion_matrix_val.png'}")

    _p(f"[OK] done. out_dir={out_dir}")
    if history and _HAS_MPL:
        _p(f"[图] 损失曲线: {out_dir / 'plot_loss_curves.png'}  准确率: {out_dir / 'plot_acc.png'}")


if __name__ == "__main__":
    main()
