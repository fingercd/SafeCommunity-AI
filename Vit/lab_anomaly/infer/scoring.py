"""
推理打分：加载端到端 checkpoint（VideoMAE v2 + MIL），二分类概率与 ranking 异常分数融合。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from lab_anomaly.models.mil_head import MILClassifier, MILHeadConfig
from lab_anomaly.models.vit_video_encoder import VideoMAEv2Encoder, VideoMAEv2EncoderConfig


@dataclass(frozen=True)
class KnownClassifierBundle:
    """端到端或单独 MIL：模型、标签映射、设备"""
    model: Any
    label2idx: dict[str, int]
    idx2label: dict[int, str]
    device: str
    encoder: Optional[Any] = None


def load_known_classifier(checkpoint_path: str | Path, device: Optional[str] = None) -> KnownClassifierBundle:
    """
    从 checkpoint 加载 MIL 头（可与 encoder_state 同文件）。
    若 ckpt 含 encoder_cfg / encoder_state，同时载入 VideoMAEv2Encoder 到 bundle.encoder。
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    enc: Optional[VideoMAEv2Encoder] = None
    if "encoder_cfg" in ckpt and "encoder_state" in ckpt:
        enc_cfg = VideoMAEv2EncoderConfig(**ckpt["encoder_cfg"])
        enc = VideoMAEv2Encoder(enc_cfg, device=device)
        enc.load_state_dict(ckpt["encoder_state"], strict=True)
        enc.eval()

    cfg_d = ckpt.get("cfg") or ckpt.get("mil_cfg")
    if cfg_d is None:
        raise RuntimeError("checkpoint 缺少 mil_cfg 或 cfg")
    cfg = MILHeadConfig(**cfg_d)
    model = MILClassifier(cfg)
    state_key = "mil_state" if "mil_state" in ckpt else "model_state"
    model.load_state_dict(ckpt[state_key], strict=True)
    model.to(device)
    model.eval()

    label2idx = {str(k): int(v) for k, v in ckpt.get("label2idx", {}).items()}
    idx2label_raw = ckpt.get("idx2label", {})
    idx2label = {int(k): str(v) for k, v in idx2label_raw.items()}

    return KnownClassifierBundle(model=model, label2idx=label2idx, idx2label=idx2label, device=device, encoder=enc)


@dataclass(frozen=True)
class FusionOutput:
    """融合输出：预测标签、置信度、是否异常、ranking 异常分数"""
    predicted_label: str
    predicted_prob: float
    is_anomaly: bool
    ranking_anomaly_score: float = 0.0


def predict_fusion(
    known_probs: np.ndarray,
    idx2label: dict[int, str],
    ranking_anomaly_score: float = 0.0,
    ranking_alarm_threshold: float = 0.5,
    min_known_prob: float = 0.5,
    treat_low_conf_as_unknown: bool = True,
    normal_idx: int = 0,
) -> FusionOutput:
    """
    二分类：已知 softmax 概率 + MIL ranking 分支分数。
    is_anomaly：非 normal 且置信度够高，或 ranking 分数超阈值。
    """
    probs = np.asarray(known_probs, dtype=np.float32).reshape(-1)
    pred_i = int(probs.argmax())
    pred_label = str(idx2label.get(pred_i, str(pred_i)))
    pred_prob = float(probs[pred_i])

    if treat_low_conf_as_unknown and pred_prob < float(min_known_prob):
        pred_label = "unknown"

    rank_trig = float(ranking_anomaly_score) >= float(ranking_alarm_threshold)
    abnormal = pred_i != normal_idx and pred_prob >= float(min_known_prob)
    is_anomaly = bool(abnormal or rank_trig)

    return FusionOutput(
        predicted_label=pred_label,
        predicted_prob=pred_prob,
        is_anomaly=is_anomaly,
        ranking_anomaly_score=float(ranking_anomaly_score),
    )
