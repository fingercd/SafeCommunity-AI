"""
推理阶段打分模块：
  - OpenSetScorer：加载 KMeans+OCSVM，对 embedding 计算 cluster、anomaly_score、是否异常
  - load_known_classifier：加载 MIL checkpoint
  - fuse_known_and_open_set：已知分类概率 + 开放集 anomaly 的简单融合
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


def require_joblib():
    """检查 joblib 是否可用"""
    try:
        import joblib  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("缺少 joblib 依赖。请安装：pip install joblib") from e


@dataclass(frozen=True)
class OpenSetScore:
    """开放集单次打分结果：簇 ID、decision 值、anomaly_score、阈值、是否异常"""
    cluster_id: int
    decision_score: float
    anomaly_score: float
    threshold: float
    is_anomaly: bool


class OpenSetScorer:
    """
    读取 fit_kmeans_ocsvm.py 产物，提供：
      - cluster
      - decision_function / anomaly_score
      - threshold 判定
    """

    def __init__(self, artifacts_dir: str | Path):
        require_joblib()
        import joblib

        self.artifacts_dir = Path(artifacts_dir)
        thr_path = self.artifacts_dir / "thresholds.json"
        if not thr_path.exists():
            raise FileNotFoundError(f"cannot find thresholds.json: {thr_path}")
        self.thresholds = json.loads(thr_path.read_text(encoding="utf-8"))

        self.kmeans = joblib.load(self.artifacts_dir / "kmeans.joblib")
        self.ocsvm_global = joblib.load(self.artifacts_dir / "ocsvm_global.joblib")

        # 按需加载 per-cluster 模型
        self._ocsvm_cache: dict[int, Any] = {}
        for k in self.thresholds.get("cluster_thresholds", {}).keys():
            ci = int(k)
            p = self.artifacts_dir / f"ocsvm_cluster_{ci:03d}.joblib"
            if p.exists():
                self._ocsvm_cache[ci] = joblib.load(p)

        self.global_threshold = float(self.thresholds.get("global_threshold", 0.0))
        self.cluster_thresholds = {int(k): float(v) for k, v in self.thresholds.get("cluster_thresholds", {}).items()}

    def _get_model_and_threshold(self, cluster_id: int) -> tuple[Any, float]:
        """获取该簇对应的 OCSVM 模型及阈值，无则用全局"""
        m = self._ocsvm_cache.get(cluster_id, None)
        if m is None:
            return self.ocsvm_global, self.global_threshold
        thr = self.cluster_thresholds.get(cluster_id, self.global_threshold)
        return m, thr

    def score_embedding(self, emb: np.ndarray) -> OpenSetScore:
        """对单个 embedding 计算簇 ID、decision_score、anomaly_score、是否异常"""
        emb = np.asarray(emb, dtype=np.float32).reshape(1, -1)
        cluster_id = int(self.kmeans.predict(emb)[0])
        m, thr = self._get_model_and_threshold(cluster_id)
        decision = float(m.decision_function(emb)[0])
        anomaly = -decision
        return OpenSetScore(
            cluster_id=cluster_id,
            decision_score=decision,
            anomaly_score=anomaly,
            threshold=float(thr),
            is_anomaly=bool(anomaly > float(thr)),
        )


@dataclass(frozen=True)
class KnownClassifierBundle:
    """已知分类器加载结果：模型、label 映射、设备"""
    model: Any
    label2idx: dict[str, int]
    idx2label: dict[int, str]
    device: str


def load_known_classifier(checkpoint_path: str | Path, device: Optional[str] = None) -> KnownClassifierBundle:
    """加载 MIL 分类器 checkpoint，返回模型、label 映射、设备"""
    import torch

    from lab_anomaly.models.mil_head import MILClassifier, MILHeadConfig

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    cfg_d = ckpt.get("cfg", None)
    if cfg_d is None:
        raise RuntimeError("checkpoint missing cfg")
    cfg = MILHeadConfig(**cfg_d)
    model = MILClassifier(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    label2idx = {str(k): int(v) for k, v in ckpt.get("label2idx", {}).items()}
    idx2label = {int(k): str(v) for k, v in ckpt.get("idx2label", {}).items()}
    return KnownClassifierBundle(model=model, label2idx=label2idx, idx2label=idx2label, device=device)


@dataclass(frozen=True)
class FusionOutput:
    """融合输出：预测标签、置信度、anomaly_score、是否异常、开放集信息"""
    predicted_label: str
    predicted_prob: float
    anomaly_score: float
    is_anomaly: bool
    open_set_cluster: int
    open_set_threshold: float


def fuse_known_and_open_set(
    known_probs: np.ndarray,
    idx2label: dict[int, str],
    open_set: OpenSetScore,
    min_known_prob: float = 0.5,
    treat_low_conf_as_unknown: bool = True,
) -> FusionOutput:
    """
    已知分类 + 开放集融合：取已知分类 argmax 作为预测；若 treat_low_conf_as_unknown 且
    置信度 < min_known_prob，则置为 unknown；anomaly_score 与 is_anomaly 来自开放集。
    """
    probs = np.asarray(known_probs, dtype=np.float32).reshape(-1)
    pred_i = int(probs.argmax())
    pred_label = str(idx2label.get(pred_i, str(pred_i)))
    pred_prob = float(probs[pred_i])

    if treat_low_conf_as_unknown and pred_prob < float(min_known_prob):
        pred_label = "unknown"

    return FusionOutput(
        predicted_label=pred_label,
        predicted_prob=pred_prob,
        anomaly_score=float(open_set.anomaly_score),
        is_anomaly=bool(open_set.is_anomaly),
        open_set_cluster=int(open_set.cluster_id),
        open_set_threshold=float(open_set.threshold),
    )

