"""
评估开放集（KMeans + OCSVM）有效性：
  - 正常样本的簇质量（轮廓系数、簇分布）
  - 边界是否合理（正常样本被误判为异常的比例应约等于 1 - quantile）
  - 若有异常标签，看异常样本的 anomaly score 是否明显高于正常、且多数被判为异常
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


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


def _resolve_emb_path(embeddings_dir: Path, p: str) -> Path:
    """解析 embedding 路径：优先绝对路径，否则相对于 embeddings_dir"""
    P = Path(p)
    if P.exists():
        return P
    cand = embeddings_dir / p
    if cand.exists():
        return cand
    return P


def load_embeddings_from_rows(
    rows: list[dict[str, Any]],
    embeddings_dir: Path,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """从 meta 记录加载 embedding，支持 npy_per_clip 与 npz_per_video"""
    xs: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    for r in rows:
        fmt = str(r.get("save_format", "npy_per_clip"))
        p = str(r["embedding_path"])
        full_path = _resolve_emb_path(embeddings_dir, p)
        if fmt == "npz_per_video" or p.lower().endswith(".npz"):
            npz = np.load(full_path, allow_pickle=True)
            E = np.asarray(npz["embeddings"], dtype=np.float32)
            clip_idx = None
            if "clip_index" in npz.files:
                clip_idx = np.asarray(npz["clip_index"]).astype(np.int32)
            for i in range(E.shape[0]):
                xs.append(E[i])
                metas.append({"video_id": r.get("video_id", ""), "clip_index": int(clip_idx[i]) if clip_idx is not None else int(i)})
        else:
            x = np.load(full_path).astype(np.float32)
            xs.append(x)
            metas.append({"video_id": r.get("video_id", ""), "clip_index": int(r.get("clip_index", 0))})
    return np.stack(xs, axis=0), metas


def main() -> None:
    ap = argparse.ArgumentParser(description="评估开放集 KMeans+OCSVM：簇质量与边界有效性")
    ap.add_argument("--embeddings_dir", type=str, default="lab_dataset/derived/embeddings")
    ap.add_argument("--open_set_dir", type=str, default="lab_dataset/derived/open_set")
    ap.add_argument("--normal_label", type=str, default="normal")
    ap.add_argument("--max_normal", type=int, default=0, help="仅用前 N 个 normal 做评估（0=全部，用于大集时加速）")
    ap.add_argument("--skip_silhouette", action="store_true", help="跳过轮廓系数（大样本时较慢）")
    args = ap.parse_args()

    _project_root = Path(__file__).resolve().parent.parent.parent
    if not Path(args.embeddings_dir).is_absolute():
        args.embeddings_dir = str(_project_root / args.embeddings_dir)
    if not Path(args.open_set_dir).is_absolute():
        args.open_set_dir = str(_project_root / args.open_set_dir)

    try:
        import joblib
    except Exception as e:
        raise RuntimeError("缺少 joblib。请安装：pip install joblib") from e

    embeddings_dir = Path(args.embeddings_dir)
    open_set_dir = Path(args.open_set_dir)
    meta_path = embeddings_dir / "embeddings_meta.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"cannot find {meta_path}")

    thr_path = open_set_dir / "thresholds.json"
    if not thr_path.exists():
        raise FileNotFoundError(f"cannot find {thr_path}")

    thresholds = json.loads(thr_path.read_text(encoding="utf-8"))
    kmeans = joblib.load(open_set_dir / "kmeans.joblib")
    ocsvm_global = joblib.load(open_set_dir / "ocsvm_global.joblib")
    ocsvm_by_cluster: dict[int, Any] = {}
    for k in thresholds.get("cluster_thresholds", {}).keys():
        ci = int(k)
        p = open_set_dir / f"ocsvm_cluster_{ci:03d}.joblib"
        if p.exists():
            ocsvm_by_cluster[ci] = joblib.load(p)

    global_thr = float(thresholds.get("global_threshold", 0.0))
    cluster_thresholds = {int(k): float(v) for k, v in thresholds.get("cluster_thresholds", {}).items()}
    quantile = float(thresholds.get("quantile", 0.95))

    rows = read_jsonl(meta_path)
    normal_rows = [r for r in rows if str(r.get("label", "")) == str(args.normal_label)]
    if args.max_normal and len(normal_rows) > args.max_normal:
        normal_rows = normal_rows[: args.max_normal]
    if not normal_rows:
        raise RuntimeError(f"no normal rows with label={args.normal_label!r}")

    X_normal, _ = load_embeddings_from_rows(normal_rows, embeddings_dir)
    clusters = kmeans.predict(X_normal)

    # 每个样本的 anomaly_score 与使用的阈值
    anomaly_scores = np.zeros((X_normal.shape[0],), dtype=np.float32)
    used_threshold = np.zeros((X_normal.shape[0],), dtype=np.float32)
    for i in range(X_normal.shape[0]):
        ci = int(clusters[i])
        m = ocsvm_by_cluster.get(ci, ocsvm_global)
        thr = cluster_thresholds.get(ci, global_thr)
        decision = float(m.decision_function(X_normal[i : i + 1])[0])
        anomaly_scores[i] = -decision
        used_threshold[i] = thr

    is_anomaly = anomaly_scores > used_threshold
    rate_flagged = float(np.mean(is_anomaly))

    # 按簇统计
    k = int(thresholds.get("k", 0))
    cluster_sizes = {ci: int(np.sum(clusters == ci)) for ci in range(k)}
    cluster_flagged: dict[int, float] = {}
    for ci in range(k):
        idx = np.where(clusters == ci)[0]
        if idx.size == 0:
            continue
        cluster_flagged[ci] = float(np.mean(is_anomaly[idx]))

    # 轮廓系数（可选）
    silhouette = None
    if not args.skip_silhouette and X_normal.shape[0] >= 2 and k >= 2:
        try:
            from sklearn.metrics import silhouette_score
            silhouette = float(silhouette_score(X_normal, clusters))
        except Exception as e:
            silhouette = None  # e.g. one cluster empty

    # ----- 报告 -----
    print("=" * 70)
    print("开放集模型有效性评估")
    print("=" * 70)
    print(f"\n[数据] normal clips: {X_normal.shape[0]} | embedding dim: {X_normal.shape[1]}")
    print(f"[阈值] quantile={quantile:.2f} | global_threshold={global_thr:.6f}")
    print()

    print("[1] 簇分布（正常样本在各簇的样本数）")
    for ci in range(k):
        n = cluster_sizes.get(ci, 0)
        bar = "#" * max(1, n * 50 // max(1, X_normal.shape[0]))
        print(f"    簇 {ci:2d}: {n:5d}  {bar}")
    print()

    if silhouette is not None:
        print(f"[2] 簇质量 - 轮廓系数 (Silhouette): {silhouette:.4f}")
        print("    范围 [-1,1]，越接近 1 表示簇内紧、簇间分离越好；<0.3 可考虑调 k 或检查数据。")
        print()
    else:
        print("[2] 簇质量 - 轮廓系数: 未计算 (--skip_silhouette 或样本/簇数不足)")
        print()

    print("[3] 边界有效性（在「正常」样本上的表现）")
    print(f"    理论：约 {(1 - quantile) * 100:.0f}% 的正常样本会落在阈值之上（因阈值取 {quantile*100:.0f} 分位数）。")
    print(f"    实际：{rate_flagged * 100:.2f}% 的正常样本被标为异常 (anomaly_score > threshold)。")
    if abs(rate_flagged - (1 - quantile)) > 0.1:
        print("    提示：若实际比例与理论差很多，可检查 quantile 或簇/OCSVM 是否过紧/过松。")
    print()
    print("    各簇被标为异常的比例:")
    for ci in range(k):
        if ci in cluster_flagged:
            print(f"      簇 {ci:2d}: {cluster_flagged[ci] * 100:.1f}%")
    print()

    # 若有非 normal 标签，简单看异常分数
    other_labels = sorted({str(r.get("label", "")) for r in rows if str(r.get("label", "")) != str(args.normal_label)})
    if other_labels:
        print("[4] 其他标签（期望：anomaly score 更高、多数被判为异常）")
        for label in other_labels:
            other_rows = [r for r in rows if str(r.get("label", "")) == label]
            X_other, _ = load_embeddings_from_rows(other_rows, embeddings_dir)
            cl_other = kmeans.predict(X_other)
            scores_other = np.zeros((X_other.shape[0],), dtype=np.float32)
            for i in range(X_other.shape[0]):
                ci = int(cl_other[i])
                m = ocsvm_by_cluster.get(ci, ocsvm_global)
                scores_other[i] = -float(m.decision_function(X_other[i : i + 1])[0])
            thr_other = np.array([cluster_thresholds.get(int(cl_other[i]), global_thr) for i in range(X_other.shape[0])])
            rate_anom = float(np.mean(scores_other > thr_other))
            print(f"    {label!r}: clips={X_other.shape[0]} | 均值 anomaly_score={np.mean(scores_other):.4f} | 被判异常比例: {rate_anom*100:.1f}%")
        print()
    else:
        print("[4] 未发现其他标签；若有异常样本，可在 CSV 中标为异常类后重新提取 embedding 再跑本脚本查看其 score。")
        print()

    print("=" * 70)


if __name__ == "__main__":
    main()
