"""
开放集异常检测：在 normal clip embeddings 上训练 KMeans + One-Class SVM

流程：
  1. 从 embeddings_meta.jsonl 加载 label=normal 的 clip embedding
  2. KMeans 聚类，将正常样本分为 k 个簇（捕捉不同“正常模式”）
  3. 全局 OCSVM + 每簇 OCSVM（小簇回退到全局）
  4. 在 normal 上统计 anomaly_score 的 quantile 作为阈值
  5. 保存 kmeans.joblib、ocsvm_*.joblib、thresholds.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """逐行解析 JSONL 文件，返回记录列表"""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def require_sklearn():
    """检查 scikit-learn 是否可用，缺则抛出运行时错误"""
    try:
        import sklearn  # noqa: F401
        from sklearn.cluster import KMeans  # noqa: F401
        from sklearn.svm import OneClassSVM  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "缺少 scikit-learn 依赖。请安装：pip install scikit-learn joblib"
        ) from e


def main() -> None:
    ap = argparse.ArgumentParser(description="开放集：在 normal clip embeddings 上训练 KMeans + One-Class SVM")
    ap.add_argument("--embeddings_dir", type=str, default="lab_dataset/derived/embeddings")
    ap.add_argument("--out_dir", type=str, default="lab_dataset/derived/open_set")
    ap.add_argument("--normal_label", type=str, default="normal")
    ap.add_argument("--k", type=int, default=16, help="KMeans 聚类数")
    ap.add_argument("--min_cluster_size", type=int, default=20, help="小于该样本数的簇不单独训练 OCSVM（回退到 global）")
    ap.add_argument("--nu", type=float, default=0.05)
    ap.add_argument("--gamma", type=str, default="scale", help="OneClassSVM gamma（scale/auto/或具体数值）")
    ap.add_argument("--quantile", type=float, default=0.95, help="阈值分位数（在 normal 上统计）")
    ap.add_argument("--limit", type=int, default=0, help="仅使用前 N 条 normal clip（0=不限制）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 相对路径按项目根解析，PyCharm 任意工作目录下都能找到
    _project_root = Path(__file__).resolve().parent.parent.parent
    if not Path(args.embeddings_dir).is_absolute():
        args.embeddings_dir = str(_project_root / args.embeddings_dir)
    if not Path(args.out_dir).is_absolute():
        args.out_dir = str(_project_root / args.out_dir)

    require_sklearn()
    from sklearn.cluster import KMeans
    from sklearn.svm import OneClassSVM

    try:
        import joblib
    except Exception as e:  # pragma: no cover
        raise RuntimeError("缺少 joblib 依赖。请安装：pip install joblib") from e

    embeddings_dir = Path(args.embeddings_dir)
    meta_path = embeddings_dir / "embeddings_meta.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"cannot find {meta_path}")

    rows = read_jsonl(meta_path)
    normal_rows = [r for r in rows if str(r.get("label", "")) == str(args.normal_label)]
    if args.limit and int(args.limit) > 0:
        normal_rows = normal_rows[: int(args.limit)]
    if not normal_rows:
        raise RuntimeError(f"no normal rows found with label={args.normal_label!r}")

    # load embeddings
    # 支持两种格式：
    # - npy_per_clip：每条记录一个 clip embedding（npy）
    # - npz_per_video：每条记录一个视频（npz: embeddings=(N,D)）
    xs: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    for r in normal_rows:
        fmt = str(r.get("save_format", "npy_per_clip"))
        p = str(r["embedding_path"])
        if fmt == "npz_per_video" or p.lower().endswith(".npz"):
            npz = np.load(p, allow_pickle=True)
            E = np.asarray(npz["embeddings"], dtype=np.float32)
            clip_idx = None
            if "clip_index" in npz.files:
                clip_idx = np.asarray(npz["clip_index"]).astype(np.int32)
            for i in range(E.shape[0]):
                xs.append(E[i])
                metas.append(
                    {
                        "video_id": r.get("video_id", ""),
                        "clip_index": int(clip_idx[i]) if clip_idx is not None else int(i),
                        "path": p,
                        "save_format": "npz_per_video",
                    }
                )
        else:
            x = np.load(p).astype(np.float32)
            xs.append(x)
            metas.append(
                {
                    "video_id": r.get("video_id", ""),
                    "clip_index": int(r.get("clip_index", 0)),
                    "path": p,
                    "save_format": "npy_per_clip",
                }
            )
    X = np.stack(xs, axis=0)  # (N,D)

    # ── 打印数据概览 ──
    print("=" * 70)
    print("[Data] normal clips loaded: %d | embedding dim: %d" % (X.shape[0], X.shape[1]))
    print("[Config] k=%d | nu=%.4f | gamma=%s | quantile=%.2f | min_cluster_size=%d" % (
        args.k, args.nu, args.gamma, args.quantile, args.min_cluster_size,
    ))
    print("=" * 70)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── KMeans 聚类 ──
    print("\n[Step 1/4] KMeans clustering (k=%d) ..." % args.k)
    try:
        kmeans = KMeans(n_clusters=int(args.k), random_state=int(args.seed), n_init="auto", verbose=0)
    except TypeError:  # pragma: no cover
        kmeans = KMeans(n_clusters=int(args.k), random_state=int(args.seed), n_init=10, verbose=0)

    kmeans.fit(X)
    cluster = kmeans.predict(X)
    print("  KMeans inertia: %.4f" % kmeans.inertia_)
    print("  Cluster distribution:")
    cluster_sizes: dict[int, int] = {}
    for ci in range(int(args.k)):
        cnt = int(np.sum(cluster == ci))
        cluster_sizes[ci] = cnt
        bar = "#" * max(1, cnt * 40 // max(1, X.shape[0]))
        print("    cluster %2d: %5d samples  %s" % (ci, cnt, bar))

    # ── global OCSVM ──
    print("\n[Step 2/4] Training global One-Class SVM (nu=%.4f, gamma=%s) on %d samples ..." % (
        args.nu, args.gamma, X.shape[0],
    ))
    ocsvm_global = OneClassSVM(kernel="rbf", nu=float(args.nu), gamma=args.gamma)
    ocsvm_global.fit(X)
    n_sv_global = ocsvm_global.support_vectors_.shape[0]
    print("  Global OCSVM done. support vectors: %d / %d (%.1f%%)" % (
        n_sv_global, X.shape[0], 100.0 * n_sv_global / max(1, X.shape[0]),
    ))

    # ── per-cluster OCSVM ──
    print("\n[Step 3/4] Training per-cluster OCSVM ...")
    ocsvm_by_cluster: dict[int, Any] = {}
    for ci in range(int(args.k)):
        idx = np.where(cluster == ci)[0]
        if idx.size < int(args.min_cluster_size):
            print("  cluster %2d: %4d samples (< min_cluster_size=%d) -> skip, use global" % (
                ci, idx.size, args.min_cluster_size,
            ))
            continue
        m = OneClassSVM(kernel="rbf", nu=float(args.nu), gamma=args.gamma)
        m.fit(X[idx])
        n_sv = m.support_vectors_.shape[0]
        print("  cluster %2d: %4d samples -> OCSVM trained, support vectors: %d (%.1f%%)" % (
            ci, idx.size, n_sv, 100.0 * n_sv / max(1, idx.size),
        ))
        ocsvm_by_cluster[ci] = m
    print("  Per-cluster OCSVM: %d / %d clusters trained" % (len(ocsvm_by_cluster), args.k))

    # ── 阈值统计 ──
    print("\n[Step 4/4] Computing anomaly score thresholds on normal data ...")
    anomaly_scores = np.zeros((X.shape[0],), dtype=np.float32)
    used_model = np.zeros((X.shape[0],), dtype=np.int32) - 1
    for i in range(X.shape[0]):
        ci = int(cluster[i])
        m = ocsvm_by_cluster.get(ci, None)
        if m is None:
            m = ocsvm_global
            used_model[i] = -1
        else:
            used_model[i] = ci
        s = float(m.decision_function(X[i : i + 1])[0])
        # OCSVM decision_function：正值表示在正常边界内；取负号作为 anomaly_score（越大越异常）
    anomaly_scores[i] = -s

    q = float(args.quantile)
    global_thr = float(np.quantile(anomaly_scores, q))
    thr_by_cluster: dict[int, float] = {}
    for ci in range(int(args.k)):
        idx = np.where(used_model == ci)[0]
        if idx.size == 0:
            continue
        thr_by_cluster[ci] = float(np.quantile(anomaly_scores[idx], q))

    # ── 打印 anomaly score 统计 ──
    print("\n" + "=" * 70)
    print("[Anomaly Score Statistics on Normal Data]")
    print("  min:    %.6f" % float(np.min(anomaly_scores)))
    print("  max:    %.6f" % float(np.max(anomaly_scores)))
    print("  mean:   %.6f" % float(np.mean(anomaly_scores)))
    print("  std:    %.6f" % float(np.std(anomaly_scores)))
    print("  median: %.6f" % float(np.median(anomaly_scores)))
    print("  q%.0f:   %.6f  (= global threshold)" % (q * 100, global_thr))
    print("")
    print("[Per-Cluster Thresholds (q=%.2f)]" % q)
    for ci in sorted(thr_by_cluster.keys()):
        print("  cluster %2d: threshold=%.6f  (size=%d)" % (ci, thr_by_cluster[ci], cluster_sizes.get(ci, 0)))
    print("  global:     threshold=%.6f  (all %d samples)" % (global_thr, X.shape[0]))
    print("=" * 70)

    # save artifacts
    joblib.dump(kmeans, out_dir / "kmeans.joblib")
    joblib.dump(ocsvm_global, out_dir / "ocsvm_global.joblib")
    for ci, m in ocsvm_by_cluster.items():
        joblib.dump(m, out_dir / f"ocsvm_cluster_{ci:03d}.joblib")

    thresholds = {
        "normal_label": str(args.normal_label),
        "k": int(args.k),
        "min_cluster_size": int(args.min_cluster_size),
        "nu": float(args.nu),
        "gamma": args.gamma,
        "quantile": q,
        "global_threshold": global_thr,
        "cluster_thresholds": {str(k): float(v) for k, v in thr_by_cluster.items()},
        "cluster_sizes": {str(k): int(v) for k, v in cluster_sizes.items()},
        "note": "anomaly_score = -decision_function(ocsvm); anomaly if anomaly_score > threshold",
    }
    (out_dir / "thresholds.json").write_text(json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[OK] saved open-set artifacts to: %s" % out_dir)
    print("[OK] global_threshold(q=%.2f): %.6f" % (q, global_thr))
    print("[OK] files: kmeans.joblib, ocsvm_global.joblib, %d x ocsvm_cluster_*.joblib, thresholds.json" % len(ocsvm_by_cluster))


if __name__ == "__main__":
    main()

