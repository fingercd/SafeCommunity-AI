"""
按 KMeans 簇把 normal 的 clip 导出成短视频，每个簇一个文件夹，便于查看各簇对应的视频类型。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any  # noqa: F401 - used in type hints

import cv2
import numpy as np


def safe_id(s: str) -> str:
    """将 video_id 转为文件系统安全字符串"""
    s = s.replace("\\", "/")
    s = re.sub(r"[^0-9a-zA-Z._/-]+", "_", s)
    s = s.replace("/", "__")
    return s


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


def main() -> None:
    ap = argparse.ArgumentParser(description="按簇导出 clip 为短视频，每簇一个文件夹")
    ap.add_argument("--embeddings_dir", type=str, default="lab_dataset/derived/embeddings")
    ap.add_argument("--open_set_dir", type=str, default="lab_dataset/derived/open_set")
    ap.add_argument("--dataset_root", type=str, default="lab_dataset")
    ap.add_argument("--out_dir", type=str, default="lab_dataset/derived/cluster_samples")
    ap.add_argument("--normal_label", type=str, default="normal")
    ap.add_argument("--max_per_cluster", type=int, default=50, help="每簇最多导出多少个 clip（避免某簇过大占满磁盘）")
    ap.add_argument("--fps", type=float, default=0, help="导出视频 FPS，0=从原视频读取")
    args = ap.parse_args()

    _project_root = Path(__file__).resolve().parent.parent.parent
    if not Path(args.embeddings_dir).is_absolute():
        args.embeddings_dir = str(_project_root / args.embeddings_dir)
    if not Path(args.open_set_dir).is_absolute():
        args.open_set_dir = str(_project_root / args.open_set_dir)
    if not Path(args.dataset_root).is_absolute():
        args.dataset_root = str(_project_root / args.dataset_root)
    if not Path(args.out_dir).is_absolute():
        args.out_dir = str(_project_root / args.out_dir)

    try:
        import joblib
    except Exception as e:
        raise RuntimeError("缺少 joblib。请安装：pip install joblib") from e

    from lab_anomaly.data.video_reader import get_video_info, read_frames_by_indices_cv2

    embeddings_dir = Path(args.embeddings_dir)
    open_set_dir = Path(args.open_set_dir)
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)

    meta_path = embeddings_dir / "embeddings_meta.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"cannot find {meta_path}")
    if not (open_set_dir / "kmeans.joblib").exists():
        raise FileNotFoundError(f"cannot find {open_set_dir / 'kmeans.joblib'}，请先运行 fit_kmeans_ocsvm.py")

    rows = read_jsonl(meta_path)
    normal_rows = [r for r in rows if str(r.get("label", "")) == str(args.normal_label)]
    if not normal_rows:
        raise RuntimeError(f"no normal rows with label={args.normal_label!r}")

    # 加载 embedding 并预测簇
    xs: list[np.ndarray] = []
    meta_per_clip: list[dict[str, Any]] = []  # 每条: video_path, video_id, clip_index, start_frame, end_frame_exclusive

    for r in normal_rows:
        fmt = str(r.get("save_format", "npy_per_clip"))
        p = str(r["embedding_path"])
        full_path = _resolve_emb_path(embeddings_dir, p)
        if fmt == "npz_per_video" or p.lower().endswith(".npz"):
            npz = np.load(full_path, allow_pickle=True)
            E = np.asarray(npz["embeddings"], dtype=np.float32)
            clip_idx_arr = np.asarray(npz["clip_index"]).astype(np.int32) if "clip_index" in npz.files else np.arange(E.shape[0])
            start_arr = np.asarray(npz["start_frame"]).astype(np.int32) if "start_frame" in npz.files else None
            end_arr = np.asarray(npz["end_frame_exclusive"]).astype(np.int32) if "end_frame_exclusive" in npz.files else None
            video_path = str(r.get("video_path", ""))
            for i in range(E.shape[0]):
                xs.append(E[i])
                start_f = int(start_arr[i]) if start_arr is not None else 0
                end_f = int(end_arr[i]) if end_arr is not None else start_f + 16
                meta_per_clip.append({
                    "video_path": video_path,
                    "video_id": r.get("video_id", ""),
                    "clip_index": int(clip_idx_arr[i]) if clip_idx_arr is not None else i,
                    "start_frame": start_f,
                    "end_frame_exclusive": end_f,
                })
        else:
            x = np.load(full_path).astype(np.float32)
            xs.append(x)
            meta_per_clip.append({
                "video_path": str(r.get("video_path", "")),
                "video_id": str(r.get("video_id", "")),
                "clip_index": int(r.get("clip_index", 0)),
                "start_frame": int(r.get("start_frame", 0)),
                "end_frame_exclusive": int(r.get("end_frame_exclusive", 0)),
            })

    X = np.stack(xs, axis=0)
    kmeans = joblib.load(open_set_dir / "kmeans.joblib")
    clusters = kmeans.predict(X)

    # 按簇分组，每簇最多 max_per_cluster 个
    k = int(np.max(clusters)) + 1
    out_dir.mkdir(parents=True, exist_ok=True)

    for c in range(k):
        idx = np.where(clusters == c)[0]
        if idx.size == 0:
            continue
        if idx.size > args.max_per_cluster:
            rng = np.random.default_rng(42)
            idx = rng.choice(idx, size=args.max_per_cluster, replace=False)
        cluster_dir = out_dir / f"cluster_{c}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        for pos, i in enumerate(idx):
            m = meta_per_clip[i]
            video_path = m["video_path"]
            abs_video = dataset_root / video_path if not Path(video_path).is_absolute() else Path(video_path)
            if not abs_video.exists():
                print(f"[skip] video not found: {abs_video}")
                continue
            start_f = m["start_frame"]
            end_f = m["end_frame_exclusive"]
            indices = list(range(start_f, end_f))
            if not indices:
                indices = [start_f]

            try:
                frames = read_frames_by_indices_cv2(abs_video, indices, to_rgb=True)
            except Exception as e:
                print(f"[skip] read error {abs_video}: {e}")
                continue
            if not frames:
                continue

            if args.fps > 0:
                fps = args.fps
            else:
                try:
                    info = get_video_info(abs_video)
                    fps = info.fps if info.fps > 0 else 25.0
                except Exception:
                    fps = 25.0

            h, w = frames[0].shape[:2]
            vid_safe = safe_id(m["video_id"])
            out_name = f"{vid_safe}_clip{m['clip_index']:03d}.mp4"
            out_path = cluster_dir / out_name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
            for fr in frames:
                bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
            writer.release()

        print(f"  cluster_{c}: {len(list(cluster_dir.glob('*.mp4')))} clips -> {cluster_dir}")

    print(f"\n[OK] 导出目录: {out_dir}")
    print("说明: 每个 cluster_* 文件夹内为该簇的 clip 短视频，可打开查看该簇对应的视频类型。")

if __name__ == "__main__":
    main()
