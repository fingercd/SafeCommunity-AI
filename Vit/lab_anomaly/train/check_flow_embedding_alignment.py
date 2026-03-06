"""
独立校验脚本：检查 embedding 与光流是否对齐。

只读 embeddings_meta.jsonl 与各视频的 flows.npz，按视频比较：
  - embedding clip 数（由 meta 推断）
  - flows.shape[0]（光流 clip 数）

判定：对齐 / 错位（embedding 数 > flow 数）/ 缺流（无 flows.npz）。
不修改 extract_embeddings.py，仅做校验；退出码 0 表示全部对齐，1 表示存在错位或缺流。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any


def safe_id(s: str) -> str:
    """将 video_id 转为文件系统安全字符串（与 precompute_optical_flow 一致）"""
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


def get_per_video_embedding_clip_counts(embeddings_dir: Path) -> dict[str, int]:
    """
    从 embeddings_meta.jsonl 按视频统计 clip 数（与 EmbeddingMILVideoDataset 逻辑一致）。
    同一 video_id 同时有 npy 与 npz 时，优先按 npy 条数计。
    返回: { video_id: num_clips }
    """
    meta_path = embeddings_dir / "embeddings_meta.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"cannot find embeddings meta: {meta_path}")

    rows = read_jsonl(meta_path)
    by_video_npy: dict[str, list[dict[str, Any]]] = {}
    by_video_npz: dict[str, dict[str, Any]] = {}

    for r in rows:
        fmt = str(r.get("save_format", "npy_per_clip"))
        vid = str(r["video_id"])
        if fmt == "npz_per_video":
            if vid not in by_video_npz:
                by_video_npz[vid] = r
        else:
            by_video_npy.setdefault(vid, []).append(r)

    result: dict[str, int] = {}
    for vid, clips in by_video_npy.items():
        result[vid] = len(clips)
    for vid, r in by_video_npz.items():
        if vid in by_video_npy:
            continue
        result[vid] = int(r.get("num_clips", 0) or 0)
    return result


def _get_flows_shape0_from_npz(npz_path: Path) -> int:
    """
    从 npz 中只读 flows 的 shape[0]（clip 数），不加载整块数组，避免解压大文件。
    npz 为 zip，内为 flows.npy；.npy 头含 'shape': (N, ...)，解析出 N。
    """
    with zipfile.ZipFile(npz_path, "r") as zf:
        if "flows.npy" not in zf.namelist():
            raise ValueError("flows.npy not in npz")
        with zf.open("flows.npy") as f:
            magic = f.read(6)
            if magic != b"\x93NUMPY":
                raise ValueError("invalid npy magic")
            version = f.read(2)  # major, minor
            if version[0] == 1:
                hlen = int.from_bytes(f.read(2), "little")
            else:
                hlen = int.from_bytes(f.read(4), "little")
            header = f.read(min(hlen, 512)).decode("ascii", errors="replace")
    # header 形如 "{'descr': '<f4', 'fortran_order': False, 'shape': (16, 15, 2, 224, 224), }"
    m = re.search(r"['\"]shape['\"]\s*:\s*\(\s*(\d+)", header)
    if not m:
        raise ValueError("shape not found in npy header")
    return int(m.group(1))


def _default_dirs() -> tuple[str, str]:
    """默认路径基于脚本所在位置：项目根 = lab_anomaly/train 的上两级（即含 lab_dataset 的目录）"""
    script_dir = Path(__file__).resolve().parent  # lab_anomaly/train
    project_root = script_dir.parent.parent  # 项目根，如 Vit
    emb = str(project_root / "lab_dataset" / "derived" / "embeddings")
    flow = str(project_root / "lab_dataset" / "derived" / "optical_flows")
    return emb, flow


def main() -> int:
    default_emb, default_flow = _default_dirs()
    ap = argparse.ArgumentParser(
        description="校验 embedding 与光流对齐：按视频比较 clip 数，报告对齐/错位/缺流"
    )
    ap.add_argument(
        "--embeddings_dir",
        type=str,
        default=default_emb,
        help="含 embeddings_meta.jsonl 的目录",
    )
    ap.add_argument(
        "--flows_dir",
        type=str,
        default=default_flow,
        help="预计算光流根目录，其下为 safe_id(video_id)/flows.npz",
    )
    ap.add_argument(
        "--list_missing",
        action="store_true",
        help="在输出中列出缺流视频的 video_id",
    )
    args = ap.parse_args()

    embeddings_dir = Path(args.embeddings_dir).resolve()
    flows_dir = Path(args.flows_dir).resolve()

    try:
        video_clip_counts = get_per_video_embedding_clip_counts(embeddings_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not video_clip_counts:
        print("No videos found in embeddings_meta.jsonl.")
        return 0

    aligned: list[tuple[str, int, int]] = []
    misaligned: list[tuple[str, int, int]] = []
    missing_flow: list[str] = []

    total_videos = len(video_clip_counts)
    for idx, (vid, emb_clips) in enumerate(video_clip_counts.items(), start=1):
        flows_path = flows_dir / safe_id(vid) / "flows.npz"
        if not flows_path.exists():
            missing_flow.append(vid)
            print(f"[{idx}/{total_videos}] {vid!r} 缺流 (无 flows.npz)", flush=True)
            continue
        try:
            flow_clips = _get_flows_shape0_from_npz(flows_path)
        except Exception as e:
            print(f"Warning: failed to read {flows_path}: {e}", file=sys.stderr)
            missing_flow.append(vid)
            print(f"[{idx}/{total_videos}] {vid!r} 缺流 (读取失败)", flush=True)
            continue

        if emb_clips > flow_clips:
            misaligned.append((vid, emb_clips, flow_clips))
            print(f"[{idx}/{total_videos}] {vid!r} 错位  emb={emb_clips}  flow={flow_clips}", flush=True)
        else:
            aligned.append((vid, emb_clips, flow_clips))
            print(f"[{idx}/{total_videos}] {vid!r} 对齐  emb={emb_clips}  flow={flow_clips}", flush=True)

    total = len(video_clip_counts)
    n_aligned = len(aligned)
    n_misaligned = len(misaligned)
    n_missing = len(missing_flow)

    print()
    print(f"Total videos (with embeddings): {total}")
    print(f"Aligned:   {n_aligned}  (flow_clips >= embedding_clips)")
    print(f"Misaligned: {n_misaligned}  (embedding_clips > flow_clips)")
    print(f"Missing flow: {n_missing}  (no flows.npz)")

    if misaligned:
        print("\nMisaligned (video_id, embedding_clips, flow_clips):")
        for vid, ec, fc in misaligned:
            print(f"  {vid!r}  {ec}  {fc}")

    if missing_flow and args.list_missing:
        print("\nMissing flow (video_id):")
        for vid in missing_flow:
            print(f"  {vid!r}")

    if n_misaligned > 0 or n_missing > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
