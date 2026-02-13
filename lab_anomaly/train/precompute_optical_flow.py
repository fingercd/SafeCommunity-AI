"""
阶段一：为所有视频预计算光流并缓存到磁盘。
与 extract_embeddings 使用相同的 clip 采样逻辑，保证帧索引一致。
输出供双流 encoder 与 SSIM+光流过滤使用。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from lab_anomaly.data.video_labels import read_video_labels_csv
from lab_anomaly.data.video_reader import (
    get_video_info,
    read_frames_by_indices_cv2,
    uniform_clip_indices,
)


def safe_id(s: str) -> str:
    """将 video_id 转为文件系统安全字符串"""
    s = s.replace("\\", "/")
    s = re.sub(r"[^0-9a-zA-Z._/-]+", "_", s)
    s = s.replace("/", "__")
    return s


def _parse_float_or_none(x: str) -> Optional[float]:
    """解析秒数等浮点值，非法或空则返回 None"""
    x = "" if x is None else str(x).strip()
    if not x:
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _time_range_to_frame_range(
    row: Any,
    fps: float,
    frame_count: int,
) -> tuple[int, int]:
    """将 CSV 中的 start_time/end_time（秒）转为帧索引范围 [start, end_exclusive)"""
    frame_count = max(0, int(frame_count))
    if frame_count <= 0:
        return 0, 0
    st = _parse_float_or_none(getattr(row, "start_time", "") or "")
    et = _parse_float_or_none(getattr(row, "end_time", "") or "")
    if st is None and et is None:
        return 0, frame_count
    fps = float(fps or 0.0)
    if fps <= 1e-3:
        fps = 25.0
    start = 0 if st is None else int(max(0.0, st) * fps)
    end_excl = frame_count if et is None else int(max(0.0, et) * fps)
    start = max(0, min(start, frame_count))
    end_excl = max(0, min(end_excl, frame_count))
    if end_excl <= start:
        return 0, frame_count
    return start, end_excl


def _load_raft_model(device: str):
    """加载 RAFT 模型（torchvision），失败则返回 None"""
    try:
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        import torch
        dev = device if device in ("cpu", "cuda") else ("cuda" if torch.cuda.is_available() else "cpu")
        model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False)
        model = model.to(dev).eval()
        return model, dev
    except Exception as e:
        return None, str(e)


def compute_flow_raft(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    model: Any,
    device: str,
    transforms: Any,
    resize: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """
    使用 torchvision RAFT 在 GPU/CPU 上计算光流。
    frame_a, frame_b: RGB HWC uint8
    Returns: (2, H, W) float32
    """
    import torch
    import torchvision.transforms.functional as F_tv

    if frame_a.shape[:2] != frame_b.shape[:2]:
        raise ValueError("frame shapes must match")
    # (H,W,3) -> (3,H,W) tensor float
    t1 = torch.from_numpy(np.ascontiguousarray(frame_a.transpose(2, 0, 1))).float() / 255.0
    t2 = torch.from_numpy(np.ascontiguousarray(frame_b.transpose(2, 0, 1))).float() / 255.0
    w, h = frame_a.shape[1], frame_a.shape[0]
    if resize:
        w, h = resize[0], resize[1]
    # RAFT 需要尺寸能被 8 整除
    w = (w // 8) * 8
    h = (h // 8) * 8
    if w < 8:
        w = 8
    if h < 8:
        h = 8
    t1 = F_tv.resize(t1, [h, w], antialias=False)
    t2 = F_tv.resize(t2, [h, w], antialias=False)
    t1, t2 = transforms(t1, t2)
    t1 = t1.unsqueeze(0).to(device)
    t2 = t2.unsqueeze(0).to(device)
    with torch.no_grad():
        flows = model(t1, t2)
    flow = flows[-1][0].cpu().numpy()
    return flow.astype(np.float32)


def compute_flow_farneback(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    resize: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """
    使用 OpenCV Farneback 计算两帧之间的光流。
    frame_a, frame_b: RGB HWC uint8 或 BGR
    Returns: (2, H, W) float32, 通道顺序 (u, v)
    """
    if frame_a.shape[:2] != frame_b.shape[:2]:
        raise ValueError("frame shapes must match")
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY) if frame_a.ndim == 3 else frame_a
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY) if frame_b.ndim == 3 else frame_b
    if resize:
        gray_a = cv2.resize(gray_a, resize)
        gray_b = cv2.resize(gray_b, resize)
    flow = cv2.calcOpticalFlowFarneback(
        gray_a,
        gray_b,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    # flow: (H, W, 2) -> (2, H, W)
    flow = np.asarray(flow, dtype=np.float32).transpose(2, 0, 1)
    return flow


def compute_flows_for_clip(
    frames: list[np.ndarray],
    backend: str = "farneback",
    resize: Optional[tuple[int, int]] = None,
    flownet_model: Any = None,
    device: str = "cpu",
    raft_model: Any = None,
    raft_transforms: Any = None,
) -> np.ndarray:
    """
    对一组连续帧计算帧间光流。
    frames: list of RGB HWC uint8, 长度 clip_len
    Returns: (clip_len-1, 2, H, W) float32
    """
    n = len(frames)
    if n < 2:
        h, w = 224, 224
        if resize:
            w, h = resize[0], resize[1]
        return np.zeros((0, 2, h, w), dtype=np.float32)

    flows_list: list[np.ndarray] = []
    for i in range(n - 1):
        if backend == "raft" and raft_model is not None and raft_transforms is not None:
            flow = compute_flow_raft(
                frames[i], frames[i + 1],
                model=raft_model, device=device, transforms=raft_transforms,
                resize=resize,
            )
        else:
            flow = compute_flow_farneback(frames[i], frames[i + 1], resize=resize)
        flows_list.append(flow)

    return np.stack(flows_list, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="预计算视频光流，与 extract_embeddings 的 clip 采样一致"
    )
    ap.add_argument("--dataset_root", type=str, default="lab_dataset")
    ap.add_argument("--labels_csv", type=str, default="lab_dataset/labels/video_labels.csv")
    ap.add_argument("--out_dir", type=str, default="lab_dataset/derived/optical_flows")
    ap.add_argument("--clip_len", type=int, default=16)
    ap.add_argument("--frame_stride", type=int, default=2)
    ap.add_argument("--num_clips_per_video", type=int, default=32)
    ap.add_argument(
        "--backend",
        type=str,
        default="raft",
        choices=["farneback", "raft"],
        help="farneback=OpenCV CPU; raft=torchvision RAFT，支持 GPU 加速",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="光流计算设备（仅 raft 时生效）: cpu/cuda/auto",
    )
    ap.add_argument(
        "--resize",
        type=str,
        default="384,384",
        help="光流计算分辨率 'W,H'，减小可加速，如 224,224",
    )
    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="若该视频的 flows.npz 已存在则跳过",
    )
    ap.add_argument(
        "--video_filter",
        type=str,
        default="all",
        help="all | non_unknown，仅处理 label!=unknown 时用 non_unknown",
    )
    args = ap.parse_args()

    _project_root = Path(__file__).resolve().parent.parent.parent
    if not Path(args.dataset_root).is_absolute():
        args.dataset_root = str(_project_root / args.dataset_root)
    if not Path(args.labels_csv).is_absolute():
        args.labels_csv = str(_project_root / args.labels_csv)
    if not Path(args.out_dir).is_absolute():
        args.out_dir = str(_project_root / args.out_dir)

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resize_tuple: Optional[tuple[int, int]] = None
    if args.resize:
        parts = args.resize.strip().split(",")
        if len(parts) == 2:
            try:
                resize_tuple = (int(parts[0].strip()), int(parts[1].strip()))
            except ValueError:
                pass

    raft_model = None
    raft_transforms = None
    flow_device = "cpu"
    if args.backend == "raft":
        raft_model, raft_dev = _load_raft_model(args.device)
        if raft_model is None:
            print(f"[WARN] RAFT 加载失败 ({raft_dev})，回退到 farneback CPU")
            args.backend = "farneback"
        else:
            flow_device = raft_dev
            try:
                from torchvision.models.optical_flow import Raft_Small_Weights
                raft_transforms = Raft_Small_Weights.DEFAULT.transforms()
            except Exception:
                raft_transforms = None
            if raft_transforms is not None:
                print(f"[OK] RAFT 已加载，device={flow_device}")
            else:
                print("[WARN] RAFT transforms 获取失败，回退到 farneback")
                args.backend = "farneback"
                raft_model = None

    rows = read_video_labels_csv(args.labels_csv)
    if args.video_filter == "non_unknown":
        rows = [r for r in rows if str(r.label).strip().lower() != "unknown"]
    if not rows:
        raise RuntimeError("no videos to process")

    meta_path = out_dir / "flows_meta.jsonl"
    meta_lines: list[dict[str, Any]] = []

    for vi, row in enumerate(rows):
        video_path = row.video_path
        abs_path = dataset_root / video_path if not Path(video_path).is_absolute() else Path(video_path)
        if not abs_path.exists():
            print(f"[skip] video not found: {abs_path}")
            continue

        vid_safe = safe_id(row.video_id)
        video_out_dir = out_dir / vid_safe
        flows_npz = video_out_dir / "flows.npz"
        if args.skip_existing and flows_npz.exists():
            meta_lines.append({
                "video_id": row.video_id,
                "video_path": row.video_path,
                "label": row.label,
                "flows_path": str(flows_npz.as_posix()),
                "num_clips": args.num_clips_per_video,
            })
            print(f"  [{vi+1}/{len(rows)}] skip (exists): {row.video_id}")
            continue

        try:
            info = get_video_info(abs_path)
        except Exception as e:
            print(f"[skip] cannot open {abs_path}: {e}")
            continue

        range_start, range_end_excl = _time_range_to_frame_range(
            row, fps=info.fps, frame_count=info.frame_count
        )
        span = max(0, range_end_excl - range_start)

        all_flows: list[np.ndarray] = []
        clip_indices_list: list[int] = []

        for ci in range(args.num_clips_per_video):
            indices_local, start_local, end_local_excl = uniform_clip_indices(
                frame_count=span,
                clip_len=args.clip_len,
                frame_stride=args.frame_stride,
                clip_idx=ci,
                num_clips=args.num_clips_per_video,
            )
            indices = [range_start + x for x in indices_local]
            frames = read_frames_by_indices_cv2(abs_path, indices=indices, to_rgb=True)
            if len(frames) < 2:
                h, w = 224, 224
                if resize_tuple:
                    h, w = resize_tuple[1], resize_tuple[0]
                all_flows.append(
                    np.zeros((max(0, args.clip_len - 1), 2, h, w), dtype=np.float32)
                )
                clip_indices_list.append(ci)
                continue

            flow_clip = compute_flows_for_clip(
                frames,
                backend=args.backend,
                resize=resize_tuple,
                device=flow_device,
                raft_model=raft_model,
                raft_transforms=raft_transforms,
            )
            all_flows.append(flow_clip)
            clip_indices_list.append(ci)

        flows_stack = np.stack(all_flows, axis=0)
        video_out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            flows_npz,
            flows=flows_stack,
            clip_indices=np.array(clip_indices_list, dtype=np.int32),
            video_id=np.array([row.video_id], dtype=object),
            label=np.array([row.label], dtype=object),
        )
        meta_lines.append({
            "video_id": row.video_id,
            "video_path": row.video_path,
            "label": row.label,
            "flows_path": str(flows_npz.as_posix()),
            "num_clips": int(flows_stack.shape[0]),
        })
        print(f"  [{vi+1}/{len(rows)}] {row.video_id} -> {flows_npz} (shape {flows_stack.shape})")

    with meta_path.open("w", encoding="utf-8") as f:
        for rec in meta_lines:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n[OK] flows saved under: {out_dir}")
    print(f"[OK] meta: {meta_path}")


if __name__ == "__main__":
    main()
