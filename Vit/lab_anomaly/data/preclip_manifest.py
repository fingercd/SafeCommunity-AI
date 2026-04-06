"""
预切片缓存：manifest.json 读写、npz 存取、参数一致性校验。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1
NPZ_KEY = "frames"


def sanitize_dir_name(video_id: str, video_index: int) -> str:
    """
    生成用于磁盘目录名的安全字符串，避免 Windows 非法字符与重名。
    """
    s = "" if video_id is None else str(video_id).strip()
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    s = s.replace("\n", "_").replace("\r", "_")[:120]
    if not s:
        s = "vid"
    return f"{s}__v{int(video_index):05d}"


def npz_save(
    path: Path,
    frames: list[np.ndarray],
    *,
    compress: bool,
) -> None:
    """frames: list of HWC uint8 -> 存为 (T,H,W,C) 的 npz"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.stack([np.asarray(f, dtype=np.uint8) for f in frames], axis=0)
    if compress:
        np.savez_compressed(path, **{NPZ_KEY: arr})
    else:
        np.savez(path, **{NPZ_KEY: arr})


def npz_load_frames_list(path: Path) -> list[np.ndarray]:
    """读取 npz，返回 list[HWC uint8] 与现读现切一致，供 VideoMAE processor 使用。"""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if NPZ_KEY not in data.files:
            raise KeyError(f"{path} 中缺少键 {NPZ_KEY!r}")
        frames = data[NPZ_KEY]
    frames = np.asarray(frames, dtype=np.uint8)
    if frames.ndim != 4:
        raise ValueError(f"{path} frames 形状应为 (T,H,W,3)，当前 {frames.shape}")
    return [frames[i] for i in range(frames.shape[0])]


def build_params_blob(
    *,
    labels_csv: Path,
    dataset_root: Path,
    frames_per_clip: int,
    interval_sec: float,
    max_clips_per_video: int,
    exclude_unknown: bool,
    normal_label: str,
) -> dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION,
        "labels_csv": str(Path(labels_csv).resolve()),
        "dataset_root": str(Path(dataset_root).resolve()),
        "frames_per_clip": int(frames_per_clip),
        "interval_sec": float(interval_sec),
        "max_clips_per_video": int(max_clips_per_video),
        "exclude_unknown": bool(exclude_unknown),
        "normal_label": str(normal_label).strip().lower(),
    }


def assert_params_match(
    manifest_params: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    """训练/推理加载预切片时调用，不一致则抛错。"""
    keys = (
        "labels_csv",
        "dataset_root",
        "frames_per_clip",
        "interval_sec",
        "max_clips_per_video",
        "exclude_unknown",
        "normal_label",
    )
    for k in keys:
        a = manifest_params.get(k)
        b = expected.get(k)
        if a != b:
            raise RuntimeError(
                f"预切片 manifest 参数与当前配置不一致: {k!r} manifest={a!r} 当前={b!r}。"
                "请用相同配置重新运行预切片脚本，或清空 preclip_root。"
            )


def save_manifest(
    root: Path,
    *,
    params: dict[str, Any],
    videos: list[dict[str, Any]],
) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    out = {
        "version": MANIFEST_VERSION,
        "params": params,
        "videos": videos,
    }
    path = root / MANIFEST_NAME
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest(root: Path) -> dict[str, Any]:
    path = Path(root) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"找不到预切片索引: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "params" not in data or "videos" not in data:
        raise ValueError(f"manifest 格式错误: {path}")
    return data


def rel_path_for_clip(safe_dir: str, clip_index: int) -> str:
    return f"{safe_dir}/c{int(clip_index):04d}.npz"
