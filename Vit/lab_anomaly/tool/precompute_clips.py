"""
离线预切片：按与训练相同的规则切 clip，写入 .npz + manifest.json。
在 PyCharm 中直接运行本文件；参数见下方 CONFIG。
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

# ============ 可调 ============
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CONFIG: dict[str, Any] = {
    "dataset_root": str(PROJECT_ROOT / "lab_dataset"),
    "labels_csv": str(PROJECT_ROOT / "lab_dataset" / "labels" / "video_labels.csv"),
    "preclip_out_dir": str(PROJECT_ROOT / "lab_dataset" / "derived" / "preclips"),
    "yaml_config": str(PROJECT_ROOT / "lab_anomaly" / "configs" / "precompute_clips.yaml"),
    "frames_per_clip": 16,
    "interval_sec": 8.0,
    "max_clips_per_video": 16,
    "exclude_unknown": True,
    "normal_label": "normal",
    # 并行：同时处理多少个视频（0 表示不启用多进程，顺序处理）
    "num_workers": 20,
    # True：已存在的 npz 不覆盖（缺的仍会补）
    "skip_existing": True,
    # True：覆盖已有 npz（与 skip_existing 同时为 True 时，overwrite 优先于跳过）
    "overwrite": False,
    # True：npz 压缩（略省空间、略慢）
    "compress": True,
}


def _p(msg: str) -> None:
    print(msg, flush=True)
    sys.stdout.flush()


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
    root = project_root.resolve()
    for key in ("dataset_root", "labels_csv", "preclip_out_dir", "yaml_config"):
        value = cfg.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        p = Path(text)
        if not p.is_absolute():
            cfg[key] = str((root / p).resolve())


def _load_filtered_rows(labels_csv: Path, exclude_unknown: bool):
    from lab_anomaly.data.video_labels import read_video_labels_csv

    rows = read_video_labels_csv(labels_csv)
    if exclude_unknown:
        exclude = {"unknown"}
        rows = [r for r in rows if not (exclude and str(r.label).strip() in exclude)]
    return rows


def _precompute_one_video_worker(payload: dict) -> dict:
    """子进程入口：处理单个视频的全部 clip。"""
    try:
        from lab_anomaly.data.clip_dataset import clip_specs_for_row
        from lab_anomaly.data.preclip_manifest import npz_save, rel_path_for_clip, sanitize_dir_name
        from lab_anomaly.data.video_reader import read_frames_by_indices_cv2

        vi = int(payload["video_index"])
        labels_csv = Path(payload["labels_csv"])
        dataset_root = Path(payload["dataset_root"])
        out_root = Path(payload["preclip_out_dir"])

        rows = _load_filtered_rows(labels_csv, bool(payload["exclude_unknown"]))
        if vi < 0 or vi >= len(rows):
            return {"video_index": vi, "error": f"行号越界: {vi} / {len(rows)}", "rel_paths": [], "video_id": "", "label": ""}

        row = rows[vi]
        specs, abs_path = clip_specs_for_row(
            row,
            dataset_root,
            frames_per_clip=int(payload["frames_per_clip"]),
            interval_sec=float(payload["interval_sec"]),
            max_clips_per_video=int(payload["max_clips_per_video"]),
        )
        if not specs or abs_path is None:
            return {
                "video_index": vi,
                "error": None,
                "rel_paths": [],
                "video_id": row.video_id,
                "label": row.label,
            }

        safe = sanitize_dir_name(row.video_id, vi)
        overwrite = bool(payload["overwrite"])
        skip_existing = bool(payload["skip_existing"])
        compress = bool(payload["compress"])

        rel_paths: list[str] = []
        for ci, (indices, _start_f, _end_excl) in enumerate(specs):
            rel = rel_path_for_clip(safe, ci)
            full = out_root / rel
            if skip_existing and full.is_file() and not overwrite:
                rel_paths.append(rel)
                continue
            frames = read_frames_by_indices_cv2(abs_path, indices=indices, to_rgb=True)
            npz_save(full, frames, compress=compress)
            rel_paths.append(rel)

        return {
            "video_index": vi,
            "error": None,
            "rel_paths": rel_paths,
            "video_id": row.video_id,
            "label": row.label,
        }
    except Exception as e:  # pragma: no cover
        return {
            "video_index": int(payload.get("video_index", -1)),
            "error": repr(e),
            "rel_paths": [],
            "video_id": "",
            "label": "",
        }


def main() -> None:
    cfg = dict(CONFIG)
    yml = Path(str(cfg.get("yaml_config", "")))
    _load_yaml_merge(cfg, yml)
    _resolve_paths_under_project(cfg, PROJECT_ROOT)

    dataset_root = Path(str(cfg["dataset_root"]))
    labels_csv = Path(str(cfg["labels_csv"]))
    out_dir = Path(str(cfg["preclip_out_dir"]))

    rows = _load_filtered_rows(labels_csv, bool(cfg["exclude_unknown"]))
    n = len(rows)
    _p(f"过滤后视频行数: {n}")

    out_dir.mkdir(parents=True, exist_ok=True)

    payload_base = {
        "dataset_root": str(dataset_root.resolve()),
        "labels_csv": str(labels_csv.resolve()),
        "preclip_out_dir": str(out_dir.resolve()),
        "frames_per_clip": int(cfg["frames_per_clip"]),
        "interval_sec": float(cfg["interval_sec"]),
        "max_clips_per_video": int(cfg["max_clips_per_video"]),
        "exclude_unknown": bool(cfg["exclude_unknown"]),
        "overwrite": bool(cfg["overwrite"]),
        "skip_existing": bool(cfg["skip_existing"]),
        "compress": bool(cfg["compress"]),
    }

    nw = int(cfg["num_workers"])
    results: dict[int, dict] = {}

    if nw <= 0:
        for vi in range(n):
            pl = dict(payload_base)
            pl["video_index"] = vi
            results[vi] = _precompute_one_video_worker(pl)
            st = results[vi]
            err = st.get("error")
            rels = st.get("rel_paths") or []
            if err:
                _p(f"[{vi}/{n}] 失败: {err}")
            else:
                _p(f"[{vi}/{n}] OK {st.get('video_id')} clips={len(rels)}")
    else:
        _p(f"多进程 workers={nw}")
        with ProcessPoolExecutor(max_workers=nw) as ex:
            futs = []
            for vi in range(n):
                pl = dict(payload_base)
                pl["video_index"] = vi
                futs.append(ex.submit(_precompute_one_video_worker, pl))
            for j, fu in enumerate(as_completed(futs)):
                st = fu.result()
                vi = int(st["video_index"])
                results[vi] = st
                err = st.get("error")
                rels = st.get("rel_paths") or []
                if err:
                    _p(f"[完成 {j+1}/{n}] vi={vi} 失败: {err}")
                else:
                    _p(f"[完成 {j+1}/{n}] vi={vi} {st.get('video_id')} clips={len(rels)}")

    errors = [results[i] for i in sorted(results.keys()) if results[i].get("error")]
    if errors:
        raise RuntimeError(f"预切片有 {len(errors)} 个视频失败，请查看上方日志")

    from lab_anomaly.data.preclip_manifest import build_params_blob, save_manifest

    videos_manifest: list[dict[str, Any]] = []
    for vi in range(n):
        st = results[vi]
        videos_manifest.append(
            {
                "video_index": vi,
                "video_id": st.get("video_id", ""),
                "label": st.get("label", ""),
                "rel_paths": list(st.get("rel_paths") or []),
            }
        )

    params = build_params_blob(
        labels_csv=labels_csv,
        dataset_root=dataset_root,
        frames_per_clip=int(cfg["frames_per_clip"]),
        interval_sec=float(cfg["interval_sec"]),
        max_clips_per_video=int(cfg["max_clips_per_video"]),
        exclude_unknown=bool(cfg["exclude_unknown"]),
        normal_label=str(cfg["normal_label"]),
    )
    save_manifest(out_dir, params=params, videos=videos_manifest)

    meta = {
        "params": params,
        "num_videos": n,
        "preclip_out_dir": str(out_dir.resolve()),
    }
    (out_dir / "precompute_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _p(f"[OK] manifest 已写入: {out_dir / 'manifest.json'}")
    _p(f"     共 {n} 条视频记录，可直接在 train_end2end CONFIG 里设置 preclip_root 为:")
    _p(f"     {out_dir.resolve()}")


if __name__ == "__main__":
    main()
