"""
批量提取 ViT 视频 embeddings（缓存到 lab_dataset/derived/embeddings）

功能：
  - 从 video_labels.csv 读取视频列表，按 clip_len / frame_stride / num_clips_per_video 均匀采样 clip
  - 每个 clip 经 VideoMAE ViT 编码得到 768 维 embedding（可选双流 RGB+光流 → 1536 维）
  - 支持两种缓存格式：npy_per_clip（每 clip 一个 .npy）、npz_per_video（每视频一个 .npz）
  - 可选 SSIM+光流过滤：去除静态/低运动 clip，减少噪声
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def safe_id(s: str) -> str:
    """将路径或 ID 转为文件系统安全的字符串（替换特殊字符，避免路径冲突）"""
    s = s.replace("\\", "/")
    s = re.sub(r"[^0-9a-zA-Z._/-]+", "_", s)
    s = s.replace("/", "__")
    return s


def _load_yaml(path: str | Path) -> dict[str, Any]:
    """加载 YAML 配置文件，顶层必须是 dict"""
    try:
        import yaml
    except Exception as e:  # pragma: no cover
        raise RuntimeError("缺少 PyYAML 依赖。请安装：pip install pyyaml") from e
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML 顶层必须是 dict：{path}")
    return data


def _cfg_get(d: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    """从嵌套 dict 中按 keys 路径取值，缺则返回 default"""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _to_bool(val: Any, name: str = "") -> bool:
    """
    安全地将 YAML 值转为 bool。
    防止 typo（如 'flase'、'ture'）被 bool(str) 误判为 True。
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in {"true", "1", "yes", "on"}:
            return True
        if s in {"false", "0", "no", "off", ""}:
            return False
        raise ValueError(
            f"配置项 '{name}' 的值 '{val}' 无法识别为布尔值。"
            f"请使用 true/false（检查是否有拼写错误，如 'flase'）"
        )
    return bool(val)


def _resolve_device(device_str: str) -> str:
    """解析设备字符串：auto 时自动选 cuda/cpu"""
    s = (device_str or "").strip().lower()
    if s in {"", "auto"}:
        try:
            import torch  # 延迟导入：确保 --help 在无 torch 环境也可用
        except Exception:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description="批量提取 ViT 视频 embeddings（缓存到 lab_dataset/derived/embeddings）")
    ap.add_argument(
        "--config",
        type=str,
        default="lab_anomaly/configs/embedding_example.yaml",
        help="YAML 配置文件，默认 lab_anomaly/configs/embedding_example.yaml；传空字符串可禁用",
    )
    ap.add_argument("--dataset_root", type=str, default="lab_dataset")
    ap.add_argument("--labels_csv", type=str, default="lab_dataset/labels/video_labels.csv")
    ap.add_argument("--out_dir", type=str, default="lab_dataset/derived/embeddings")
    ap.add_argument("--model_name", type=str, default="MCG-NJU/videomae-base")
    ap.add_argument("--pooling", type=str, default="auto", help="embedding 池化方式：auto/pooler/cls/mean")
    ap.add_argument("--use_half", action="store_true", help="启用 FP16（仅 CUDA 有效）")
    ap.add_argument("--no_half", action="store_true", help="禁用 FP16（覆盖 --use_half 或 YAML）")
    ap.add_argument("--clip_len", type=int, default=16)
    ap.add_argument("--frame_stride", type=int, default=2)
    ap.add_argument("--num_clips_per_video", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=32, help="4090 可用 32～64，显存小可改 8 或 16")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 个 clip（0=不限制）")
    ap.add_argument("--device", type=str, default="", help="cuda/cpu（默认自动）")
    ap.add_argument(
        "--save_format",
        type=str,
        default="npy_per_clip",
        choices=["npy_per_clip", "npz_per_video"],
        help="缓存格式：每个 clip 一个 .npy，或每个视频一个 .npz（包含全部 clip embeddings）",
    )
    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="若输出文件已存在则跳过（断点续跑）",
    )
    ap.add_argument("--enable_filtering", action="store_true", default=True, help="启用 SSIM 过滤，去除静态 clips")
    ap.add_argument("--enable_flow_filter", action="store_true", default=True, help="启用光流幅值过滤（默认开启）")
    ap.add_argument("--no_flow_filter", action="store_true", help="关闭光流幅值过滤，保留 SSIM")
    ap.add_argument("--no_filtering", action="store_true", help="关闭全部过滤（SSIM+光流）")
    ap.add_argument("--ssim_threshold", type=float, default=0.99, help="SSIM 高于此值视为静态，丢弃")
    ap.add_argument("--flow_threshold", type=float, default=0.1, help="光流幅值均值低于此值视为运动不足，丢弃")
    ap.add_argument("--flows_dir", type=str, default="lab_dataset/derived/optical_flows", help="光流缓存目录，用于二级过滤与双流")
    ap.add_argument("--use_dual_stream", action="store_true", help="启用 RGB+光流双流 encoder，输出 1536 维")
    ap.add_argument("--fusion_method", type=str, default="concat", choices=["concat", "add", "mlp"])
    args = ap.parse_args()

    # 相对路径按项目根解析，避免从非项目根运行时找不到 lab_dataset
    _project_root = Path(__file__).resolve().parent.parent.parent
    # 配置文件路径：相对路径按项目根解析
    _config_path: Path | None = None
    if args.config.strip():
        _config_path = Path(args.config) if Path(args.config).is_absolute() else _project_root / args.config
    if not Path(args.dataset_root).is_absolute():
        args.dataset_root = str(_project_root / args.dataset_root)
    if not Path(args.labels_csv).is_absolute():
        args.labels_csv = str(_project_root / args.labels_csv)
    if not Path(args.out_dir).is_absolute():
        args.out_dir = str(_project_root / args.out_dir)
    if not Path(args.flows_dir).is_absolute():
        args.flows_dir = str(_project_root / args.flows_dir)

    # 说明：parse_args 遇到 --help 会直接退出，因此把重依赖放到 parse_args 后面导入，
    # 这样即便环境里没装 torch，也可以先查看帮助/参数。
    try:
        import numpy as np
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError("缺少运行依赖（numpy/torch）。请先安装后再运行提取脚本。") from e

    from lab_anomaly.data.clip_dataset import VideoClipDataset
    from lab_anomaly.data.clip_filters import compute_optical_flow_magnitude, compute_ssim_variation
    from lab_anomaly.models.vit_video_encoder import HfVideoEncoder, HfVideoEncoderConfig

    # YAML 配置优先提供“默认值”，CLI 显式传参会覆盖这些默认值
    if _config_path is not None and _config_path.exists():
        cfg = _load_yaml(_config_path)
        print(f"[config] loaded: {_config_path}")
        args.dataset_root = _cfg_get(cfg, ["dataset_root"], args.dataset_root)
        args.labels_csv = _cfg_get(cfg, ["labels_csv"], args.labels_csv)
        args.out_dir = _cfg_get(cfg, ["out_dir"], args.out_dir)
        args.model_name = _cfg_get(cfg, ["encoder", "model_name"], args.model_name)
        args.pooling = _cfg_get(cfg, ["encoder", "pooling"], args.pooling)
        args.device = str(_cfg_get(cfg, ["encoder", "device"], args.device) or args.device)
        yaml_use_half = bool(_cfg_get(cfg, ["encoder", "use_half"], False))
        if yaml_use_half:
            args.use_half = True

        args.clip_len = int(_cfg_get(cfg, ["sampling", "clip_len"], args.clip_len))
        args.frame_stride = int(_cfg_get(cfg, ["sampling", "frame_stride"], args.frame_stride))
        args.num_clips_per_video = int(_cfg_get(cfg, ["sampling", "num_clips_per_video"], args.num_clips_per_video))

        args.batch_size = int(_cfg_get(cfg, ["runtime", "batch_size"], args.batch_size))
        args.limit = int(_cfg_get(cfg, ["runtime", "limit"], args.limit))
        args.use_dual_stream = _to_bool(
            _cfg_get(cfg, ["encoder", "use_dual_stream"], args.use_dual_stream),
            name="encoder.use_dual_stream",
        )
        args.fusion_method = str(_cfg_get(cfg, ["encoder", "fusion_method"], args.fusion_method))
        args.flows_dir = str(_cfg_get(cfg, ["flows", "flows_dir"], args.flows_dir))
        args.enable_filtering = _to_bool(
            _cfg_get(cfg, ["enable_filtering"], args.enable_filtering),
            name="enable_filtering",
        )
        args.enable_flow_filter = _to_bool(
            _cfg_get(cfg, ["enable_flow_filter"], args.enable_flow_filter),
            name="enable_flow_filter",
        )
    elif _config_path is not None:
        print(f"[config] not found: {_config_path}，使用 argparse 默认值")

    # YAML 覆盖后再次解析相对路径
    if not Path(args.dataset_root).is_absolute():
        args.dataset_root = str(_project_root / args.dataset_root)
    if not Path(args.labels_csv).is_absolute():
        args.labels_csv = str(_project_root / args.labels_csv)
    if not Path(args.out_dir).is_absolute():
        args.out_dir = str(_project_root / args.out_dir)
    if not Path(args.flows_dir).is_absolute():
        args.flows_dir = str(_project_root / args.flows_dir)

    if args.no_half:
        args.use_half = False
    if args.no_filtering:
        args.enable_filtering = False
        args.enable_flow_filter = False
    if args.no_flow_filter:
        args.enable_flow_filter = False

    device = _resolve_device(args.device)
    # GPU 时默认 FP16，加快推理
    if device == "cuda" and not args.no_half:
        args.use_half = True
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "embeddings_meta.jsonl"

    ds = VideoClipDataset(
        dataset_root=args.dataset_root,
        labels_csv=args.labels_csv,
        clip_len=args.clip_len,
        frame_stride=args.frame_stride,
        num_clips_per_video=args.num_clips_per_video,
        shuffle_clips=False,
        video_filter=lambda r: r.label != "unknown",
    )

    enc_cfg = HfVideoEncoderConfig(
        model_name=args.model_name,
        use_half=bool(args.use_half),
        pooling=str(args.pooling),
        use_dual_stream=getattr(args, "use_dual_stream", False),
        flow_embedding_dim=768,
        fusion_method=getattr(args, "fusion_method", "concat"),
    )
    enc = HfVideoEncoder(enc_cfg, device=device)
    enc.eval()
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    processed = 0  # clip 计数（两种格式都统一统计 clip 数）
    total_clips = len(ds)
    # ── 启动配置摘要（一目了然哪些功能开着）──
    print("=" * 60)
    print("  Embedding 提取 - 配置摘要")
    print("=" * 60)
    print(f"  device           : {device}" + (f" (FP16={args.use_half})" if device == "cuda" else ""))
    print(f"  model_name       : {args.model_name}")
    print(f"  use_dual_stream  : {args.use_dual_stream}")
    print(f"  enable_filtering : {args.enable_filtering} (SSIM)")
    print(f"  enable_flow_filter: {args.enable_flow_filter}")
    print(f"  clip_len={args.clip_len}  frame_stride={args.frame_stride}  num_clips_per_video={args.num_clips_per_video}")
    print(f"  batch_size={args.batch_size}  save_format={args.save_format}")
    print(f"  total videos     : {len(ds.rows)}")
    print(f"  total clips      : {total_clips}")
    print("=" * 60)
    print("Starting extraction...")

    def _append_meta_line(fh, rec: dict[str, Any]) -> None:
        """往 jsonl 文件追加一行 meta 记录"""
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _save_npy_per_clip() -> None:
        """按 clip 粒度保存：每个 clip 一个 .npy，支持 batch 推理与过滤。按视频遍历，每视频解码一次、npz 加载一次。"""
        nonlocal processed
        batch_frames: list[Any] = []
        batch_meta: list[dict[str, Any]] = []
        flows_dir_path = Path(args.flows_dir)
        npz_cache: dict[str, np.ndarray] = {}

        with meta_path.open("a", encoding="utf-8") as meta_f:
            def flush() -> None:
                """将当前 batch 送 ViT 推理并保存 embedding 与 meta"""
                nonlocal batch_frames, batch_meta, processed
                if not batch_frames:
                    return
                batch_flows = None
                if getattr(enc.cfg, "use_dual_stream", False):
                    batch_flows = []
                    for m in batch_meta:
                        vid = m["video_id"]
                        if vid not in npz_cache:
                            npz_path = flows_dir_path / safe_id(vid) / "flows.npz"
                            if npz_path.exists():
                                npz = np.load(npz_path, allow_pickle=True)
                                npz_cache[vid] = np.asarray(npz["flows"], dtype=np.float32)
                            else:
                                npz_cache[vid] = None
                        flows = npz_cache[vid]
                        if flows is not None:
                            ci = int(m["clip_index"])
                            if ci < flows.shape[0]:
                                batch_flows.append(flows[ci])
                            else:
                                batch_flows.append(flows[0])
                        else:
                            batch_flows.append(np.zeros((args.clip_len - 1, 2, 224, 224), dtype=np.float32))
                with torch.no_grad():
                    emb = enc(batch_frames, batch_flows) if batch_flows is not None else enc(batch_frames)
                emb_np = emb.cpu().numpy()

                for i, m in enumerate(batch_meta):
                    vid_safe = safe_id(m["video_id"])
                    save_dir = out_dir / vid_safe
                    save_dir.mkdir(parents=True, exist_ok=True)
                    emb_path = save_dir / f"clip_{m['clip_index']:03d}.npy"
                    if args.skip_existing and emb_path.exists():
                        processed += 1
                        continue
                    np.save(emb_path, emb_np[i].astype(np.float32))
                    rec = dict(m)
                    rec["embedding_path"] = str(emb_path.as_posix())
                    rec["model_name"] = args.model_name
                    rec["pooling"] = str(args.pooling)
                    _append_meta_line(meta_f, rec)
                    processed += 1

                batch_frames = []
                batch_meta = []
                print(f"  processed {processed}/{total_clips} clips", flush=True)

            flows_dir = Path(args.flows_dir)
            n_videos = len(ds.rows)
            import time as _time
            _t_start = _time.time()
            for vi in range(n_videos):
                if args.limit and processed >= args.limit:
                    break
                _t_vi = _time.time()
                print(f"  video {vi + 1}/{n_videos}: {ds.rows[vi].video_id}", end="", flush=True)
                clips = ds.get_video_clips(vi)
                _t_read = _time.time() - _t_vi
                if not clips:
                    print(f"  (no clips, skip)", flush=True)
                    continue
                flows_arr = None
                if args.enable_filtering and args.enable_flow_filter:
                    npz_path = flows_dir / safe_id(clips[0].video_id) / "flows.npz"
                    if npz_path.exists():
                        fnpz = np.load(npz_path, allow_pickle=True)
                        flows_arr = np.asarray(fnpz["flows"], dtype=np.float32)
                _n_filtered = 0
                for sample in clips:
                    if args.enable_filtering:
                        ssim_val = compute_ssim_variation(sample.frames)
                        if ssim_val > getattr(args, "ssim_threshold", 0.95):
                            _n_filtered += 1
                            continue
                        if args.enable_flow_filter:
                            flow_mag = compute_optical_flow_magnitude(
                                sample.frames,
                                flows_npz_path=flows_dir / safe_id(sample.video_id) / "flows.npz" if flows_arr is None else None,
                                clip_index=sample.clip_index,
                                flows_array=flows_arr,
                            )
                            if flow_mag < getattr(args, "flow_threshold", 2.0):
                                _n_filtered += 1
                                continue
                    batch_frames.append(sample.frames)
                    batch_meta.append(
                        {
                            "video_id": sample.video_id,
                            "video_path": sample.video_path,
                            "label": sample.label,
                            "clip_index": sample.clip_index,
                            "start_frame": sample.start_frame,
                            "end_frame_exclusive": sample.end_frame_exclusive,
                            "save_format": "npy_per_clip",
                        }
                    )
                    if len(batch_frames) >= args.batch_size:
                        flush()
                _t_total_vi = _time.time() - _t_vi
                _filter_info = f" filtered={_n_filtered}" if _n_filtered else ""
                print(f"  read={_t_read:.1f}s total={_t_total_vi:.1f}s clips={len(clips)}{_filter_info}", flush=True)
            flush()
            _t_all = _time.time() - _t_start
            print(f"  All videos done in {_t_all:.1f}s")

    def _save_npz_per_video() -> None:
        """按视频粒度保存：每个视频一个 .npz，内含该视频全部 clip 的 embeddings"""
        nonlocal processed
        total_videos = len(ds.rows)
        print(f"Total videos to process: {total_videos}. First video may take 1–2 min (read + ViT).")
        with meta_path.open("a", encoding="utf-8") as meta_f:
            for vi, row in enumerate(ds.rows):
                if args.limit and processed >= args.limit:
                    break
                vid_safe = safe_id(row.video_id)
                npz_path = out_dir / f"{vid_safe}.npz"
                if args.skip_existing and npz_path.exists():
                    # 仍然计数：按 num_clips_per_video 估算（不精确，但用于 limit/进度足够）
                    processed += int(args.num_clips_per_video)
                    continue

                print(f"  video {vi + 1}/{total_videos}: {row.video_id}", flush=True)
                clips = ds.get_video_clips(vi)
                clip_frames = [c.frames for c in clips]
                clip_flows = None
                if getattr(enc.cfg, "use_dual_stream", False):
                    flows_npz_path = Path(args.flows_dir) / vid_safe / "flows.npz"
                    if flows_npz_path.exists():
                        fnpz = np.load(flows_npz_path, allow_pickle=True)
                        flows_arr = np.asarray(fnpz["flows"], dtype=np.float32)
                        clip_flows = [flows_arr[i] for i in range(min(flows_arr.shape[0], len(clip_frames)))]
                        if len(clip_flows) < len(clip_frames):
                            clip_flows.extend([flows_arr[0]] * (len(clip_frames) - len(clip_flows)))
                with torch.no_grad():
                    emb = enc(clip_frames, clip_flows) if clip_flows else enc(clip_frames)
                emb_np = emb.cpu().numpy().astype(np.float32)

                clip_index = np.array([c.clip_index for c in clips], dtype=np.int32)
                start_frame = np.array([c.start_frame for c in clips], dtype=np.int32)
                end_frame_exclusive = np.array([c.end_frame_exclusive for c in clips], dtype=np.int32)

                np.savez_compressed(
                    npz_path,
                    embeddings=emb_np,
                    clip_index=clip_index,
                    start_frame=start_frame,
                    end_frame_exclusive=end_frame_exclusive,
                    video_id=np.array([row.video_id], dtype=object),
                    video_path=np.array([row.video_path], dtype=object),
                    label=np.array([row.label], dtype=object),
                    model_name=np.array([args.model_name], dtype=object),
                    pooling=np.array([str(args.pooling)], dtype=object),
                )
                _append_meta_line(
                    meta_f,
                    {
                        "video_id": row.video_id,
                        "video_path": row.video_path,
                        "label": row.label,
                        "embedding_path": str(npz_path.as_posix()),
                        "num_clips": int(len(clips)),
                        "model_name": args.model_name,
                        "pooling": str(args.pooling),
                        "save_format": "npz_per_video",
                    },
                )
                processed += int(len(clips))

    if args.save_format == "npy_per_clip":
        _save_npy_per_clip()
    else:
        _save_npz_per_video()
    print(f"[OK] embeddings saved to: {out_dir}")
    print(f"[OK] meta jsonl: {meta_path}")


if __name__ == "__main__":
    main()

