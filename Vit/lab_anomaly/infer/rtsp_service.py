from __future__ import annotations

"""
RTSP 实时推理服务（实验室异常行为识别）：
  抽帧/滑窗 → VideoMAE v2 编码 → MIL 二分类 + ranking 分数 → 报警输出（日志/截图/API）。

特点：
  - 支持 `yaml/json` 配置（见 `lab_anomaly/configs/rtsp_service_example.yaml`）
  - 抽帧：按 `sample_fps` 从 RTSP 降采样
  - 缓冲区存最近 `clip_len` 帧，推理时在缓冲区内均匀取 `clip_len` 帧送编码器（当前默认 16 帧，与训练一致）
  - 去抖：`min_consecutive` + `cooldown_sec`
  - 输出：`events.jsonl`、可选截图、可选 HTTP POST(JSON)

依赖：
  - opencv-python, numpy, torch, transformers（VideoMAE v2）
  - PyYAML（如用 yaml 配置）
"""

import argparse
import json
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lab_anomaly.infer.scoring import load_known_classifier, predict_fusion


def _uniform_positions(buf_len: int, num_frames: int) -> list[int]:
    """在缓冲区下标 [0, buf_len-1] 上均匀取 num_frames 个位置。"""
    if buf_len <= 0 or num_frames <= 0:
        return []
    if num_frames == 1:
        return [0]
    hi = buf_len - 1
    return [min(max(0, int(round(i * hi / (num_frames - 1)))), hi) for i in range(num_frames)]


def _now_ts() -> float:
    """当前 Unix 时间戳"""
    return time.time()


def _now_iso() -> str:
    """当前时间 ISO 格式"""
    return datetime.now().isoformat(timespec="seconds")


def _now_str_ms() -> str:
    """当前时间字符串（含毫秒），用于截图文件名"""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _try_load_yaml_or_json(path: Path) -> dict[str, Any]:
    """加载 yaml 或 json 配置文件"""
    txt = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json"}:
        return json.loads(txt)
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("缺少 PyYAML 依赖。请安装：pip install PyYAML 或改用 json 配置。") from e
        return yaml.safe_load(txt) or {}
    # fallback：尝试 json
    try:
        return json.loads(txt)
    except Exception:
        raise ValueError(f"unknown config format: {path} (expected .json/.yaml/.yml)")


def _cfg_get(cfg: dict[str, Any], path: str, default: Any) -> Any:
    """按点分隔路径从嵌套 dict 取值"""
    cur: Any = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _post_json(url: str, payload: dict[str, Any], timeout_sec: float) -> tuple[bool, str]:
    """
    尽量做到“零额外依赖”的 HTTP POST(JSON)。
    返回：(ok, error_message)
    """
    url = str(url).strip()
    if not url:
        return True, ""
    try:
        import urllib.request

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=float(timeout_sec)) as resp:
            _ = resp.read()
        return True, ""
    except Exception as e:  # pragma: no cover
        return False, str(e)


def _open_capture(rtsp_url: str, open_timeout_sec: float) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(rtsp_url)
    t0 = _now_ts()
    while not cap.isOpened():
        if _now_ts() - t0 > float(open_timeout_sec):
            break
        time.sleep(0.2)
    return cap


def _open_source(cfg: ServiceConfig) -> cv2.VideoCapture:
    """根据配置打开视频源：video_path 非空则打开本地视频，否则打开 RTSP。"""
    if str(cfg.video_path).strip():
        path = Path(cfg.video_path)
        # 若路径不存在且无扩展名，尝试常见视频扩展名
        to_try: list[str] = [cfg.video_path]
        if not path.exists():
            if not path.suffix:
                for ext in (".mp4", ".avi", ".mkv", ".mov", ".webm"):
                    to_try.append(cfg.video_path.rstrip("/\\") + ext)
        for p in to_try:
            cap = cv2.VideoCapture(p)
            if cap.isOpened():
                return cap
            cap.release()
        msg = f"无法打开视频：{cfg.video_path}"
        if not path.exists():
            msg += "（路径不存在）"
        if not path.suffix:
            msg += "；若为视频文件请加上扩展名（如 .mp4）"
        raise RuntimeError(msg)
    return _open_capture(cfg.rtsp_url, open_timeout_sec=float(cfg.open_timeout_sec))


@dataclass
class DebounceState:
    consecutive: int = 0
    last_fire_ts: float = 0.0


def _should_fire(state: DebounceState, triggered: bool, *, min_consecutive: int, cooldown_sec: float) -> bool:
    """去抖：需连续 min_consecutive 次触发且距上次 fire 超过 cooldown_sec 才返回 True"""
    if triggered:
        state.consecutive += 1
    else:
        state.consecutive = 0
    if state.consecutive < int(min_consecutive):
        return False
    now = _now_ts()
    if (now - state.last_fire_ts) < float(cooldown_sec):
        return False
    state.last_fire_ts = now
    state.consecutive = 0
    return True


@dataclass(frozen=True)
class ServiceConfig:
    video_path: str  # 非空则从视频文件读取，否则用 rtsp_url
    rtsp_url: str
    loop_video: bool  # 仅视频文件：播完后是否循环
    camera_id: str
    out_dir: Path

    known_checkpoint: str
    open_set_dir: str

    encoder_model_name: str
    device: str
    use_half: bool
    use_dual_stream: bool
    inference_use_rgb_only: bool

    clip_len: int
    frame_stride: int
    sample_fps: float
    window_stride: int

    min_known_prob: float
    known_alarm_prob: float
    treat_low_conf_as_unknown: bool
    unknown_alarm: bool

    min_consecutive: int
    cooldown_sec: float
    open_timeout_sec: float
    reconnect_wait_sec: float
    show: bool

    save_snapshot: bool
    api_url: str
    api_timeout_sec: float
    ranking_alarm_threshold: float = 0.5  # MIL ranking 异常分数报警阈值


def _build_config(args: argparse.Namespace) -> ServiceConfig:
    """从 argparse 与可选配置文件构建 ServiceConfig"""
    cfg: dict[str, Any] = {}
    if str(args.config).strip():
        cfg = _try_load_yaml_or_json(Path(args.config))

    video_path = str(getattr(args, "video_path", "") or "").strip() or str(_cfg_get(cfg, "source.video_path", "")).strip()
    rtsp_url = str(args.rtsp).strip() or str(_cfg_get(cfg, "rtsp_url", "")).strip()
    loop_video = bool(getattr(args, "loop_video", False)) or bool(_cfg_get(cfg, "runtime.loop_video", False))

    if not video_path and not rtsp_url:
        raise ValueError("请设置 video_path 或 rtsp_url（二选一：视频文件或 RTSP 流）。")
    if video_path:
        rtsp_url = rtsp_url or ""  # 使用视频时 rtsp_url 可空
    else:
        if not rtsp_url:
            raise ValueError("使用 RTSP 时请设置 rtsp_url。")

    camera_id = str(args.camera_id).strip() or str(_cfg_get(cfg, "camera_id", "cam0")).strip()
    out_dir = Path(str(args.out_dir).strip() or str(_cfg_get(cfg, "out_dir", "lab_dataset/derived/realtime")).strip())

    known_checkpoint = str(args.known_checkpoint).strip() or str(
        _cfg_get(cfg, "artifacts.known_checkpoint", "")
    ).strip()
    open_set_dir = str(args.open_set_dir).strip() or str(_cfg_get(cfg, "artifacts.open_set_dir", "")).strip()

    device = str(args.device).strip() or str(_cfg_get(cfg, "encoder.device", "auto")).strip()
    if not device or device.lower() == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder_model_name = str(args.model_name).strip() or str(
        _cfg_get(cfg, "encoder.model_name", "OpenGVLab/VideoMAEv2-Base")
    ).strip()
    use_half = bool(args.use_half) if args.use_half else bool(_cfg_get(cfg, "encoder.use_half", False))
    use_dual_stream = bool(_cfg_get(cfg, "encoder.use_dual_stream", False))
    inference_use_rgb_only = bool(_cfg_get(cfg, "runtime.inference_use_rgb_only", False))

    clip_len = int(args.clip_len) if args.clip_len else int(_cfg_get(cfg, "sampling.clip_len", 16))
    frame_stride = int(args.frame_stride) if args.frame_stride else int(_cfg_get(cfg, "sampling.frame_stride", 2))
    sample_fps = float(args.sample_fps) if args.sample_fps is not None else float(_cfg_get(cfg, "sampling.sample_fps", 5.0))
    window_stride = int(args.window_stride) if args.window_stride else int(_cfg_get(cfg, "sampling.window_stride", 4))

    min_known_prob = float(args.min_known_prob) if args.min_known_prob is not None else float(_cfg_get(cfg, "fusion.min_known_prob", 0.5))
    known_alarm_prob = float(args.known_alarm_prob) if args.known_alarm_prob is not None else float(_cfg_get(cfg, "fusion.known_alarm_prob", min_known_prob))
    treat_low_conf_as_unknown = bool(args.treat_low_conf_as_unknown) if args.treat_low_conf_as_unknown else bool(
        _cfg_get(cfg, "fusion.treat_low_conf_as_unknown", True)
    )
    unknown_alarm = bool(args.unknown_alarm) if args.unknown_alarm else bool(_cfg_get(cfg, "fusion.unknown_alarm", True))

    min_consecutive = int(args.min_consecutive) if args.min_consecutive else int(_cfg_get(cfg, "runtime.min_consecutive", 1))
    cooldown_sec = float(args.cooldown_sec) if args.cooldown_sec is not None else float(_cfg_get(cfg, "runtime.cooldown_sec", 3.0))
    open_timeout_sec = float(args.open_timeout_sec) if args.open_timeout_sec is not None else float(_cfg_get(cfg, "runtime.open_timeout_sec", 10.0))
    reconnect_wait_sec = float(args.reconnect_wait_sec) if args.reconnect_wait_sec is not None else float(_cfg_get(cfg, "runtime.reconnect_wait_sec", 3.0))
    show = bool(args.show) if args.show else bool(_cfg_get(cfg, "runtime.show", False))

    save_snapshot = bool(args.save_snapshot) if args.save_snapshot else bool(_cfg_get(cfg, "outputs.save_snapshot", True))
    api_url = str(args.api_url).strip() or str(_cfg_get(cfg, "outputs.api_url", "")).strip()
    api_timeout_sec = float(args.api_timeout_sec) if args.api_timeout_sec is not None else float(_cfg_get(cfg, "outputs.api_timeout_sec", 3.0))
    ranking_alarm_threshold = float(args.ranking_alarm_threshold) if args.ranking_alarm_threshold is not None else float(_cfg_get(cfg, "fusion.ranking_alarm_threshold", 0.5))

    return ServiceConfig(
        video_path=video_path,
        rtsp_url=rtsp_url,
        loop_video=loop_video,
        camera_id=camera_id,
        out_dir=out_dir,
        known_checkpoint=known_checkpoint,
        open_set_dir=open_set_dir,
        encoder_model_name=encoder_model_name,
        device=device,
        use_half=use_half,
        use_dual_stream=use_dual_stream,
        inference_use_rgb_only=inference_use_rgb_only,
        clip_len=int(clip_len),
        frame_stride=max(1, int(frame_stride)),
        sample_fps=float(sample_fps),
        window_stride=max(1, int(window_stride)),
        min_known_prob=float(min_known_prob),
        known_alarm_prob=float(known_alarm_prob),
        treat_low_conf_as_unknown=bool(treat_low_conf_as_unknown),
        unknown_alarm=bool(unknown_alarm),
        min_consecutive=max(1, int(min_consecutive)),
        cooldown_sec=float(cooldown_sec),
        open_timeout_sec=float(open_timeout_sec),
        reconnect_wait_sec=float(reconnect_wait_sec),
        show=bool(show),
        save_snapshot=bool(save_snapshot),
        api_url=api_url,
        api_timeout_sec=float(api_timeout_sec),
        ranking_alarm_threshold=float(ranking_alarm_threshold),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="RTSP 实时异常推理：VideoMAE v2 + MIL 二分类")
    ap.add_argument("--config", type=str, default="", help="yaml/json 配置文件（可选）")

    ap.add_argument("--video", "--video_path", dest="video_path", type=str, default="", help="本地视频路径；非空则从视频读取，否则用 --rtsp")
    ap.add_argument("--rtsp", type=str, default="", help="RTSP URL；仅当未设 --video 时使用")
    ap.add_argument("--loop_video", action="store_true", help="视频文件播完后是否循环（仅 --video 时有效）")
    ap.add_argument("--camera_id", type=str, default="cam0")
    ap.add_argument("--out_dir", type=str, default="lab_dataset/derived/realtime")

    ap.add_argument("--known_checkpoint", type=str, default="", help="已知异常分类器 checkpoint_best.pt（必填）")
    ap.add_argument("--open_set_dir", type=str, default="", help="（当前未使用，保留兼容）")

    ap.add_argument("--model_name", type=str, default="", help="encoder model name（HF）")
    ap.add_argument("--device", type=str, default="", help="cuda/cpu/auto（默认 auto）")
    ap.add_argument("--use_half", action="store_true", help="在 CUDA 上启用 FP16（部分环境可能不兼容）")

    ap.add_argument("--clip_len", type=int, default=0)
    ap.add_argument("--frame_stride", type=int, default=0)
    ap.add_argument("--sample_fps", type=float, default=None)
    ap.add_argument("--window_stride", type=int, default=0)

    ap.add_argument("--min_known_prob", type=float, default=None)
    ap.add_argument("--known_alarm_prob", type=float, default=None, help="已知异常报警阈值（默认等于 min_known_prob）")
    ap.add_argument("--treat_low_conf_as_unknown", action="store_true")
    ap.add_argument("--unknown_alarm", action="store_true", help="开启未知异常报警（默认按配置；无配置时默认开）")
    ap.add_argument("--ranking_alarm_threshold", type=float, default=None, help="MIL ranking 异常分数报警阈值（0~1，默认 0.5）")

    ap.add_argument("--min_consecutive", type=int, default=0)
    ap.add_argument("--cooldown_sec", type=float, default=None)
    ap.add_argument("--open_timeout_sec", type=float, default=None)
    ap.add_argument("--reconnect_wait_sec", type=float, default=None)
    ap.add_argument("--show", action="store_true")

    ap.add_argument("--save_snapshot", action="store_true")
    ap.add_argument("--api_url", type=str, default="")
    ap.add_argument("--api_timeout_sec", type=float, default=None)

    args = ap.parse_args()

    # 未传 --config 时用文件末尾 BUILDIN_DEFAULTS 兜底（便于 PyCharm 直接运行）
    if not str(args.config).strip():
        for key, default_val in BUILDIN_DEFAULTS.items():
            if not hasattr(args, key):
                continue
            val = getattr(args, key)
            if val is None or val == "" or val == 0:
                setattr(args, key, default_val)

    cfg = _build_config(args)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    events_path = cfg.out_dir / "events.jsonl"

    init_info: dict[str, Any] = {
        "source": "video" if str(cfg.video_path).strip() else "rtsp",
        "camera_id": cfg.camera_id,
        "device": cfg.device,
        "encoder": cfg.encoder_model_name,
        "clip_len": cfg.clip_len,
        "sample_fps": cfg.sample_fps,
        "window_stride": cfg.window_stride,
        "known_checkpoint": cfg.known_checkpoint,
    }
    if str(cfg.video_path).strip():
        init_info["video_path"] = cfg.video_path
        init_info["loop_video"] = cfg.loop_video
    else:
        init_info["rtsp_url"] = cfg.rtsp_url
    print("[INIT]", json.dumps(init_info, ensure_ascii=False))

    if not str(cfg.known_checkpoint).strip():
        raise ValueError("请提供 known_checkpoint（端到端 checkpoint_best.pt）。")
    bundle = load_known_classifier(cfg.known_checkpoint, device=cfg.device)
    enc = bundle.encoder
    if enc is None:
        raise RuntimeError("checkpoint 中缺少 encoder_state，请使用 train_end2end 保存的权重。")
    enc.eval()
    known = bundle

    known_normal_label = "normal"
    if known is not None:
        known_normal_label = "normal" if "normal" in known.label2idx else str(known.idx2label.get(0, "normal"))

    normal_cls_idx = int(known.label2idx.get("normal", 0))

    # 缓冲区：最近 clip_len 个采样帧（RGB）
    clip_total = int(cfg.clip_len)
    if clip_total <= 0:
        raise ValueError("clip_len 须为正整数。")
    buf_rgb: deque[np.ndarray] = deque(maxlen=int(clip_total))
    last_bgr: Optional[np.ndarray] = None

    push_count = 0
    last_infer_push_count = 0
    debounce = DebounceState()

    use_video_file = bool(str(cfg.video_path).strip())

    try:
        while True:
            try:
                cap = _open_source(cfg)
            except RuntimeError as e:
                if use_video_file:
                    raise
                print(f"[WARN] 无法打开 RTSP：{e}；{cfg.reconnect_wait_sec}s 后重试。")
                time.sleep(float(cfg.reconnect_wait_sec))
                continue
            if not cap.isOpened():
                if use_video_file:
                    raise RuntimeError(f"无法打开视频文件：{cfg.video_path}")
                print(f"[WARN] 无法打开 RTSP；{cfg.reconnect_wait_sec}s 后重试。")
                time.sleep(float(cfg.reconnect_wait_sec))
                continue

            src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if src_fps <= 1e-3:
                src_fps = 25.0

            # 抽帧 interval：每隔 interval 帧取一帧
            if float(cfg.sample_fps) and float(cfg.sample_fps) > 1e-3:
                interval = max(1, int(round(src_fps / float(cfg.sample_fps))))
            else:
                interval = 1

            frame_i = 0
            buf_rgb.clear()
            push_count = 0
            last_infer_push_count = 0
            src_label = "video" if use_video_file else "rtsp"
            print(f"[OK] 已连接（{src_label}）。src_fps={src_fps:.2f} sample_interval={interval} clip_total_frames={clip_total}")

            while True:
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    if use_video_file:
                        if cfg.loop_video:
                            cap.release()
                            cap = cv2.VideoCapture(cfg.video_path)
                            if not cap.isOpened():
                                print(f"[WARN] 视频循环时无法重新打开：{cfg.video_path}")
                                break
                            frame_i = 0
                            buf_rgb.clear()
                            push_count = 0
                            last_infer_push_count = 0
                            continue
                        print("[INFO] 视频播放结束。")
                    else:
                        print("[WARN] 读取帧失败，准备重连。")
                    break

                frame_i += 1
                last_bgr = frame_bgr

                if (frame_i - 1) % int(interval) != 0:
                    if cfg.show:
                        cv2.imshow(f"rtsp_service:{cfg.camera_id}", frame_bgr)
                        if cv2.waitKey(1) & 0xFF == 27:
                            return
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                buf_rgb.append(frame_rgb)
                push_count += 1

                # 不够组成 clip
                if len(buf_rgb) < int(clip_total):
                    if cfg.show:
                        cv2.imshow(f"rtsp_service:{cfg.camera_id}", frame_bgr)
                        if cv2.waitKey(1) & 0xFF == 27:
                            return
                    continue

                # 每 window_stride 个新采样帧做一次推理
                if (push_count - last_infer_push_count) < int(cfg.window_stride):
                    if cfg.show:
                        cv2.imshow(f"rtsp_service:{cfg.camera_id}", frame_bgr)
                        if cv2.waitKey(1) & 0xFF == 27:
                            return
                    continue
                last_infer_push_count = push_count

                buf_list = list(buf_rgb)
                pos = _uniform_positions(len(buf_list), int(cfg.clip_len))
                clip_frames = [buf_list[i].copy() for i in pos]

                with torch.no_grad():
                    emb_t = enc([clip_frames])
                emb_np = emb_t.detach().cpu().numpy().astype(np.float32).reshape(-1)

                probs: Optional[np.ndarray] = None
                pred_label = "unknown"
                pred_prob = 0.0
                ranking_score = 0.0

                x = torch.from_numpy(emb_np).view(1, 1, -1).to(known.device)
                mask = torch.ones((1, 1), device=known.device, dtype=torch.bool)
                with torch.no_grad():
                    logits, details = known.model(x, mask=mask, y=None, return_details=True)
                    probs = F.softmax(logits, dim=-1).detach().cpu().numpy().reshape(-1)
                    if "anomaly_scores" in details:
                        ranking_score = float(details["anomaly_scores"].max().item())

                idx2label = known.idx2label
                fused = predict_fusion(
                    known_probs=probs,
                    idx2label=idx2label,
                    ranking_anomaly_score=ranking_score,
                    ranking_alarm_threshold=float(cfg.ranking_alarm_threshold),
                    min_known_prob=float(cfg.min_known_prob),
                    treat_low_conf_as_unknown=bool(cfg.treat_low_conf_as_unknown),
                    normal_idx=normal_cls_idx,
                )

                pred_label = str(fused.predicted_label)
                pred_prob = float(fused.predicted_prob)

                known_trigger = (
                    pred_label not in {known_normal_label, "unknown"}
                    and pred_prob >= float(cfg.known_alarm_prob)
                )
                unknown_trigger = bool(cfg.unknown_alarm) and bool(fused.is_anomaly)
                triggered = bool(known_trigger or unknown_trigger)

                if cfg.show:
                    rank_txt = f" rank={fused.ranking_anomaly_score:.2f}" if fused.ranking_anomaly_score > 0 else ""
                    text = f"{pred_label} p={pred_prob:.2f} rank={ranking_score:.2f}{rank_txt}"
                    cv2.putText(frame_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow(f"rtsp_service:{cfg.camera_id}", frame_bgr)
                    if cv2.waitKey(1) & 0xFF == 27:
                        return

                if not _should_fire(debounce, triggered, min_consecutive=int(cfg.min_consecutive), cooldown_sec=float(cfg.cooldown_sec)):
                    continue

                event_type = "known" if known_trigger else "unknown"
                t = _now_ts()
                event_source = "video" if use_video_file else "rtsp"
                event: dict[str, Any] = {
                    "ts_unix": t,
                    "ts": _now_iso(),
                    "source": event_source,
                    "camera_id": cfg.camera_id,
                    "event_type": event_type,
                    "predicted_label": pred_label,
                    "predicted_prob": pred_prob,
                    "fusion": {
                        "predicted_label": fused.predicted_label,
                        "predicted_prob": fused.predicted_prob,
                        "is_anomaly": fused.is_anomaly,
                        "ranking_anomaly_score": fused.ranking_anomaly_score,
                    },
                }
                if use_video_file:
                    event["video_path"] = cfg.video_path
                else:
                    event["rtsp_url"] = cfg.rtsp_url

                snapshot_path = ""
                if cfg.save_snapshot and last_bgr is not None:
                    snap_dir = cfg.out_dir / "snapshots" / cfg.camera_id
                    snap_dir.mkdir(parents=True, exist_ok=True)
                    snapshot_path = str((snap_dir / f"{_now_str_ms()}_{event_type}_{pred_label}.jpg").as_posix())
                    cv2.imwrite(snapshot_path, last_bgr)
                    event["snapshot_path"] = snapshot_path

                with events_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")

                print("[ALARM]", json.dumps(event, ensure_ascii=False))

                ok_post, err = _post_json(cfg.api_url, payload=event, timeout_sec=float(cfg.api_timeout_sec))
                if not ok_post:
                    print(f"[WARN] API 推送失败：{err}")

            cap.release()
            if use_video_file and not cfg.loop_video:
                break
            time.sleep(float(cfg.reconnect_wait_sec))
    except KeyboardInterrupt:
        print("[EXIT] KeyboardInterrupt")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


# ============ 可调参数与路径（直接修改此处即可，无需命令行）============
#
# 使用说明：下面 BUILDIN_DEFAULTS 中的值会在「未传命令行参数且未用 --config 指定 yaml」时生效；
# 若传了 --config 或某个命令行参数，则以配置文件/命令行为准。调节时只需改下面字典里的值即可。
#
# --- 输入源（二选一）---
#   video_path   ：本地视频文件路径。填非空则从视频文件读帧做预测；留空 "" 则使用 rtsp。
#                  调节：用视频测试时填如 "D:/data/test.mp4"；用摄像头/RTSP 时留空并填好 rtsp。
#   rtsp         ：RTSP 流地址或摄像头 URL。仅当 video_path 为空时使用，此时必填。
#                  调节：换成你的 rtsp 地址或 0/1 等摄像头索引。
#   loop_video   ：仅当 video_path 非空时有效。True=视频播完后自动重播，False=播完退出。
#                  调节：需要反复看同一段视频时改为 True。
#
# --- 模型与权重路径 ---
#   known_checkpoint ：已知异常分类器权重路径（.pt），必填。预测仅使用已知分类 + MIL ranking。
#   open_set_dir     ：（当前未使用）保留兼容，可留空。
#   out_dir          ：输出目录，events.jsonl、截图等会写在这里。
#
# --- 编码器与设备 ---
#   model_name ：应与训练时一致，如 OpenGVLab/VideoMAEv2-Base（权重内已存配置时可忽略）。
#   device     ："auto" / "cuda" / "cpu"。auto 表示有 GPU 则用 cuda。
#   use_half   ：是否用 FP16（仅 CUDA 时有效），部分环境可能不稳定。
#
# --- 抽帧与滑窗（与训练时保持一致效果更好）---
#   clip_len     ：编码器输入帧数（当前默认 16），缓冲区保持最近 clip_len 个采样帧。
#   sample_fps   ：从视频中抽帧的目标帧率，例如 5 表示每秒取约 5 帧入缓冲。
#   window_stride：每隔多少「个采样帧」做一次推理；越大推理越稀疏，延迟越高。
#                  调节：想更实时可减小 window_stride；想省算力可增大。
#
# --- 融合与报警阈值 ---
#   min_known_prob / known_alarm_prob ：已知分类置信度阈值，高于此且非 normal 才按已知异常报警。
#   treat_low_conf_as_unknown         ：True 表示已知分类置信度低时视为 unknown。
#   unknown_alarm                     ：是否对预测为 unknown（低置信度）也触发报警。
#   ranking_alarm_threshold           ：MIL 异常分支分数阈值（0~1），超过视为异常参与融合。
#                  调节：误报多可适当提高阈值；漏报多可适当降低。
#
# --- 去抖与运行时 ---
#   min_consecutive ：连续多少次判定为异常才真正触发一次报警（防抖）。
#   cooldown_sec    ：触发一次报警后，多少秒内不再触发（冷却）。
#   open_timeout_sec：打开 RTSP 时等待超时（秒）。
#   reconnect_wait_sec ：RTSP 断线后重连前等待秒数。
#
# --- 输出与界面 ---
#   show          ：是否弹窗显示当前帧和预测结果。
#   save_snapshot  ：是否在报警时保存截图到 out_dir。
#   api_url        ：报警时 POST JSON 的 URL，留空则不推送。
#   api_timeout_sec：POST 超时秒数。
#
BUILDIN_DEFAULTS = {
    "config": "",
    "video_path": r"C:\Users\Administrator\Desktop\Vit\lab_dataset\raw_videos\steal\Stealing\Stealing008_x264",
    "rtsp": "",
    "loop_video": False,
    "camera_id": "cam0",
    "out_dir": "lab_dataset/derived/realtime",
    "known_checkpoint": r"C:\Users\Administrator\Desktop\Vit\lab_dataset\derived\end2end_classifier\checkpoint_best.pt",
    "model_name": "OpenGVLab/VideoMAEv2-Base",
    "device": "auto",
    "clip_len": 16,
    "sample_fps": 5.0,
    "window_stride": 4,
    "min_known_prob": 0.5,
    "known_alarm_prob": 0.5,
    "treat_low_conf_as_unknown": True,
    "unknown_alarm": True,
    "ranking_alarm_threshold": 0.5,
    "min_consecutive": 1,
    "cooldown_sec": 3.0,
    "open_timeout_sec": 10.0,
    "reconnect_wait_sec": 3.0,
    "show": False,
    "save_snapshot": True,
    "api_url": "",
    "api_timeout_sec": 3.0,
    "use_half": False,
}

if __name__ == "__main__":
    main()

