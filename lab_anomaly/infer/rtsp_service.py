from __future__ import annotations

"""
RTSP 实时推理服务（实验室异常行为识别）：
  抽帧/滑窗 → 视频 encoder embedding → 已知异常分类 + 开放集异常分数 → 报警输出（日志/截图/API）。

特点：
  - 支持 `yaml/json` 配置（见 `lab_anomaly/configs/rtsp_service_example.yaml`）
  - 抽帧：按 `sample_fps` 从 RTSP 降采样
  - 滑窗 clip：按 `clip_len` + `frame_stride` 组帧（即 clip 覆盖帧数 = (clip_len-1)*frame_stride+1）
  - 去抖：`min_consecutive` + `cooldown_sec`
  - 输出：`events.jsonl`、可选截图、可选 HTTP POST(JSON)

依赖：
  - opencv-python, numpy, torch
  - transformers（HfVideoEncoder）
  - scikit-learn + joblib（OpenSetScorer）
  - PyYAML（如用 yaml 配置）
"""

import argparse
import json
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lab_anomaly.infer.scoring import OpenSetScore, OpenSetScorer, fuse_known_and_open_set, load_known_classifier
from lab_anomaly.models.vit_video_encoder import HfVideoEncoder, HfVideoEncoderConfig


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


def _compute_flow_clip_farneback(clip_frames: list[np.ndarray]) -> np.ndarray:
    """对 clip 内连续帧用 Farneback 计算光流。返回 (T-1, 2, H, W) float32。"""
    if len(clip_frames) < 2:
        return np.zeros((0, 2, 224, 224), dtype=np.float32)
    flows = []
    for i in range(len(clip_frames) - 1):
        g0 = cv2.cvtColor(clip_frames[i], cv2.COLOR_RGB2GRAY)
        g1 = cv2.cvtColor(clip_frames[i + 1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            g0, g1, None, pyr_scale=0.5, levels=2, winsize=15,
            iterations=2, poly_n=5, poly_sigma=1.2, flags=0,
        )
        flow = np.asarray(flow, dtype=np.float32).transpose(2, 0, 1)
        flows.append(flow)
    return np.stack(flows, axis=0)


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
    rtsp_url: str
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


def _build_config(args: argparse.Namespace) -> ServiceConfig:
    """从 argparse 与可选配置文件构建 ServiceConfig"""
    cfg: dict[str, Any] = {}
    if str(args.config).strip():
        cfg = _try_load_yaml_or_json(Path(args.config))

    rtsp_url = str(args.rtsp).strip() or str(_cfg_get(cfg, "rtsp_url", "")).strip()
    if not rtsp_url:
        raise ValueError("缺少 RTSP URL：请传 --rtsp 或在配置里设置 rtsp_url。")

    camera_id = str(args.camera_id).strip() or str(_cfg_get(cfg, "camera_id", "cam0")).strip()
    out_dir = Path(str(args.out_dir).strip() or str(_cfg_get(cfg, "out_dir", "lab_dataset/derived/realtime")).strip())

    known_checkpoint = str(args.known_checkpoint).strip() or str(
        _cfg_get(cfg, "artifacts.known_checkpoint", "")
    ).strip()
    open_set_dir = str(args.open_set_dir).strip() or str(_cfg_get(cfg, "artifacts.open_set_dir", "")).strip()

    device = str(args.device).strip() or str(_cfg_get(cfg, "encoder.device", "auto")).strip()
    if not device or device.lower() == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder_model_name = str(args.model_name).strip() or str(_cfg_get(cfg, "encoder.model_name", "MCG-NJU/videomae-base")).strip()
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

    return ServiceConfig(
        rtsp_url=rtsp_url,
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
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="RTSP 实时异常推理：抽帧/滑窗 → embedding → 已知分类 + 开放集 → 报警")
    ap.add_argument("--config", type=str, default="", help="yaml/json 配置文件（可选）")

    ap.add_argument("--rtsp", type=str, default="", help="RTSP URL（或本地视频路径也可）")
    ap.add_argument("--camera_id", type=str, default="cam0")
    ap.add_argument("--out_dir", type=str, default="lab_dataset/derived/realtime")

    ap.add_argument("--known_checkpoint", type=str, default="", help="已知异常分类器 checkpoint_best.pt（可空：仅 open-set）")
    ap.add_argument("--open_set_dir", type=str, default="", help="开放集目录（fit_kmeans_ocsvm.py 输出，可空：仅 known）")

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

    ap.add_argument("--min_consecutive", type=int, default=0)
    ap.add_argument("--cooldown_sec", type=float, default=None)
    ap.add_argument("--open_timeout_sec", type=float, default=None)
    ap.add_argument("--reconnect_wait_sec", type=float, default=None)
    ap.add_argument("--show", action="store_true")

    ap.add_argument("--save_snapshot", action="store_true")
    ap.add_argument("--api_url", type=str, default="")
    ap.add_argument("--api_timeout_sec", type=float, default=None)

    args = ap.parse_args()
    cfg = _build_config(args)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    events_path = cfg.out_dir / "events.jsonl"

    print(
        "[INIT]",
        json.dumps(
            {
                "camera_id": cfg.camera_id,
                "device": cfg.device,
                "encoder": cfg.encoder_model_name,
                "clip_len": cfg.clip_len,
                "frame_stride": cfg.frame_stride,
                "sample_fps": cfg.sample_fps,
                "window_stride": cfg.window_stride,
                "known_checkpoint": cfg.known_checkpoint,
                "open_set_dir": cfg.open_set_dir,
            },
            ensure_ascii=False,
        ),
    )

    # load models
    enc_cfg = HfVideoEncoderConfig(
        model_name=cfg.encoder_model_name,
        use_half=cfg.use_half,
        use_dual_stream=cfg.use_dual_stream and not cfg.inference_use_rgb_only,
        fusion_method="concat",
    )
    enc = HfVideoEncoder(enc_cfg, device=cfg.device)
    enc.eval()

    known = None
    if str(cfg.known_checkpoint).strip():
        known = load_known_classifier(cfg.known_checkpoint, device=cfg.device)
    open_set = None
    if str(cfg.open_set_dir).strip():
        open_set = OpenSetScorer(cfg.open_set_dir)

    if known is None and open_set is None:
        raise ValueError("known_checkpoint 与 open_set_dir 至少需要提供一个（否则无法输出任何判定）。")

    known_normal_label = "normal"
    if known is not None:
        known_normal_label = "normal" if "normal" in known.label2idx else str(known.idx2label.get(0, "normal"))

    # buffer：存抽帧后的 RGB（frame_stride 在 clip 组装时再用）
    clip_total = (int(cfg.clip_len) - 1) * int(cfg.frame_stride) + 1
    if clip_total <= 0:
        raise ValueError("clip_len/frame_stride 非法。")
    buf_rgb: deque[np.ndarray] = deque(maxlen=int(clip_total))
    last_bgr: Optional[np.ndarray] = None

    push_count = 0
    last_infer_push_count = 0
    debounce = DebounceState()

    try:
        while True:
            cap = _open_capture(cfg.rtsp_url, open_timeout_sec=float(cfg.open_timeout_sec))
            if not cap.isOpened():
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
            print(f"[OK] 已连接。src_fps={src_fps:.2f} sample_interval={interval} clip_total_frames={clip_total}")

            while True:
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
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

                # 组装 clip：长度 clip_len，内部按 frame_stride 取帧
                clip_frames = [buf_rgb[i * int(cfg.frame_stride)] for i in range(int(cfg.clip_len))]

                with torch.no_grad():
                    if getattr(enc.cfg, "use_dual_stream", False):
                        clip_flows = [_compute_flow_clip_farneback(clip_frames)]
                        emb_t = enc([clip_frames], clip_flows)
                    else:
                        emb_t = enc([clip_frames])
                emb_np = emb_t.detach().cpu().numpy().astype(np.float32).reshape(-1)

                probs: Optional[np.ndarray] = None
                pred_label = "unknown"
                pred_prob = 0.0

                if known is not None:
                    x = torch.from_numpy(emb_np).view(1, 1, -1).to(known.device)
                    mask = torch.ones((1, 1), device=known.device, dtype=torch.bool)
                    with torch.no_grad():
                        logits, _ = known.model(x, mask=mask, y=None, return_details=False)
                        probs = F.softmax(logits, dim=-1).detach().cpu().numpy().reshape(-1)

                # open-set
                if open_set is not None:
                    os = open_set.score_embedding(emb_np)
                else:
                    os = OpenSetScore(cluster_id=-1, decision_score=0.0, anomaly_score=0.0, threshold=float("inf"), is_anomaly=False)

                if probs is None:
                    # 仅 open-set：把 known_probs 伪造为 [1.0]，label 映射为 unknown
                    probs = np.asarray([1.0], dtype=np.float32)
                    idx2label = {0: "unknown"}
                else:
                    idx2label = known.idx2label

                fused = fuse_known_and_open_set(
                    known_probs=probs,
                    idx2label=idx2label,
                    open_set=os,
                    min_known_prob=float(cfg.min_known_prob),
                    treat_low_conf_as_unknown=bool(cfg.treat_low_conf_as_unknown),
                )

                pred_label = str(fused.predicted_label)
                pred_prob = float(fused.predicted_prob)

                known_trigger = (
                    known is not None
                    and pred_label not in {known_normal_label, "unknown"}
                    and pred_prob >= float(cfg.known_alarm_prob)
                )
                unknown_trigger = bool(cfg.unknown_alarm) and bool(fused.is_anomaly)
                triggered = bool(known_trigger or unknown_trigger)

                if cfg.show:
                    text = f"{pred_label} p={pred_prob:.2f} anom={float(fused.anomaly_score):.3f}"
                    cv2.putText(frame_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow(f"rtsp_service:{cfg.camera_id}", frame_bgr)
                    if cv2.waitKey(1) & 0xFF == 27:
                        return

                if not _should_fire(debounce, triggered, min_consecutive=int(cfg.min_consecutive), cooldown_sec=float(cfg.cooldown_sec)):
                    continue

                event_type = "known" if known_trigger else "unknown"
                t = _now_ts()
                event: dict[str, Any] = {
                    "ts_unix": t,
                    "ts": _now_iso(),
                    "camera_id": cfg.camera_id,
                    "event_type": event_type,
                    "predicted_label": pred_label,
                    "predicted_prob": pred_prob,
                    "open_set": asdict(os),
                    "fusion": asdict(fused),
                    "rtsp_url": cfg.rtsp_url,
                }

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
            time.sleep(float(cfg.reconnect_wait_sec))
    except KeyboardInterrupt:
        print("[EXIT] KeyboardInterrupt")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()

