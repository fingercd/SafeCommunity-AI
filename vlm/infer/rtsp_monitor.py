"""
RTSP / 本地视频实时异常监控：拉流 → 滑动窗口组 clip → VLM 分析 → 时序平滑与置信度过滤 → 告警输出。

- 支持 RTSP URL 与本地视频文件
- 帧缓冲 + 每 N 帧组成一个 clip（默认 8 帧），送入 VLM
- 连续至少 consecutive_anomaly_clips 个 clip 判定为异常且置信度 >= threshold 才触发告警
- 告警输出到控制台与 JSON 日志文件
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

# 支持以模块或脚本运行
try:
    from vlm.infer.vlm_engine import VLMEngine, create_engine
except Exception:
    try:
        from .vlm_engine import VLMEngine, create_engine
    except Exception:
        from vlm_engine import VLMEngine, create_engine


def parse_args():
    ap = argparse.ArgumentParser(description="RTSP/视频流 VLM 异常监控")
    ap.add_argument("--source", type=str, required=True, help="RTSP URL 或本地视频路径")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--model_path", type=str, default="outputs/qlora/final")
    ap.add_argument("--clip_frames", type=int, default=8)
    ap.add_argument("--fps_sample", type=float, default=4.0, help="采样帧率，用于组 clip")
    ap.add_argument("--confidence_threshold", type=float, default=0.7)
    ap.add_argument("--consecutive_anomaly_clips", type=int, default=2)
    ap.add_argument("--log_path", type=str, default="outputs/rtsp_alerts.jsonl")
    ap.add_argument("--no_display", action="store_true", help="不弹窗显示画面")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / args.config
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        inf_cfg = cfg.get("inference", {})
        confidence_threshold = args.confidence_threshold or inf_cfg.get("confidence_threshold", 0.7)
        consecutive_n = args.consecutive_anomaly_clips or inf_cfg.get("consecutive_anomaly_clips", 2)
    else:
        confidence_threshold = args.confidence_threshold
        consecutive_n = args.consecutive_anomaly_clips

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = project_root / model_path
    log_path = Path(args.log_path)
    if not log_path.is_absolute():
        log_path = project_root / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(str(model_path), use_vllm=False)
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("无法打开视频源:", args.source)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(round(fps / args.fps_sample)))
    clip_frames = args.clip_frames
    buffer: deque = deque(maxlen=clip_frames * 2)
    anomaly_streak = 0
    last_alert_time = 0.0
    alert_cooldown = 5.0

    def write_alert(obj: dict) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    frame_idx = 0
    print("开始监控，按 q 退出...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % interval != 0:
            continue
        buffer.append(frame)
        if len(buffer) < clip_frames:
            if not args.no_display:
                cv2.imshow("monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            continue
        clip = list(buffer)[-clip_frames:]
        result = engine.analyze_clip([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in clip])
        is_anomaly = result.get("is_anomaly", False)
        conf = result.get("confidence", 0.0)
        if is_anomaly and conf >= confidence_threshold:
            anomaly_streak += 1
            if anomaly_streak >= consecutive_n and (time.time() - last_alert_time) >= alert_cooldown:
                last_alert_time = time.time()
                alert = {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "source": args.source,
                    "is_anomaly": True,
                    "anomaly_type": result.get("anomaly_type", ""),
                    "confidence": conf,
                    "reasoning": result.get("reasoning", ""),
                }
                print("[告警]", alert)
                write_alert(alert)
        else:
            anomaly_streak = 0

        if not args.no_display:
            label = "ANOMALY" if is_anomaly else "normal"
            cv2.putText(frame, f"{label} {conf:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if not is_anomaly else (0, 0, 255), 2)
            cv2.imshow("monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()
    print("监控结束")


if __name__ == "__main__":
    main()
