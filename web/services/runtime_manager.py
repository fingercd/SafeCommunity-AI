"""
Global monitor runtime: loads stream config, starts pipeline with YOLO + ViT + VLM,
maintains callbacks and frame_sink for the web frontend.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("web.runtime_manager")

WEB_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = WEB_DIR.parent
VIT_DIR = PROJECT_ROOT / "Vit"
YOLO_DIR = PROJECT_ROOT / "yolo"

DEFAULT_YOLO_WEIGHTS = PROJECT_ROOT / "yolo" / "logs" / "best_epoch_weights.pth"
DEFAULT_YOLO_CLASSES = PROJECT_ROOT / "yolo" / "Class" / "coco_classes.txt"
DEFAULT_VIT_CHECKPOINT = (
    PROJECT_ROOT / "Vit" / "lab_dataset" / "derived" / "end2end_classifier" / "checkpoint_best.pt"
)
DEFAULT_VLM_MERGED = PROJECT_ROOT / "vlm" / "outputs" / "merged"
DEFAULT_VLM_BASE = PROJECT_ROOT / "vlm" / "Qwen"

_DEFAULT_ROI_ALARM_CLASSES = {"person"}
_DEFAULT_GLOBAL_ALARM_CLASSES = {"fire", "smoke"}
_DEFAULT_VEHICLE_ALARM_CLASSES = {"car", "truck", "bus", "motorcycle"}


def _ensure_paths():
    for p in (str(PROJECT_ROOT), str(VIT_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)


class MonitorRuntimeManager:
    """
    Reads enabled streams from stream_store, starts one run_pipeline thread
    (YOLO + ViT + optional VLM review + frame_sink).
    """

    def __init__(
        self,
        stream_store_load: Callable[[], List[dict]],
        frame_state: Any,
        config_path: Optional[Path] = None,
    ):
        self._stream_store_load = stream_store_load
        self._frame_state = frame_state
        self._config_path = config_path
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._vit_runtime: Any = None
        self._vlm_loaded = False
        self._vlm_is_finetuned = False
        self._review_runtime: Any = None
        self._vit_clip_len = 16
        self._vit_frame_stride = 2
        self._clip_total = (self._vit_clip_len - 1) * self._vit_frame_stride + 1
        self._mirror_buffers: Dict[str, deque] = {}
        self._per_stream_vit_threshold: Dict[str, float] = {}
        self._per_stream_agent_enabled: Dict[str, bool] = {}
        self._per_stream_vlm_interval: Dict[str, float] = {}
        self._last_vlm_auto_ts: Dict[str, float] = {}

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def is_vit_loaded(self) -> bool:
        return self._vit_runtime is not None

    def is_vlm_loaded(self) -> bool:
        return self._vlm_loaded

    def is_vlm_finetuned(self) -> bool:
        return self._vlm_is_finetuned

    def is_vlm_base_model(self) -> bool:
        """Return True if loaded VLM is a base (not instruct) model."""
        if self._review_runtime is None or self._review_runtime.engine is None:
            return False
        return getattr(self._review_runtime.engine, "model_type", "") == "base"

    def start(self) -> bool:
        _ensure_paths()
        if self.is_running():
            logger.warning("Pipeline already running")
            return False
        streams = self._stream_store_load()
        enabled = [s for s in streams if s.get("enabled", True)]
        if not enabled:
            logger.info("No enabled streams to run")
            return True

        try:
            from yolo.realtime.config import SourceConfig, StreamConfig, SystemConfig, TrackerConfig
            from yolo.realtime.pipeline import run_pipeline
            from yolo import YOLO
        except Exception as e:
            logger.error("Failed to import yolo: %s", e)
            return False

        sources: List[SourceConfig] = []
        stream_cfg: Dict[str, StreamConfig] = {}

        for s in enabled:
            uri = (s.get("rtsp_url") or "").strip()
            if not uri:
                continue
            sources.append(SourceConfig(uri=uri, type="rtsp", enabled=True))

            raw_rois = s.get("rois") or []
            rois = [
                [(pt[0], pt[1]) for pt in poly]
                for poly in raw_rois
                if isinstance(poly, list) and len(poly) >= 3
            ]

            roi_alarm = set(s.get("roi_alarm_classes") or []) or _DEFAULT_ROI_ALARM_CLASSES
            global_alarm = set(s.get("global_alarm_classes") or []) or _DEFAULT_GLOBAL_ALARM_CLASSES

            stream_cfg[uri] = StreamConfig(
                rois=rois,
                roi_point_mode="bottom_center",
                enable_dwell=True,
                dwell_classes=roi_alarm,
                enable_alarm=True,
                alarm_classes=global_alarm,
                vehicle_alarm_classes=_DEFAULT_VEHICLE_ALARM_CLASSES,
                vehicle_alarm_sec=30.0,
            )
            self._per_stream_vit_threshold[uri] = float(s.get("vit_threshold", 0.6))
            self._per_stream_agent_enabled[uri] = bool(s.get("agent_enabled", True))
            self._per_stream_vlm_interval[uri] = float(s.get("vlm_auto_interval_sec", 16.0))

        if not sources:
            logger.info("No valid RTSP URLs in enabled streams")
            return True

        infer_conf = float(enabled[0].get("yolo_confidence", 0.5)) if enabled else 0.3

        sys_cfg = SystemConfig(
            max_batch=4,
            max_wait_ms=80.0,
            enable_display=False,
            resize_width=640,
            resize_height=480,
            infer_queue_maxsize=64,
            dwell_warning_sec=10.0,
            vit_anomaly_threshold=0.6,
            infer_conf_thres=infer_conf,
            alarm_cooldown_sec=30.0,
        )
        tracker_cfg = TrackerConfig(
            track_high_th=0.5,
            track_low_th=0.1,
            track_id_classes={"person"},
            vehicle_id_classes=_DEFAULT_VEHICLE_ALARM_CLASSES,
        )

        yolo_weights = os.environ.get("YOLO_WEIGHTS") or str(DEFAULT_YOLO_WEIGHTS)
        yolo_classes = os.environ.get("YOLO_CLASSES") or str(DEFAULT_YOLO_CLASSES)
        if not Path(yolo_weights).exists():
            logger.warning("YOLO weights not found: %s", yolo_weights)
        if not Path(yolo_classes).exists():
            logger.warning("YOLO classes not found: %s", yolo_classes)

        yolo = YOLO(model_path=yolo_weights, classes_path=yolo_classes)
        class_names = yolo.class_names

        vit_ckpt = os.environ.get("VIT_CHECKPOINT") or str(DEFAULT_VIT_CHECKPOINT)

        try:
            from lab_anomaly.infer.known_event_runtime import KnownEventRuntime
            self._vit_runtime = KnownEventRuntime(
                known_checkpoint=vit_ckpt,
                frames_per_clip=self._vit_clip_len,
                window_stride=4,
                encoder_model_name="OpenGVLab/VideoMAEv2-Base",
                use_half=False,
            )
        except Exception as e:
            logger.warning("ViT runtime not loaded: %s", e)
            self._vit_runtime = None

        # VLM: prefer fine-tuned merged model, fall back to base model
        self._review_runtime = None
        self._vlm_loaded = False
        self._vlm_is_finetuned = False

        vlm_merged = os.environ.get("VLM_MERGED") or str(DEFAULT_VLM_MERGED)
        vlm_base = os.environ.get("VLM_BASE") or str(DEFAULT_VLM_BASE)

        vlm_load_path: Optional[str] = None
        if Path(vlm_merged).exists():
            vlm_load_path = vlm_merged
            self._vlm_is_finetuned = True
            logger.info("VLM: using fine-tuned merged model at %s", vlm_merged)
        elif Path(vlm_base).exists():
            vlm_load_path = vlm_base
            self._vlm_is_finetuned = False
            logger.info("VLM: using base model at %s", vlm_base)
        else:
            logger.info("VLM: no model found (merged=%s, base=%s), VLM disabled", vlm_merged, vlm_base)

        if vlm_load_path:
            try:
                from vlm.infer.vlm_engine import create_engine
                from web.services.vlm_review_runtime import VlmReviewRuntime
                engine = create_engine(vlm_load_path, use_vllm=False)
                # 检测 Base 模型并给出明确提示
                if getattr(engine, "model_type", "") == "base":
                    logger.warning(
                        "VLM: 检测到基础模型(Base)@%s。Base模型不支持指令遵循，VLM将返回提示而非推理结果。"
                        "请下载 Qwen3.5-VL-Instruct 替换此目录。", vlm_load_path
                    )
                # Eagerly load the underlying model so import/version errors are
                # caught here instead of during the first inference call.
                try:
                    engine._ensure_loaded()
                except Exception as load_err:
                    raise RuntimeError(
                        f"VLM backbone load failed: {load_err}. "
                        f"Your transformers version may not support this model type. "
                        f"Try: pip install --upgrade transformers"
                    ) from load_err
                self._review_runtime = VlmReviewRuntime(
                    engine=engine,
                    vit_threshold=0.6,
                    cooldown_sec=15.0,
                )
                self._vlm_loaded = True
                logger.info("VLM loaded successfully (finetuned=%s, base=%s)", self._vlm_is_finetuned, getattr(engine, "model_type", "unknown"))
            except Exception as e:
                logger.warning("VLM review runtime not loaded: %s", e)
                self._review_runtime = None
                self._vlm_loaded = False
                self._vlm_is_finetuned = False

        self._mirror_buffers = {s.uri: deque(maxlen=self._clip_total) for s in sources}
        self._last_vlm_auto_ts = {}

        def save_duration(stream_id, roi_id, track_id, cls, enter_ts, end_ts, duration):
            logger.info(
                "DWELL stream=%s roi=%d track=%d class=%s duration=%.1fs",
                stream_id, roi_id, track_id, cls, duration,
            )

        def on_frame_after_yolo(stream_id: str, frame_bgr: Any, ts: float, state: Dict[str, Any]) -> None:
            if self._vit_runtime is not None:
                self._vit_runtime.add_frame(stream_id, frame_bgr, state)

            if self._review_runtime is not None:
                vlm_result = self._review_runtime.pop_result(stream_id)
                if vlm_result is not None and stream_id in state:
                    state[stream_id].agent_result = vlm_result

            if stream_id not in self._mirror_buffers:
                return
            if not self._per_stream_agent_enabled.get(stream_id, True):
                return
            if self._review_runtime is None:
                return

            import cv2
            buf = self._mirror_buffers[stream_id]
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            buf.append(rgb)

            if len(buf) < self._clip_total:
                return

            clip_frames = [buf[i * self._vit_frame_stride] for i in range(self._vit_clip_len)]

            # Path 1: ViT-anomaly-triggered (existing, with 15s cooldown)
            self._review_runtime.set_last_clip(stream_id, clip_frames)
            vit_result = self._vit_runtime.get_result(stream_id) if self._vit_runtime else None
            th = self._per_stream_vit_threshold.get(stream_id, 0.6)
            self._review_runtime.try_submit_if_triggered(stream_id, vit_result, th)

            # Path 2: periodic auto-trigger (ignores ViT judgment)
            interval = self._per_stream_vlm_interval.get(stream_id, 0.0)
            if interval > 0:
                now = time.time()
                last = self._last_vlm_auto_ts.get(stream_id, 0.0)
                if now - last >= interval:
                    submitted = self._review_runtime.submit_periodic(stream_id, clip_frames)
                    if submitted:
                        self._last_vlm_auto_ts[stream_id] = now
                        logger.debug("VLM periodic triggered for stream=%s", stream_id)

        def frame_sink(stream_id: str, drawn_frame: Any, ts: float, state: Dict[str, Any]) -> None:
            self._frame_state.update(stream_id, drawn_frame, ts, state)

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=run_pipeline,
            kwargs=dict(
                sources=sources,
                stream_cfg=stream_cfg,
                sys_cfg=sys_cfg,
                tracker_cfg=tracker_cfg,
                yolo=yolo,
                class_names=class_names,
                save_duration=save_duration,
                log_stream_end=lambda sid: logger.info("Stream ended: %s", sid),
                stop_event=self._stop_event,
                on_frame_after_yolo=on_frame_after_yolo,
                frame_sink=frame_sink,
            ),
            daemon=True,
        )
        self._thread.start()
        logger.info("Pipeline started with %d streams", len(sources))
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10.0)
            self._thread = None
        logger.info("Pipeline stopped")
