"""
管理全局监控实例：从 stream_store 加载配置，启动/停止 pipeline，维护 ViT + Agent 回调和 frame_sink。
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("web.runtime_manager")

# 项目根与路径
WEB_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = WEB_DIR.parent
VIT_DIR = PROJECT_ROOT / "Vit"
YOLO_DIR = PROJECT_ROOT / "yolo"


def _ensure_paths():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    if str(VIT_DIR) not in sys.path:
        sys.path.insert(0, str(VIT_DIR))


# 默认模型路径（可被环境变量或配置覆盖）
DEFAULT_YOLO_WEIGHTS = PROJECT_ROOT / "yolo" / "logs" / "best_epoch_weights.pth"
DEFAULT_YOLO_CLASSES = PROJECT_ROOT / "yolo" / "Class" / "coco_classes.txt"
DEFAULT_VIT_CHECKPOINT = PROJECT_ROOT / "Vit" / "lab_dataset" / "derived" / "known_classifier" / "checkpoint_best.pt"
DEFAULT_VLM_MERGED = PROJECT_ROOT / "vlm" / "outputs" / "merged"


class MonitorRuntimeManager:
    """
    从 stream_store 读取 enabled 流，启动一条 run_pipeline（YOLO + ViT + Agent + frame_sink）；
    支持 stop/start 后通过重启 pipeline 生效。
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
        self._state: Dict[str, Any] = {}
        self._vit_runtime: Any = None
        self._review_runtime: Any = None
        self._vit_clip_len = 16
        self._vit_frame_stride = 2
        self._clip_total = (self._vit_clip_len - 1) * self._vit_frame_stride + 1
        self._mirror_buffers: Dict[str, deque] = {}
        self._per_stream_vit_threshold: Dict[str, float] = {}

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def is_vit_loaded(self) -> bool:
        """ViT 模型是否已成功加载（用于前端显示）。"""
        return self._vit_runtime is not None

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
            stream_cfg[uri] = StreamConfig(
                rois=[],
                enable_dwell=True,
                dwell_classes=set(),
                enable_alarm=True,
                alarm_classes=set(),
            )
            self._per_stream_vit_threshold[uri] = float(s.get("vit_threshold", 0.6))

        if not sources:
            logger.info("No valid RTSP URLs in enabled streams")
            return True

        infer_conf = 0.3
        for s in enabled:
            infer_conf = float(s.get("yolo_confidence", 0.5))
            break

        sys_cfg = SystemConfig(
            max_batch=4,
            max_wait_ms=80.0,
            enable_display=False,
            resize_width=640,
            resize_height=480,
            infer_queue_maxsize=64,
            dwell_warning_sec=0,
            vit_anomaly_threshold=0.6,
            infer_conf_thres=infer_conf,
        )
        tracker_cfg = TrackerConfig(
            track_high_th=0.5,
            track_low_th=0.1,
            track_id_classes={"person"},
        )

        yolo_weights = os.environ.get("YOLO_WEIGHTS") or str(DEFAULT_YOLO_WEIGHTS)
        yolo_classes = os.environ.get("YOLO_CLASSES") or str(DEFAULT_YOLO_CLASSES)
        if not Path(yolo_weights).exists():
            logger.warning("YOLO weights not found: %s", yolo_weights)
        if not Path(yolo_classes).exists():
            logger.warning("YOLO classes not found: %s", yolo_classes)

        yolo = YOLO(model_path=yolo_weights, classes_path=yolo_classes)
        class_names = yolo.class_names

        self._state = {s.uri: {} for s in sources}
        vit_ckpt = os.environ.get("VIT_CHECKPOINT") or str(DEFAULT_VIT_CHECKPOINT)
        vlm_path = os.environ.get("VLM_MERGED") or str(DEFAULT_VLM_MERGED)

        try:
            from lab_anomaly.infer.known_event_runtime import KnownEventRuntime
            self._vit_runtime = KnownEventRuntime(
                known_checkpoint=vit_ckpt,
                clip_len=self._vit_clip_len,
                frame_stride=self._vit_frame_stride,
                window_stride=4,
                encoder_model_name="MCG-NJU/videomae-base",
                use_half=False,
            )
        except Exception as e:
            logger.warning("ViT runtime not loaded, anomaly review disabled: %s", e)
            self._vit_runtime = None

        self._review_runtime = None
        if Path(vlm_path).exists():
            try:
                from vlm.infer.vlm_engine import create_engine
                from web.services.vlm_review_runtime import VlmReviewRuntime
                engine = create_engine(vlm_path, use_vllm=False)
                self._review_runtime = VlmReviewRuntime(
                    engine=engine,
                    state_ref=self._state,
                    vit_threshold=0.6,
                    cooldown_sec=15.0,
                )
            except Exception as e:
                logger.warning("VLM review runtime not loaded: %s", e)
        else:
            logger.warning("VLM merged path not found: %s", vlm_path)

        self._mirror_buffers = {s.uri: deque(maxlen=self._clip_total) for s in sources}

        def save_duration(stream_id, roi_id, track_id, cls, enter_ts, end_ts, duration):
            logger.debug("DWELL %s roi=%s track=%s class=%s duration=%.2f", stream_id, roi_id, track_id, cls, duration)

        def on_frame_after_yolo(stream_id: str, frame_bgr: Any, ts: float, state: Dict[str, Any]) -> None:
            if self._vit_runtime is not None:
                self._vit_runtime.add_frame(stream_id, frame_bgr, state)
            if stream_id not in self._mirror_buffers:
                return
            buf = self._mirror_buffers[stream_id]
            import cv2
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            buf.append(rgb)
            if len(buf) >= self._clip_total and self._review_runtime is not None:
                clip_frames = [buf[i * self._vit_frame_stride] for i in range(self._vit_clip_len)]
                self._review_runtime.set_last_clip(stream_id, clip_frames)
                vit_result = self._vit_runtime.get_result(stream_id) if self._vit_runtime else None
                th = self._per_stream_vit_threshold.get(stream_id, 0.6)
                self._review_runtime.try_submit_if_triggered(stream_id, vit_result, th)

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