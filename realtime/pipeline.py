"""
Main pipeline: capture threads (RTSP/File) -> infer queue -> batch aggregator -> YOLO -> on_frame (ByteTrack, dwell, alarm, GC).
Handles EndOfStream: flush ROI and log.
"""
import queue
import threading
import logging
import time
from typing import List, Dict, Callable, Optional, Any

import cv2

from .config import SourceConfig, StreamConfig, SystemConfig, TrackerConfig
from .batch_aggregator import collect_batch
from .state import make_stream_state
from .bytetrack_wrapper import BYTETracker
from .dwell import update_roi_dwell
from .alarm import update_alarm, update_vehicle_alarm
from .gc_flush import garbage_collect, on_end_of_stream_flush
from .capture_rtsp import capture_loop as capture_rtsp_loop
from .capture_file import capture_loop as capture_file_loop
from .yolo_batch import yolo_infer_batch
from .display import draw_frame

logger = logging.getLogger("realtime.pipeline")


def run_pipeline(
    sources: List[SourceConfig],
    stream_cfg: Dict[str, StreamConfig],
    sys_cfg: SystemConfig,
    tracker_cfg: TrackerConfig,
    yolo,
    class_names: List[str],
    save_duration: Callable[[str, int, int, str, float, float, float], None],
    log_stream_end: Optional[Callable[[str], None]] = None,
    stop_event: Optional[threading.Event] = None,
    on_frame_after_yolo: Optional[Callable[[str, Any, float, Dict[str, Any]], None]] = None,
) -> None:
    """
    Start capture threads for each source (RTSP or file), run infer worker in main thread.
    on_frame_after_yolo: optional callback(stream_id, frame_bgr, ts, state) after YOLO/dwell/alarm for this frame (e.g. push to ViT buffer and update state["vit_event"]).
    """
    infer_queue = queue.Queue(maxsize=sys_cfg.infer_queue_maxsize)
    stop = stop_event or threading.Event()

    # Per-stream state and trackers; stream_id -> display window index
    enabled_sources = [s for s in sources if s.enabled]
    state = {s.uri: make_stream_state() for s in enabled_sources}
    trackers = {s.uri: BYTETracker(tracker_cfg, class_names) for s in enabled_sources}
    stream_index_map = {s.uri: idx for idx, s in enumerate(enabled_sources)}

    def _save_duration(stream_id, roi_id, track_id, cls, enter_ts, end_ts, duration):
        save_duration(stream_id, roi_id, track_id, cls, enter_ts, end_ts, duration)

    def _log_stream_end(sid):
        if log_stream_end:
            log_stream_end(sid)
        logger.info("Stream ended: %s", sid)

    # Start capture threads
    threads = []
    for src in sources:
        if not src.enabled:
            continue
        if src.type == "rtsp":
            t = threading.Thread(
                target=capture_rtsp_loop,
                args=(src, infer_queue, sys_cfg),
                kwargs={"stop_event": stop},
                daemon=True,
            )
        else:
            t = threading.Thread(
                target=capture_file_loop,
                args=(src, infer_queue, sys_cfg.drop_policy),
                kwargs={"stop_event": stop, "target_fps": 20.0},
                daemon=True,
            )
        t.start()
        threads.append(t)

    # Infer worker
    high_th = tracker_cfg.track_high_th
    low_th = tracker_cfg.track_low_th
    # FPS 统计：每 2 秒输出一次主线程处理帧率，便于验证优化效果
    fps_frame_count = 0
    fps_last_time = time.monotonic()

    while not stop.is_set():
        frame_events, eos_events = collect_batch(
            infer_queue, sys_cfg.max_batch, sys_cfg.max_wait_ms,
        )

        for eos in eos_events:
            on_end_of_stream_flush(
                eos.stream_id, time.monotonic(), state, _save_duration, _log_stream_end,
            )

        if not frame_events:
            continue

        fps_frame_count += len(frame_events)
        elapsed = time.monotonic() - fps_last_time
        if elapsed >= 2.0:
            logger.info("Pipeline FPS: %.1f (%.0f frames in %.2fs)", fps_frame_count / elapsed, fps_frame_count, elapsed)
            # GPU 利用率可另开终端用: nvidia-smi dmon -s u
            fps_frame_count = 0
            fps_last_time = time.monotonic()

        frames = [ev.frame for ev in frame_events]
        stream_ids = [ev.stream_id for ev in frame_events]
        ts_list = [ev.ts for ev in frame_events]

        conf_thres = getattr(sys_cfg, "infer_conf_thres", None)
        det_list = yolo_infer_batch(
            yolo, frames,
            conf_thres=conf_thres if conf_thres is not None else yolo.confidence,
        )

        for i in range(len(frame_events)):
            stream_id = stream_ids[i]
            ts = ts_list[i]
            det = det_list[i]

            boxes = det.boxes_xyxy
            scores = det.score
            classes = det.class_id
            n = len(boxes)

            high = [
                (boxes[j], float(scores[j]), int(classes[j]))
                for j in range(n) if scores[j] >= high_th
            ]
            low = [
                (boxes[j], float(scores[j]), int(classes[j]))
                for j in range(n) if low_th <= scores[j] < high_th
            ]

            tracks = trackers[stream_id].update(high, low, ts)

            cfg = stream_cfg.get(stream_id)
            if cfg and cfg.enable_dwell:
                update_roi_dwell(stream_id, ts, tracks, stream_cfg, state, _save_duration)
            if cfg and cfg.enable_alarm:
                update_alarm(stream_id, ts, tracks, stream_cfg, state, sys_cfg.alarm_cooldown_sec)
            vehicle_alarm_active = []
            if cfg:
                vehicle_alarm_active = update_vehicle_alarm(
                    stream_id, ts, tracks, cfg, state, sys_cfg.alarm_cooldown_sec,
                )

            # Update track history for trajectory (bbox center, max 50 points)
            st = state[stream_id]
            track_history = st.setdefault("track_history", {})
            max_len = 50
            for t in tracks:
                x1, y1, x2, y2 = t.bbox_xyxy
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                if t.id not in track_history:
                    track_history[t.id] = []
                track_history[t.id].append((float(cx), float(cy)))
                if len(track_history[t.id]) > max_len:
                    track_history[t.id] = track_history[t.id][-max_len:]

            garbage_collect(stream_id, ts, state, sys_cfg, _save_duration)

            if on_frame_after_yolo is not None:
                try:
                    on_frame_after_yolo(stream_id, frames[i], ts, state)
                except Exception as e:
                    logger.warning("on_frame_after_yolo error: %s", e)

            if sys_cfg.enable_display and cfg:
                stream_index = stream_index_map.get(stream_id, 0)
                roi_start = state[stream_id]["roi_start"]
                st = state[stream_id]
                draw_frame(
                    stream_id=stream_id,
                    stream_index=stream_index,
                    frame_bgr=frames[i],
                    tracks=tracks,
                    stream_cfg=cfg,
                    roi_start=roi_start,
                    track_history=st.get("track_history", {}),
                    vehicle_alarm_active=vehicle_alarm_active,
                    ts=ts,
                    dwell_warning_sec=sys_cfg.dwell_warning_sec,
                    enable_display=sys_cfg.enable_display,
                    display_scale=sys_cfg.display_scale,
                    display_width=getattr(sys_cfg, "display_width", 0),
                    display_height=getattr(sys_cfg, "display_height", 0),
                    bbox_thickness=sys_cfg.bbox_thickness,
                    box_expand_px=sys_cfg.box_expand_px,
                    label_font_scale=sys_cfg.label_font_scale,
                    label_thickness=sys_cfg.label_thickness,
                    banner_font_size=sys_cfg.banner_font_size,
                    chinese_font_path=sys_cfg.chinese_font_path,
                    force_pil_text=sys_cfg.force_pil_text,
                    display_id_trajectory_classes=getattr(cfg, "display_id_trajectory_classes", None) or sys_cfg.display_id_trajectory_classes or {"person"},
                    vit_event=st.get("vit_event"),
                    show_corner_overlay=getattr(sys_cfg, "show_corner_overlay", True),
                    vit_anomaly_threshold=getattr(sys_cfg, "vit_anomaly_threshold", 0.7),
                    show_corner_rules_summary=getattr(sys_cfg, "show_corner_rules_summary", False),
                    corner_reserve_left=getattr(sys_cfg, "corner_reserve_left", 100),
                    show_corner_vit_label=getattr(sys_cfg, "show_corner_vit_label", False),
                )

        if sys_cfg.enable_display and frame_events:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                stop.set()

    if sys_cfg.enable_display:
        cv2.destroyAllWindows()
