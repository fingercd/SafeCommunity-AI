"""
Alarm: filter by alarm_classes, cooldown per track, trigger_sound + write_alarm_log. Per-stream.
Vehicle alarm: when ROI 存在时仅在禁区内计时，边上的车不会误报。
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from .types import Track
from .config import StreamConfig
from .roi_geometry import roi_membership
from .state import keys_where_track_id

logger = logging.getLogger("realtime.alarm")


def trigger_sound(stream_id: str, class_name: str) -> None:
    """Platform-dependent beep. Windows: winsound; else optional."""
    try:
        import winsound
        winsound.Beep(1000, 300)
    except Exception:
        pass


def write_alarm_log(
    stream_id: str,
    ts: float,
    track_id: int,
    class_name: str,
    score: float,
    bbox_xyxy,
) -> None:
    """Log alarm event."""
    logger.warning(
        "ALARM stream_id=%s ts=%.2f track_id=%d class=%s score=%.2f bbox=%s",
        stream_id, ts, track_id, class_name, score, bbox_xyxy.tolist() if hasattr(bbox_xyxy, "tolist") else bbox_xyxy,
    )


def update_alarm(
    stream_id: str,
    ts: float,
    tracks: List[Track],
    stream_cfg: Dict[str, "StreamConfig"],
    state: Dict[str, Any],
    alarm_cooldown_sec: float,
) -> None:
    """
    For each track in alarm_classes: if cooldown passed, trigger_sound + write_alarm_log, set alarm_last_ts.
    """
    cfg = stream_cfg.get(stream_id)
    if not cfg or not cfg.enable_alarm or not cfg.alarm_classes:
        return

    st = state[stream_id]
    alarm_last_ts = st["alarm_last_ts"]

    for trk in tracks:
        tid = trk.id
        cls = trk.class_name
        if cls not in cfg.alarm_classes:
            continue

        last = alarm_last_ts.get(tid, None)
        if last is not None and (ts - last) < alarm_cooldown_sec:
            continue

        trigger_sound(stream_id, cls)
        write_alarm_log(stream_id, ts, tid, cls, trk.score, trk.bbox_xyxy)
        alarm_last_ts[tid] = ts


def update_vehicle_alarm(
    stream_id: str,
    ts: float,
    tracks: List[Track],
    stream_cfg: "StreamConfig",
    state: Dict[str, Any],
    alarm_cooldown_sec: float = 30.0,
) -> List[Tuple[int, float]]:
    """
    车辆报警：有 ROI 时仅在禁区内计时（锚点 bottom_center 在框内才累计），边上在框外的车不会触发。
    无 ROI 时沿用“画面内可见时长”逻辑。返回 [(track_id, duration_sec), ...] 供顶部栏显示。
    """
    if not stream_cfg.vehicle_alarm_classes or stream_cfg.vehicle_alarm_sec <= 0:
        return []
    st = state.get(stream_id)
    if not st:
        return []
    alarm_last_ts = st.setdefault("alarm_last_ts", {})
    sec = stream_cfg.vehicle_alarm_sec
    rois = getattr(stream_cfg, "rois", None) or []
    mode = getattr(stream_cfg, "roi_point_mode", "bottom_center")

    if not rois:
        # 无 ROI：按画面内可见时长
        first_seen_ts = st.setdefault("first_seen_ts", {})
        active: List[Tuple[int, float]] = []
        for trk in tracks:
            if trk.class_name not in stream_cfg.vehicle_alarm_classes:
                continue
            tid = trk.id
            if tid not in first_seen_ts:
                first_seen_ts[tid] = ts
            duration = ts - first_seen_ts[tid]
            if duration >= sec:
                active.append((tid, duration))
                last = alarm_last_ts.get(tid)
                if last is None or (ts - last) >= alarm_cooldown_sec:
                    trigger_sound(stream_id, f"vehicle_{trk.class_name}")
                    logger.warning(
                        "VEHICLE_ALARM stream_id=%s ts=%.2f track_id=%d class=%s visible_sec=%.2f",
                        stream_id, ts, tid, trk.class_name, duration,
                    )
                    alarm_last_ts[tid] = ts
        return active

    # 有 ROI：仅在禁区内计时
    vehicle_roi_start = st.setdefault("vehicle_roi_start", {})
    current_vehicle_tids = {t.id for t in tracks if t.class_name in stream_cfg.vehicle_alarm_classes}
    for key in list(vehicle_roi_start.keys()):
        if key[1] not in current_vehicle_tids:
            del vehicle_roi_start[key]
    for trk in tracks:
        if trk.class_name not in stream_cfg.vehicle_alarm_classes:
            continue
        tid = trk.id
        inside = roi_membership(trk.bbox_xyxy, rois, mode)
        for roi_id in inside:
            key = (roi_id, tid)
            if key not in vehicle_roi_start:
                vehicle_roi_start[key] = ts
        for key in list(keys_where_track_id(vehicle_roi_start, tid)):
            roi_id = key[0]
            if roi_id not in inside:
                del vehicle_roi_start[key]

    # 禁区内停留 >= sec 的 (tid, duration)，同 tid 取最大 duration
    by_tid: Dict[int, float] = {}
    for (roi_id, tid), enter_ts in vehicle_roi_start.items():
        duration = ts - enter_ts
        if duration >= sec:
            by_tid[tid] = max(by_tid.get(tid, 0), duration)
    for tid, duration in by_tid.items():
        last = alarm_last_ts.get(tid)
        if last is None or (ts - last) >= alarm_cooldown_sec:
            trigger_sound(stream_id, "vehicle")
            logger.warning(
                "VEHICLE_ALARM stream_id=%s ts=%.2f track_id=%d in_roi_sec=%.2f",
                stream_id, ts, tid, duration,
            )
            alarm_last_ts[tid] = ts
    return [(tid, by_tid[tid]) for tid in sorted(by_tid.keys())]
