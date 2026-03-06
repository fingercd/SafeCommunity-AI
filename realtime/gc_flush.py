"""
GC: lost-timeout exit (settle roi_start), prune old last_seen/alarm_last_ts.
EndOfStream flush: settle all roi_start for that stream and log.
"""
from typing import Dict, Any, Callable
from .config import SystemConfig


def garbage_collect(
    stream_id: str,
    ts: float,
    state: Dict[str, Any],
    sys_cfg: SystemConfig,
    save_duration: Callable[[str, int, int, str, float, float, float], None],
) -> None:
    """
    Lost-exit: (roi_id, track_id) in roi_start with last_seen too old -> settle and delete.
    Prune: last_seen / alarm_last_ts entries older than MAX_STATE_SEC.
    """
    st = state.get(stream_id)
    if not st:
        return

    roi_start = st["roi_start"]
    last_seen = st["last_seen"]
    alarm_last_ts = st.get("alarm_last_ts", {})
    lost_timeout = sys_cfg.lost_timeout_sec
    max_state = sys_cfg.max_state_sec

    # lost-exit for ROI
    to_del = []
    for (roi_id, track_id), enter_ts in list(roi_start.items()):
        last = last_seen.get(track_id, None)
        if last is None or (ts - last) > lost_timeout:
            end_ts = last or ts
            duration = end_ts - enter_ts
            save_duration(stream_id, roi_id, track_id, "unknown", enter_ts, end_ts, duration)
            to_del.append((roi_id, track_id))
    for k in to_del:
        del roi_start[k]

    # prune old states
    first_seen_ts = st.get("first_seen_ts", {})
    track_history = st.get("track_history", {})
    vehicle_roi_start = st.get("vehicle_roi_start", {})
    for track_id, last in list(last_seen.items()):
        if (ts - last) > max_state:
            del last_seen[track_id]
            if track_id in alarm_last_ts:
                del alarm_last_ts[track_id]
            if track_id in first_seen_ts:
                del first_seen_ts[track_id]
            if track_id in track_history:
                del track_history[track_id]
            for key in list(vehicle_roi_start.keys()):
                if key[1] == track_id:
                    del vehicle_roi_start[key]


def on_end_of_stream_flush(
    stream_id: str,
    ts: float,
    state: Dict[str, Any],
    save_duration: Callable[[str, int, int, str, float, float, float], None],
    log_stream_end: Callable[[str], None],
) -> None:
    """
    On EOF: settle all roi_start for this stream (use last_seen or ts), then log stream end.
    """
    st = state.get(stream_id)
    if not st:
        log_stream_end(stream_id)
        return

    roi_start = st["roi_start"]
    last_seen = st["last_seen"]

    for (roi_id, track_id), enter_ts in list(roi_start.items()):
        last = last_seen.get(track_id, ts)
        duration = last - enter_ts
        save_duration(stream_id, roi_id, track_id, "unknown", enter_ts, last, duration)
    roi_start.clear()
    st.get("vehicle_roi_start", {}).clear()

    log_stream_end(stream_id)
