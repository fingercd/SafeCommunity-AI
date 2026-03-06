"""
Per-stream state: roi_start, last_seen, alarm_last_ts. One dict per stream_id.
"""
from typing import Dict, Any


def make_stream_state() -> Dict[str, Any]:
    return {
        "roi_start": {},       # (roi_id, track_id) -> enter_ts
        "last_seen": {},       # track_id -> last_ts
        "alarm_last_ts": {},   # track_id -> last_alarm_ts
        "first_seen_ts": {},   # track_id -> first_ts (for vehicle-alarm when no ROI)
        "vehicle_roi_start": {},  # (roi_id, track_id) -> enter_ts（车辆在禁区内才开始计时）
        "track_history": {},   # track_id -> list of (x, y) center points for trajectory
    }


def keys_where_track_id(roi_start: dict, track_id: int):
    """All keys (roi_id, tid) in roi_start with tid == track_id."""
    return [k for k in roi_start if k[1] == track_id]
