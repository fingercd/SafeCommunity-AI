"""
ROI dwell: enter / geometry-exit. State in state[stream_id]; lost-exit handled in gc_flush.
"""
from typing import Dict, List, Callable, Any
from .types import Track
from .config import StreamConfig
from .roi_geometry import roi_membership
from .state import keys_where_track_id


def update_roi_dwell(
    stream_id: str,
    ts: float,
    tracks: List[Track],
    stream_cfg: Dict[str, StreamConfig],
    state: Dict[str, Any],
    save_duration: Callable[[str, int, int, str, float, float, float], None],
) -> None:
    """
    Update last_seen for all tracks; for dwell_classes only: enter ROI, geometry-exit and save duration.
    """
    cfg = stream_cfg.get(stream_id)
    if not cfg or not cfg.enable_dwell or not cfg.rois:
        return

    rois = cfg.rois
    mode = cfg.roi_point_mode
    dwell_classes = cfg.dwell_classes
    st = state[stream_id]
    roi_start = st["roi_start"]
    last_seen = st["last_seen"]

    for trk in tracks:
        tid = trk.id
        cls = trk.class_name
        bbox = trk.bbox_xyxy
        last_seen[tid] = ts

        if cls not in dwell_classes:
            continue

        inside = roi_membership(bbox, rois, mode)

        # enter
        for roi_id in inside:
            key = (roi_id, tid)
            if key not in roi_start:
                roi_start[key] = ts

        # geometry exit
        for key in keys_where_track_id(roi_start, tid):
            roi_id = key[0]
            if roi_id not in inside:
                enter_ts = roi_start[key]
                duration = ts - enter_ts
                save_duration(stream_id, roi_id, tid, cls, enter_ts, ts, duration)
                del roi_start[key]
