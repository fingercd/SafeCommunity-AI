"""
Validation checklist (plan section 10): verify before production.
"""
# Cross-stream ID isolation: trackers and state keyed by stream_id; logs must include stream_id.
# Occlusion stability: tune TRACK_HIGH_TH, TRACK_LOW_TH, LOST_TIMEOUT_SEC; verify ID continuity after occlusion.
# ROI correctness: spot-check enter/exit events on recorded video; confirm bottom_center anchor matches use case.
# Alarm no spam: verify cooldown and per-track dedup; different tracks (e.g. different animals) can each alarm once.
# File source EOF: EOF triggers flush and does not affect other streams.

VALIDATION_CHECKLIST = [
    "Cross-stream ID isolation: track_id only within trackers[stream_id]; all logs include stream_id.",
    "Occlusion stability: adjust TRACK_HIGH_TH / TRACK_LOW_TH / LOST_TIMEOUT_SEC; confirm ID persists after occlusion.",
    "ROI correctness: manually verify enter/exit events on clips; confirm roi_point_mode (e.g. bottom_center) is correct.",
    "Alarm no spam: same track does not alarm every frame; cooldown works; different tracks can alarm separately.",
    "File EOF: on EndOfStream(stream_id), flush roi_start for that stream only; other streams keep running.",
]
