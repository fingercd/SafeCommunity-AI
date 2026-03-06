"""
ByteTrack integration: per-stream tracker, update(high, low, ts) -> list[Track].
Track class_id/class_name/score from matched detection; else keep previous.
Only classes in track_id_classes get new IDs; lost tracks use expanded bbox for matching (occlusion).
"""
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from .types import Track
from .config import TrackerConfig


def _expand_bbox(bbox: np.ndarray, scale: float = 1.2) -> np.ndarray:
    """Expand bbox from center by scale (e.g. 1.2 = 20% larger) for matching when track is lost."""
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1) * scale / 2.0, (y2 - y1) * scale / 2.0
    return np.array([cx - w, cy - h, cx + w, cy + h], dtype=np.float64)


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """boxes (N,4) xyxy; returns (N, M) IoU."""
    na, nb = len(boxes_a), len(boxes_b)
    if na == 0 or nb == 0:
        return np.zeros((na, nb), dtype=np.float32)
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-7)


class _TrackState:
    """Internal state for one track."""
    __slots__ = ("id", "bbox_xyxy", "score", "class_id", "class_name", "ts")

    def __init__(self, id: int, bbox: np.ndarray, score: float, class_id: int, class_name: str, ts: float):
        self.id = id
        self.bbox_xyxy = np.asarray(bbox, dtype=np.float64)
        self.score = float(score)
        self.class_id = int(class_id)
        self.class_name = str(class_name)
        self.ts = ts


class BYTETracker:
    """
    ByteTrack: match high-score dets first, then low-score to remaining tracks;
    new tracks for unmatched high. Class/score from matched det, else keep previous.
    """

    def __init__(self, cfg: TrackerConfig, class_names: List[str]):
        self.cfg = cfg
        self.class_names = class_names
        self._next_id = 1            # track_id_classes（如 person）分配 1,2,3…
        self._next_vehicle_id = getattr(self.cfg, "vehicle_id_offset", 10000)  # 车辆等独立区间，与人 ID 不重复
        self._next_anon_id = -1     # 其他类别用 -1,-2,… 仅内部跟踪
        self._tracks: Dict[int, _TrackState] = {}
        self._lost: Dict[int, _TrackState] = {}

    def update(
        self,
        high: List[Tuple[np.ndarray, float, int]],
        low: List[Tuple[np.ndarray, float, int]],
        ts: float,
    ) -> List[Track]:
        """
        high/low: list of (bbox_xyxy, score, class_id). bbox_xyxy (4,) x1,y1,x2,y2.
        Returns list of Track (active this frame); class from matched det or previous.
        """
        th_high = self.cfg.track_high_th
        th_low = self.cfg.track_low_th
        match_thresh = self.cfg.match_thresh

        # Collect all active track states (current + lost for matching)
        # For lost tracks use expanded bbox so half-occluded person can still match
        lost_expand = getattr(self.cfg, "lost_bbox_expand", 1.2)
        all_track_ids = list(self._tracks.keys()) + list(self._lost.keys())
        if not all_track_ids:
            track_boxes = np.zeros((0, 4))
            track_ids = []
        else:
            states = {k: self._tracks.get(k) or self._lost[k] for k in all_track_ids}
            box_list = []
            for k in all_track_ids:
                st = states[k]
                if k in self._lost:
                    box_list.append(_expand_bbox(st.bbox_xyxy, lost_expand))
                else:
                    box_list.append(st.bbox_xyxy)
            track_boxes = np.array(box_list)
            track_ids = all_track_ids

        out_tracks: List[Track] = []
        matched_det = set()
        matched_track = set()

        # 1) Match high-score dets to tracks (class-aware: only match same class)
        if high and track_boxes.size:
            h_boxes = np.array([x[0] for x in high])
            h_scores = np.array([x[1] for x in high])
            h_cls = [x[2] for x in high]
            iou = _iou_matrix(h_boxes, track_boxes)
            for i in range(iou.shape[0]):
                for j in range(iou.shape[1]):
                    if h_cls[i] != states[track_ids[j]].class_id:
                        iou[i, j] = 0.0
            for _ in range(min(len(high), len(track_ids))):
                i, j = np.unravel_index(np.argmax(iou), iou.shape)
                if iou[i, j] < match_thresh:
                    break
                tid = track_ids[j]
                st = self._tracks.get(tid) or self._lost.get(tid)
                cid = h_cls[i]
                cname = self.class_names[cid] if 0 <= cid < len(self.class_names) else "unknown"
                st.bbox_xyxy = h_boxes[i]
                st.score = h_scores[i]
                st.class_id = cid
                st.class_name = cname
                st.ts = ts
                self._tracks[tid] = st
                if tid in self._lost:
                    del self._lost[tid]
                out_tracks.append(Track(id=tid, bbox_xyxy=st.bbox_xyxy, score=st.score, class_id=st.class_id, class_name=st.class_name))
                matched_det.add(i)
                matched_track.add(j)
                iou[i, :] = 0
                iou[:, j] = 0

        # 2) Match low-score dets to remaining tracks (class-aware)
        remaining_track_idx = [j for j in range(len(track_ids)) if j not in matched_track]
        remaining_track_boxes = track_boxes[remaining_track_idx] if remaining_track_idx else np.zeros((0, 4))
        if low and remaining_track_boxes.size:
            l_boxes = np.array([x[0] for x in low])
            l_scores = np.array([x[1] for x in low])
            l_cls = [x[2] for x in low]
            iou2 = _iou_matrix(l_boxes, remaining_track_boxes)
            for i in range(iou2.shape[0]):
                for jj in range(iou2.shape[1]):
                    j = remaining_track_idx[jj]
                    if l_cls[i] != states[track_ids[j]].class_id:
                        iou2[i, jj] = 0.0
            for _ in range(min(len(low), len(remaining_track_idx))):
                i, jj = np.unravel_index(np.argmax(iou2), iou2.shape)
                if iou2[i, jj] < match_thresh:
                    break
                j = remaining_track_idx[jj]
                tid = track_ids[j]
                st = self._tracks.get(tid) or self._lost.get(tid)
                cid = l_cls[i]
                cname = self.class_names[cid] if 0 <= cid < len(self.class_names) else "unknown"
                st.bbox_xyxy = l_boxes[i]
                st.score = l_scores[i]
                st.class_id = cid
                st.class_name = cname
                st.ts = ts
                self._tracks[tid] = st
                if tid in self._lost:
                    del self._lost[tid]
                out_tracks.append(Track(id=tid, bbox_xyxy=st.bbox_xyxy, score=st.score, class_id=st.class_id, class_name=st.class_name))
                matched_track.add(j)
                iou2[i, :] = 0
                iou2[:, jj] = 0

        # 3) Unmatched tracks -> move to lost (keep state for GC / re-match later)
        for j in range(len(track_ids)):
            if j in matched_track:
                continue
            tid = track_ids[j]
            st = self._tracks.get(tid) or self._lost.get(tid)
            self._lost[tid] = st
            if tid in self._tracks:
                del self._tracks[tid]

        # 3b) Remove from _lost if lost too long (reduces ID reuse and keeps same person same ID)
        lost_timeout_sec = getattr(self.cfg, "lost_timeout_sec", 2.0)
        for tid in list(self._lost.keys()):
            if (ts - self._lost[tid].ts) > lost_timeout_sec:
                del self._lost[tid]

        # 4) Unmatched high -> new tracks：所有类别都建 track；人用 1,2,3…，车辆用 10000+，其余用负 ID
        track_id_classes: Optional[Set[str]] = getattr(self.cfg, "track_id_classes", None)
        if track_id_classes is None:
            track_id_classes = {"person"}
        vehicle_id_classes: Optional[Set[str]] = getattr(self.cfg, "vehicle_id_classes", None)
        for i in range(len(high)):
            if i in matched_det:
                continue
            bbox, score, cid = high[i]
            cname = self.class_names[cid] if 0 <= cid < len(self.class_names) else "unknown"
            if cname in track_id_classes:
                tid = self._next_id
                self._next_id += 1
            elif vehicle_id_classes and cname in vehicle_id_classes:
                tid = self._next_vehicle_id
                self._next_vehicle_id += 1
            else:
                tid = self._next_anon_id
                self._next_anon_id -= 1
            st = _TrackState(tid, bbox, score, cid, cname, ts)
            self._tracks[tid] = st
            out_tracks.append(Track(id=tid, bbox_xyxy=st.bbox_xyxy, score=st.score, class_id=st.class_id, class_name=st.class_name))

        return out_tracks
