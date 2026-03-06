"""
ROI geometry: point_in_polygon (ray casting), bbox_anchor_point (center / bottom_center),
roi_membership for multiple polygons.
"""
import numpy as np
from typing import List, Tuple

# Polygon = list of (x, y)
Polygon = List[Tuple[float, float]]


def point_in_polygon(point_xy: Tuple[float, float], polygon: Polygon) -> bool:
    """Ray casting: point inside polygon."""
    x, y = point_xy
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi):
            inside = not inside
        j = i
    return inside


def bbox_anchor_point(bbox_xyxy, mode: str):
    """bbox (4,) x1,y1,x2,y2. mode: 'center' | 'bottom_center'."""
    x1, y1, x2, y2 = bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3]
    if mode == "center":
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    if mode == "bottom_center":
        return ((x1 + x2) / 2.0, float(y2))
    return ((x1 + x2) / 2.0, float(y2))


def roi_membership(bbox_xyxy, rois: List[Polygon], mode: str = "bottom_center") -> List[int]:
    """Return list of roi_id for which anchor point is inside polygon. Can hit multiple ROIs."""
    p = bbox_anchor_point(bbox_xyxy, mode)
    inside = []
    for roi_id, poly in enumerate(rois):
        if point_in_polygon(p, poly):
            inside.append(roi_id)
    return inside
