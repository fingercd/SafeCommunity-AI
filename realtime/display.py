"""
Draw ROI, tracks (bbox + id), alarm and dwell-warning prompts on frame; show via cv2.imshow.
Pipeline calls draw_frame then waitKey(1) separately.
"""
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np
import cv2
import os

from PIL import Image, ImageDraw, ImageFont

from .types import Track
from .config import StreamConfig

# Font cache by (path, size) to avoid loading ImageFont.truetype on every frame
_font_cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}


def _get_font(font_path: str, font_size: int) -> ImageFont.FreeTypeFont:
    """Return cached PIL font for (path, size); avoids repeated file I/O."""
    key = (font_path, font_size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(font_path, font_size)
    return _font_cache[key]


def _win_name(stream_index: int) -> str:
    return f"Realtime-{stream_index}"


def draw_frame(
    stream_id: str,
    stream_index: int,
    frame_bgr: np.ndarray,
    tracks: List[Track],
    stream_cfg: StreamConfig,
    roi_start: Dict[Tuple[int, int], float],
    track_history: Dict[int, List[Tuple[float, float]]],
    vehicle_alarm_active: List[Tuple[int, float]],
    ts: float,
    dwell_warning_sec: float,
    enable_display: bool,
    display_scale: float = 1.0,
    display_width: int = 0,
    display_height: int = 0,
    bbox_thickness: int = 2,
    box_expand_px: int = 0,
    label_font_scale: float = 0.5,
    label_thickness: int = 1,
    banner_font_size: int = 22,
    chinese_font_path: str = os.path.join("model_data", "simhei.ttf"),
    force_pil_text: bool = True,
    display_id_trajectory_classes: Optional[Set[str]] = None,
    vit_event: Optional[Dict[str, Any]] = None,
    show_corner_overlay: bool = True,
    vit_anomaly_threshold: float = 0.7,
    show_corner_rules_summary: bool = False,
    corner_reserve_left: int = 100,
    show_corner_vit_label: bool = False,
) -> None:
    """
    Draw ROI (thin red), track bbox + id, trajectories, alarm/dwell/vehicle banners, and optional ViT dynamic event. Optionally imshow.
    vit_event: optional dict with pred_label, pred_prob, ranking_score (from KnownEventRuntime).
    show_corner_overlay: if True, draw ViT label at top-left (green normal / red steal|violent).
    show_corner_rules_summary: if True, draw 禁区/禁现 summary at top-right; False to hide.
    corner_reserve_left: left margin (px) for dynamic event banner so it does not overlap the corner ViT text.
    vit_anomaly_threshold: only show red alarm for non-normal when pred_prob >= this (e.g. 0.7); below threshold show green normal.
    """
    frame = frame_bgr
    h, w = frame.shape[:2]
    vehicle_alarm_track_ids = {tid for tid, _ in vehicle_alarm_active}
    show_id_traj = display_id_trajectory_classes if display_id_trajectory_classes is not None else {"person"}
    track_by_id = {t.id: t for t in tracks}

    # 1. Trajectories (only for classes that show ID/trajectory, e.g. person)
    for tid, points in track_history.items():
        if len(points) < 2:
            continue
        if tid not in track_by_id or track_by_id[tid].class_name not in show_id_traj:
            continue
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 255), thickness=1)

    # 2. ROI: thin red polylines
    if stream_cfg.rois:
        for poly in stream_cfg.rois:
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=1)

    # 3. Current-frame alarm track ids (for red bbox + banner)
    alarm_track_ids = {t.id for t in tracks if t.class_name in stream_cfg.alarm_classes}
    # Dwell-warning: (track_id, roi_id, duration_sec) for roi_start entries over threshold
    dwell_warnings: List[Tuple[int, int, float]] = []
    if dwell_warning_sec > 0 and stream_cfg.enable_dwell and stream_cfg.dwell_classes:
        for (roi_id, track_id), enter_ts in roi_start.items():
            duration = ts - enter_ts
            if duration >= dwell_warning_sec:
                dwell_warnings.append((track_id, roi_id, duration))
    dwell_track_ids = {tid for tid, _, _ in dwell_warnings}

    # 4. Track bboxes and IDs (only person gets #id in label; others get class name only)
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.bbox_xyxy)
        if box_expand_px and box_expand_px > 0:
            x1 = max(0, x1 - box_expand_px)
            y1 = max(0, y1 - box_expand_px)
            x2 = min(w - 1, x2 + box_expand_px)
            y2 = min(h - 1, y2 + box_expand_px)
        if t.id in alarm_track_ids:
            color = (0, 0, 255)  # BGR red: alarm (cat/dog)
        elif t.id in vehicle_alarm_track_ids:
            color = (0, 0, 255)  # BGR red: vehicle >10s
        elif t.id in dwell_track_ids:
            color = (0, 165, 255)  # BGR orange: dwell warning
        else:
            color = (0, 255, 0)  # BGR green: normal
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, int(bbox_thickness))
        if t.class_name in show_id_traj:
            label = f"#{t.id} {t.class_name} {t.score:.2f}"
        else:
            label = t.class_name
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, float(label_font_scale), int(label_thickness))
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(
            frame, label, (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX, float(label_font_scale), (255, 255, 255), int(label_thickness), cv2.LINE_AA,
        )

    def _get_pil_text_size(text: str, font_size: int) -> Tuple[int, int]:
        """用 PIL 字体测量中文/英文混合文本的宽高（使用缓存字体，避免每帧加载）。"""
        try:
            font = _get_font(chinese_font_path, font_size)
            bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=font)
            return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
        except Exception:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            return tw, th

    def _truncate_text_to_fit(text: str, font_size: int, max_width: int, suffix: str = "…") -> Tuple[str, int, int]:
        """若文字宽度超过 max_width 则截断并加后缀，返回 (显示文本, 宽, 高)。"""
        tw, th = _get_pil_text_size(text, font_size)
        if tw <= max_width:
            return text, tw, th
        try:
            suffix_w, _ = _get_pil_text_size(suffix, font_size)
            remain = max_width - suffix_w
            if remain <= 0:
                return suffix, suffix_w, th
            # 二分或逐字缩短，直到 (s[:n] + suffix) 宽度 <= max_width
            lo, hi = 0, len(text)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                t = text[:mid] + suffix
                tw_t, _ = _get_pil_text_size(t, font_size)
                if tw_t <= max_width:
                    lo = mid
                else:
                    hi = mid - 1
            out = text[:lo] + suffix if lo < len(text) else text
            tw_out, th_out = _get_pil_text_size(out, font_size)
            return out, tw_out, th_out
        except Exception:
            return text, tw, th

    # Collect all Chinese text draws; do one BGR->PIL->BGR at the end to avoid 5-8 full-frame conversions
    text_draws: List[Tuple[Tuple[int, int], str, int, Tuple[int, int, int]]] = []
    pad = 4
    max_bar_width = max(w - 2 * pad, 100)  # 警告栏最大可用宽度，避免文字超出画面

    # 5. Alarm banner (猫狗等不该出现)
    alarm_bar_bottom = 0
    if alarm_track_ids:
        names = sorted({track_by_id[tid].class_name for tid in alarm_track_ids})
        text = "警告: 检测到 " + ",".join(names)
        if force_pil_text:
            display_text, tw, th = _truncate_text_to_fit(text, banner_font_size, max_bar_width)
        else:
            display_text = text
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        bar_w = min(tw + 2 * pad, w)
        alarm_bar_bottom = th + 2 * pad
        cv2.rectangle(frame, (0, 0), (bar_w, alarm_bar_bottom), (0, 0, 255), -1)
        if force_pil_text:
            text_draws.append(((pad, pad), display_text, banner_font_size, (255, 255, 255)))
        else:
            cv2.putText(
                frame, display_text, (pad, th + pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )

    # 6. Vehicle alarm banner (车辆超过10s)，每路可单独设置字号与最大宽度
    if vehicle_alarm_active:
        parts = [f"ID:{tid} {dur:.1f}s" for tid, dur in vehicle_alarm_active[:5]]
        if len(vehicle_alarm_active) > 5:
            parts.append("...")
        text = "车辆超过10s 报警 " + " | ".join(parts)
        vh_font = getattr(stream_cfg, "vehicle_alarm_banner_font_size", None) or banner_font_size
        vh_max_w = getattr(stream_cfg, "vehicle_alarm_banner_max_width", None)
        vh_max_bar = (vh_max_w if vh_max_w is not None and vh_max_w > 0 else max_bar_width)
        if force_pil_text:
            display_text, tw, th = _truncate_text_to_fit(text, vh_font, vh_max_bar)
        else:
            display_text = text
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        bar_w = min(tw + 2 * pad, w)
        y0_v = alarm_bar_bottom
        bar_h = th + 2 * pad
        cv2.rectangle(frame, (0, y0_v), (bar_w, y0_v + bar_h), (0, 0, 255), -1)
        if force_pil_text:
            text_draws.append(((pad, y0_v + pad), display_text, vh_font, (255, 255, 255)))
        else:
            cv2.putText(
                frame, display_text, (pad, y0_v + th + pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
            )
        alarm_bar_bottom = y0_v + bar_h

    # 7. Dwell-warning banner (禁止长时间停留)，每路可单独设置字号与最大宽度以调小该列
    next_banner_top = alarm_bar_bottom
    if dwell_warnings:
        parts = []
        for tid, roi_id, dur in dwell_warnings[:5]:
            parts.append(f"ID:{tid} 滞留{dur:.1f}s")
        if len(dwell_warnings) > 5:
            parts.append("...")
        text = "禁止长时间停留 " + " | ".join(parts)
        dw_font = getattr(stream_cfg, "dwell_warning_banner_font_size", None) or banner_font_size
        dw_max_w = getattr(stream_cfg, "dwell_warning_banner_max_width", None)
        dw_max_bar = (dw_max_w if dw_max_w is not None and dw_max_w > 0 else max_bar_width)
        if force_pil_text:
            display_text, tw, th = _truncate_text_to_fit(text, dw_font, dw_max_bar)
        else:
            display_text = text
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        bar_w = min(tw + 2 * pad, w)
        y0 = alarm_bar_bottom
        bar_h = th + 2 * pad
        next_banner_top = y0 + bar_h
        cv2.rectangle(frame, (0, y0), (bar_w, y0 + bar_h), (0, 165, 255), -1)
        if force_pil_text:
            text_draws.append(((pad, y0 + pad), display_text, dw_font, (255, 255, 255)))
        else:
            cv2.putText(
                frame, display_text, (pad, y0 + th + pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
            )

    # 8. ViT 动态事件 banner（与角标一致：仅当 steal/violent 且 pred_prob >= 阈值时才显示异常；预留左侧空间避免与左上角重叠；每路可单独设置字号与宽度）
    if vit_event:
        pred_label = (vit_event.get("pred_label") or "").strip()
        pred_prob = float(vit_event.get("pred_prob") or 0.0)
        ranking_score = float(vit_event.get("ranking_score") or 0.0)
        pl_lower = pred_label.lower()
        if pl_lower == "normal":
            show_label, show_prob = "normal", pred_prob
        elif pred_prob < vit_anomaly_threshold:
            show_label, show_prob = "normal", pred_prob  # 低于阈值不显示异常类别
        else:
            show_label, show_prob = pred_label, pred_prob
        text = f"动态事件: {show_label} {show_prob:.2f}" + (f" rank={ranking_score:.2f}" if ranking_score > 0 else "")
        dyn_font = getattr(stream_cfg, "dynamic_event_banner_font_size", None) or banner_font_size
        dyn_max_w = getattr(stream_cfg, "dynamic_event_banner_max_width", None)
        effective_max_bar = (dyn_max_w if dyn_max_w is not None and dyn_max_w > 0 else max_bar_width)
        effective_max_bar = min(effective_max_bar, max(w - corner_reserve_left - 2 * pad, 100))
        if force_pil_text:
            display_text, tw, th = _truncate_text_to_fit(text, dyn_font, effective_max_bar)
        else:
            display_text = text
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        bar_w = min(tw + 2 * pad, w - corner_reserve_left)
        y0 = next_banner_top
        bar_h = th + 2 * pad
        x0 = corner_reserve_left
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (128, 128, 255), -1)
        if force_pil_text:
            text_draws.append(((x0 + pad, y0 + pad), display_text, dyn_font, (255, 255, 255)))
        else:
            cv2.putText(
                frame, display_text, (x0 + pad, y0 + th + pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
            )

    # 9. 角标覆盖：左上 ViT 状态（绿 normal / 红 steal|violent），右上 禁区/禁现摘要（仅当 show_corner_overlay 为 True）
    corner_pad = 8
    corner_font_size = max(banner_font_size, 24)
    if show_corner_overlay:
        # 左上角：仅当 show_corner_vit_label 为 True 时绘制 ViT 状态（绿 normal / 红 steal|violent）
        if show_corner_vit_label and vit_event:
            pred_label = (vit_event.get("pred_label") or "").strip()
            pred_prob = float(vit_event.get("pred_prob") or 0.0)
            pl_lower = pred_label.lower()
            if pl_lower == "normal":
                vit_text = "normal"
                vit_color = (0, 255, 0)  # BGR green，normal 无阈值
            else:
                # steal/violent 等：超过阈值才显示类别和概率，否则只显示 normal
                if pred_prob < vit_anomaly_threshold:
                    vit_text = "normal"
                    vit_color = (0, 255, 0)  # BGR green
                else:
                    if "steal" in pl_lower or "盗窃" in pred_label:
                        vit_text = f"steal {pred_prob:.2f}"
                    elif "violent" in pl_lower or "暴力" in pred_label:
                        vit_text = f"violent {pred_prob:.2f}"
                    else:
                        vit_text = f"{pred_label or 'unknown'} {pred_prob:.2f}"
                    vit_color = (0, 0, 255)  # BGR red
            text_draws.append(((corner_pad, corner_pad), vit_text, corner_font_size, vit_color))
        # 右上角：仅当 show_corner_rules_summary 为 True 时显示禁区/禁现摘要（默认不显示）
        right_lines: List[str] = []
        if show_corner_rules_summary:
            if stream_cfg.rois:
                right_lines.append(f"禁区: {len(stream_cfg.rois)}个")
            if stream_cfg.alarm_classes:
                right_lines.append("禁现: " + ", ".join(sorted(stream_cfg.alarm_classes)))
        if right_lines:
            try:
                font = _get_font(chinese_font_path, corner_font_size)
                y_right = corner_pad
                for line in right_lines:
                    bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), line, font=font)
                    tw = int(bbox[2] - bbox[0])
                    th_line = int(bbox[3] - bbox[1])
                    x_right = w - corner_pad - tw
                    text_draws.append(((x_right, y_right), line, corner_font_size, (255, 255, 255)))
                    y_right += th_line + 4
            except Exception:
                x_right = w - corner_pad - 80
                y_off = corner_pad
                for line in right_lines[:2]:
                    text_draws.append(((x_right, y_off), line, corner_font_size, (255, 255, 255)))
                    y_off += 22

    # Single BGR->PIL->BGR pass for all Chinese text (replaces 5-8 per-banner conversions)
    if text_draws and force_pil_text:
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil)
            for (xy, text, font_size, color_bgr) in text_draws:
                font = _get_font(chinese_font_path, font_size)
                color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
                draw.text(xy, text, font=font, fill=color_rgb)
            frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            for (xy, text, _fs, color_bgr) in text_draws:
                cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2, cv2.LINE_AA)

    if enable_display:
        win = _win_name(stream_index)
        h, w = frame.shape[:2]
        if display_width > 0 and display_height > 0:
            # 按指定视频窗口宽高缩放，保持比例（整幅画面放进窗口内）
            scale = min(display_width / w, display_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            if new_w != w or new_h != h:
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        elif display_scale != 1.0:
            new_w, new_h = int(w * display_scale), int(h * display_scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(win, frame)
