"""
Configuration contract: sources (RTSP + file mixed), per-stream cfg (multi-polygon ROI, switches),
tracker/system params. stream_id = source.uri.
"""
from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Tuple, Optional
import os

# Polygon: list of (x, y) points
Polygon = List[Tuple[float, float]]


@dataclass
class SourceConfig:
    """Single input source: RTSP or local file."""
    uri: str
    type: str  # "rtsp" | "file"
    enabled: bool = True

    @property
    def stream_id(self) -> str:
        return self.uri


@dataclass
class StreamConfig:
    """Per-stream business config: multi-polygon ROI, dwell/alarm classes, vehicle alarm, switches."""
    rois: List[Polygon] = field(default_factory=list)
    roi_point_mode: str = "bottom_center"  # "center" | "bottom_center"
    enable_dwell: bool = True
    dwell_classes: Set[str] = field(default_factory=set)
    enable_alarm: bool = True
    alarm_classes: Set[str] = field(default_factory=set)
    # Vehicle alarm: trigger when vehicle class visible >= vehicle_alarm_sec (e.g. 10s)
    vehicle_alarm_classes: Set[str] = field(default_factory=set)
    vehicle_alarm_sec: float = 10.0
    # 本路显示 ID/轨迹的类别；None 表示用 SystemConfig.display_id_trajectory_classes
    display_id_trajectory_classes: Optional[Set[str]] = None
    # 本路「动态事件」横幅：字号与最大宽度；None 表示用系统默认，可每路单独设置
    dynamic_event_banner_font_size: Optional[int] = None
    dynamic_event_banner_max_width: Optional[int] = None
    # 本路「车辆超过10s」横幅：字号与最大宽度；None 表示用系统默认
    vehicle_alarm_banner_font_size: Optional[int] = None
    vehicle_alarm_banner_max_width: Optional[int] = None
    # 本路「禁止长时间停留」横幅：字号与最大宽度；None 表示用系统默认，调小字号可缩小该列
    dwell_warning_banner_font_size: Optional[int] = None
    dwell_warning_banner_max_width: Optional[int] = None


@dataclass
class TrackerConfig:
    """ByteTrack thresholds (and optional extra params)."""
    track_high_th: float = 0.5
    track_low_th: float = 0.1
    match_thresh: float = 0.5   # 降低便于遮挡/半身时重新匹配同一人
    min_box_area: float = 10.0
    lost_timeout_sec: float = 4.0  # 跟丢超过此秒数才从 _lost 删除，给遮挡后复现留足时间
    lost_bbox_expand: float = 1.2   # 跟丢轨迹匹配时 bbox 从中心膨胀比例，便于半身遮挡后复匹配
    # 仅对这些类别分配显示用 ID 1,2,3…（如 person）
    track_id_classes: Optional[Set[str]] = None
    # 车辆等类别用独立正数区间，避免与人 ID 重复；None 表示不分配（沿用负的 anon ID）
    vehicle_id_classes: Optional[Set[str]] = None
    vehicle_id_offset: int = 10000  # 车辆 ID 从 10000 起，如 10001, 10002…


@dataclass
class SystemConfig:
    """Global pipeline params."""
    max_batch: int = 8
    max_wait_ms: float = 40.0
    lost_timeout_sec: float = 2.0
    max_state_sec: float = 300.0
    alarm_cooldown_sec: float = 30.0
    drop_policy: str = "drop_old"
    infer_queue_maxsize: int = 64
    # Display: show video window with bbox, ROI, alarm/dwell prompts
    enable_display: bool = True
    display_scale: float = 1.5  # 显示窗口缩放倍数（仅当 display_width/height 为 0 时生效）
    display_width: int = 0     # 视频窗口宽度（像素），>0 时与 display_height 一起限定窗口大小，保持比例
    display_height: int = 0    # 视频窗口高度（像素），0 表示不按宽高限制，用 display_scale
    dwell_warning_sec: float = 5.0  # show "滞留超时" when track in ROI longer than this; 0 = off
    # 绘制框/文字（run_realtime 可直接改这些参数）
    bbox_thickness: int = 2          # 目标框线粗细
    box_expand_px: int = 0           # 绘制时把框四周扩大多少像素（可为 0）
    label_font_scale: float = 0.5    # 目标标签字体缩放（OpenCV Hershey）
    label_thickness: int = 1         # 目标标签字体线宽
    banner_font_size: int = 22       # 顶部中文 banner 字号（PIL）
    chinese_font_path: str = os.path.join("model_data", "simhei.ttf")  # 中文字体（相对项目根目录）
    force_pil_text: bool = True      # True: banner 强制用 PIL 画中文，避免 OpenCV 乱码
    # 仅对这些类别显示 ID 和轨迹，None 表示 {"person"}
    display_id_trajectory_classes: Optional[Set[str]] = None
    # True 时启用角标区域（右上角禁区/禁现摘要仍由 show_corner_rules_summary 控制）
    show_corner_overlay: bool = True
    # True 时在画面左上角显示 ViT 状态（绿 normal / 红 steal 或 violent）；False 则不显示左上角该文字
    show_corner_vit_label: bool = False
    # True 时在右上角显示禁区/禁现物品摘要（如「禁现: cat」）；False 则不显示
    show_corner_rules_summary: bool = False
    # 动态事件横幅左侧预留像素，避免与左上角 ViT 文字重叠
    corner_reserve_left: int = 100
    # ViT 异常报警阈值（0~1）：仅当预测为 steal/violent 等且 pred_prob >= 此值时才显示红色报警，否则按 normal 显示绿色
    vit_anomaly_threshold: float = 0.7
    # 推理置信度阈值，None 表示使用 yolo.confidence；提高可减少误检（如误报为人的背景）
    infer_conf_thres: Optional[float] = None
    # RTSP reconnect
    rtsp_reconnect_delay_sec: float = 2.0
    rtsp_max_reconnect_attempts: int = 10


def build_stream_config_map(
    sources: List[SourceConfig],
    roi_config: Dict[str, List[Polygon]],
    dwell_classes_global: Set[str],
    alarm_classes_global: Set[str],
    enable_dwell_default: bool = True,
    enable_alarm_default: bool = True,
    vehicle_alarm_classes_global: Optional[Set[str]] = None,
    vehicle_alarm_sec: float = 10.0,
) -> Dict[str, StreamConfig]:
    """
    Build stream_cfg[stream_id] for each enabled source.
    roi_config[stream_id] = [poly0, poly1, ...]; missing keys => [].
    """
    out = {}
    v_alc = set(vehicle_alarm_classes_global) if vehicle_alarm_classes_global is not None else set()
    for src in sources:
        if not src.enabled:
            continue
        sid = src.stream_id
        rois = roi_config.get(sid, [])
        out[sid] = StreamConfig(
            rois=rois,
            roi_point_mode="bottom_center",
            enable_dwell=enable_dwell_default,
            dwell_classes=set(dwell_classes_global),
            enable_alarm=enable_alarm_default,
            alarm_classes=set(alarm_classes_global),
            vehicle_alarm_classes=v_alc,
            vehicle_alarm_sec=vehicle_alarm_sec,
        )
    return out
