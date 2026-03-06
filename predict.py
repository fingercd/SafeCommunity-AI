# -*- coding: utf-8 -*-
"""
统一监控入口：YOLO 物体检测 + 规则引擎（禁区/滞留/禁现物品/车辆禁区）+ ViT 动态事件分类（normal/盗窃/暴力冲突）。
单路或多路 RTSP / 本地视频混合输入；所有可调参数在下方「配置区」集中修改并配有中文注释。
运行方式：在 moniter 目录下执行  python predict.py
"""
from __future__ import annotations

import logging
import os
import sys
import threading

# 保证可导入 yolo 与 Vit 子项目（以 moniter 为项目根）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_DIR = os.path.join(BASE_DIR, "yolo")
VIT_DIR = os.path.join(BASE_DIR, "Vit")
if YOLO_DIR not in sys.path:
    sys.path.insert(0, YOLO_DIR)
if VIT_DIR not in sys.path:
    sys.path.insert(0, VIT_DIR)

from yolo import YOLO
from realtime.config import (
    SourceConfig,
    StreamConfig,
    SystemConfig,
    TrackerConfig,
)
from realtime.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("predict")


# =============================================================================
# 配置区：所有重要可调参数（按功能分组，中文注释）
# =============================================================================

# ------------------------------ 1. 输入源（支持单路或多路，RTSP 与本地视频可混合）----------------------
# 每路一个 SourceConfig：uri=流地址或文件路径，type="rtsp"|"file"，enabled=True/False
# 修改 _sources() 的返回值即可切换单路/多路、本地视频/RTSP。
def _sources():
    """返回实际使用的源列表；修改此处实现单路/多路、本地视频/RTSP 混合。"""
    # 单路本地视频（请改为实际存在的视频路径，否则运行时会报错）
    return [
        SourceConfig(uri=r"C:\Users\Administrator\Desktop\Vit\lab_dataset\raw_videos\violent conflict\Fighting\Fighting038_x264.mp4", type="file", enabled=True),
        SourceConfig(uri=r"C:\Users\Administrator\Desktop\Vit\lab_dataset\raw_videos\violent conflict\Fighting\Fighting036_x264.mp4", type="file", enabled=True),
    ]


# ------------------------------ 2. 每路规则配置（禁区、滞留、禁现物品、车辆禁区）----------------------
# 键为 stream_id（即 SourceConfig.uri），值与 SOURCES 中 enabled 的源一一对应。
# 多路时可按 sid（即 uri）单独设置：不同路不同 rois、alarm_classes 等。
# 禁区显示：仅当该路 rois 非空且至少有一个多边形时，画面上才会画出红色禁区线；rois=[] 则不显示禁区。
# rois：该路禁区多边形列表，每个多边形为 [(x1,y1), (x2,y2), ...]，像素坐标与原图分辨率一致（如 1920x1080）。
# roi_point_mode：判定“是否在禁区内”的锚点："center" 框中心；"bottom_center" 框底边中点（推荐）。
# enable_dwell：是否统计该路禁区停留时长；dwell_classes：仅对这些类别统计（如 person、backpack）。
# enable_alarm：是否启用“禁现物品”报警；alarm_classes：出现即报警的类别（如 cat、dog）。
# vehicle_alarm_classes：车辆类在禁区内超过 vehicle_alarm_sec 秒则报警；设为 set() 则关闭车辆报警。
# display_id_trajectory_classes：该路显示 ID 与轨迹的类别；None 表示用系统默认（如 person）。
def _stream_config_map(sources):
    """根据当前 sources 生成每路规则配置；按 sid（uri）单独设置每路的禁区与禁现物品。"""
    cfgs = {}
    enabled_list = [s for s in sources if s.enabled]
    for i, s in enumerate(enabled_list):
        sid = s.uri
        # 按路单独配置：第 1 路、第 2 路可设置不同 rois 与 alarm_classes（也可用 sid 字符串匹配）
        if i == 0:
            # 第一路：禁区 + 禁现物品；可单独设置本路「动态事件」「车辆超过10s」横幅字号与宽度
            # 禁区：取消下面 rois 的注释并按实际分辨率改坐标即可显示红色禁区；每路可设不同多边形
            cfgs[sid] = StreamConfig(
                rois=[
                    [(180, 10), (180, 100), (280, 100), (280, 10)],
                ],
                roi_point_mode="bottom_center",
                enable_dwell=True,
                dwell_classes={"person"},
                enable_alarm=True,
                alarm_classes={"cat", "dog"},  # 第一路禁现：出现猫/狗即报警
                vehicle_alarm_classes={"car", "truck", "bus", "motorcycle"},
                vehicle_alarm_sec=10.0,
                display_id_trajectory_classes={"car", "truck", "bus", "motorcycle"},
                dynamic_event_banner_font_size=12,
                dynamic_event_banner_max_width=1200,
                vehicle_alarm_banner_font_size=8,
                vehicle_alarm_banner_max_width=1200,
                dwell_warning_banner_font_size=12,   # 「禁止长时间停留」字号，改小可调小该列（如 12、10）
                dwell_warning_banner_max_width=1200,
            )
        elif i == 1:
            # 第二路：可设置不同禁区与禁现、动态事件与车辆报警横幅大小；禁区同样在 rois 里填多边形
            cfgs[sid] = StreamConfig(
                rois=[[(50,50),(50,200),(200,200),(200,50)]],  # 如 [(0,0),(400,0),(400,300),(0,300)] 可显示该路禁区
                roi_point_mode="bottom_center",
                enable_dwell=True,
                dwell_classes={"person"},
                enable_alarm=True,
                alarm_classes={"cat"},
                vehicle_alarm_classes={"car", "truck", "bus", "motorcycle"},
                vehicle_alarm_sec=10.0,
                display_id_trajectory_classes={"person"},
                dynamic_event_banner_font_size=12,
                dynamic_event_banner_max_width=1200,
                vehicle_alarm_banner_font_size=8,
                vehicle_alarm_banner_max_width=1200,
                dwell_warning_banner_font_size=12,
                dwell_warning_banner_max_width=1200,
            )
        else:
            # 第三路及以后：默认配置
            cfgs[sid] = StreamConfig(
                rois=[],
                roi_point_mode="bottom_center",
                enable_dwell=True,
                dwell_classes={"person"},
                enable_alarm=True,
                alarm_classes={"cat", "dog"},
                vehicle_alarm_classes={"car", "truck", "bus", "motorcycle"},
                vehicle_alarm_sec=10.0,
                display_id_trajectory_classes={"person"},
            )
    return cfgs


# ------------------------------ 3. YOLO 模型（所有流通用）----------------------
# 权重与类别文件路径（建议使用相对 moniter 的路径或 BASE_DIR 绝对路径）。
YOLO_MODEL_PATH = os.path.join(YOLO_DIR, "logs", "best_epoch_weights.pth")
YOLO_CLASSES_PATH = os.path.join(YOLO_DIR, "Class", "coco_classes.txt")
# YOLO 推理置信度阈值，高于此才保留检测框；可减小以减少误检、增大以降低漏检。
YOLO_CONFIDENCE = 0.3


# ------------------------------ 4. ViT 动态事件（所有流通用，必启用）----------------------
# 已知分类器权重路径（训练得到的 checkpoint_best.pt）；不使用 KMeans/OCSVM 开放集。
VIT_KNOWN_CHECKPOINT = os.path.join(BASE_DIR, "Vit", "lab_dataset", "derived", "known_classifier", "checkpoint_best.pt")
# ViT 采样与滑窗（与训练时一致效果更好）：clip 帧数、帧步长、每隔多少采样帧做一次推理。
VIT_CLIP_LEN = 16
VIT_FRAME_STRIDE = 2
VIT_WINDOW_STRIDE = 4
VIT_ENCODER_MODEL = "MCG-NJU/videomae-base"
VIT_USE_HALF = True


# ------------------------------ 5. 系统与显示（全局）----------------------
# 推理批大小、队列长度、显示窗口尺寸等；display 相关影响画面与 banner。
# show_corner_overlay=True 时启用角标区域。
# show_corner_vit_label=False 时不显示左上角绿的 normal/红 steal|violent；改为 True 则显示。
# show_corner_rules_summary=False 时不显示右上角「禁现: cat」等；改为 True 则显示禁区/禁现摘要。
# corner_reserve_left：动态事件横幅左侧预留像素，避免与左上角文字重叠。
# vit_anomaly_threshold：ViT 异常（steal/violent 等）仅当 pred_prob >= 此值时才显示红色报警，否则按 normal 显示绿色；例如 0.7 表示超过 0.7 才报警。
SYSTEM_CONFIG = SystemConfig(
    max_batch=4,
    max_wait_ms=50.0,
    lost_timeout_sec=2.0,
    max_state_sec=300.0,
    alarm_cooldown_sec=30.0,
    infer_queue_maxsize=64,
    enable_display=True,
    display_scale=1.5,
    display_width=1280,
    display_height=720,
    dwell_warning_sec=5.0,
    infer_conf_thres=YOLO_CONFIDENCE,
    bbox_thickness=2,
    box_expand_px=0,
    label_font_scale=0.5,
    label_thickness=1,
    banner_font_size=22,
    chinese_font_path=os.path.join(YOLO_DIR, "model_data", "simhei.ttf")
    if os.path.exists(os.path.join(YOLO_DIR, "model_data", "simhei.ttf"))
    else os.path.join(YOLO_DIR, "Class", "simhei.ttf"),
    force_pil_text=True,
    display_id_trajectory_classes={"person"},
    show_corner_overlay=True,
    show_corner_vit_label=False,
    show_corner_rules_summary=False,
    corner_reserve_left=100,
    vit_anomaly_threshold=0.55,
    rtsp_reconnect_delay_sec=2.0,
    rtsp_max_reconnect_attempts=10,
)

# ------------------------------ 6. ByteTrack 跟踪（全局）----------------------
# 高/低置信度阈值、匹配 IoU、跟丢超时等；影响 ID 稳定性与滞留统计准确性。
# vehicle_id_offset=10000：车辆 ID 从 10001 起分配，与人的 ID 1,2,3… 区分开，避免同屏重复；若希望车辆也从 1 起可改小（如 0），但可能与人的 ID 冲突。
TRACKER_CONFIG = TrackerConfig(
    track_high_th=0.5,
    track_low_th=0.1,
    match_thresh=0.5,
    lost_timeout_sec=4.0,
    lost_bbox_expand=1.2,
    track_id_classes={"person"},
    vehicle_id_classes={"car", "truck", "bus", "motorcycle"},
    vehicle_id_offset=10000,
)


# =============================================================================
# 主流程：加载 YOLO + ViT，构建回调，运行统一管线
# =============================================================================

def save_duration(stream_id, roi_id, track_id, cls, enter_ts, end_ts, duration):
    """禁区停留回调：离开禁区或超时结算时调用；可改为写库、写文件等。"""
    logger.info(
        "DWELL stream_id=%s roi_id=%d track_id=%d class=%s enter=%.2f end=%.2f duration=%.2fs",
        stream_id, roi_id, track_id, cls, enter_ts, end_ts, duration,
    )


def main():
    sources = _sources()
    if not sources:
        logger.error("未配置任何输入源，请修改 SOURCES 或 _sources()")
        return

    stream_cfg = _stream_config_map(sources)
    sys_cfg = SYSTEM_CONFIG
    tracker_cfg = TRACKER_CONFIG

    # YOLO（所有流通用）
    yolo = YOLO(
        model_path=YOLO_MODEL_PATH,
        classes_path=YOLO_CLASSES_PATH,
        confidence=YOLO_CONFIDENCE,
    )
    class_names = yolo.class_names

    # ViT 动态事件（仅已知分类，必启用）
    if not os.path.isfile(VIT_KNOWN_CHECKPOINT):
        raise FileNotFoundError(
            f"ViT 已知分类器权重不存在: {VIT_KNOWN_CHECKPOINT}，请先训练或修改 VIT_KNOWN_CHECKPOINT。"
        )
    try:
        from lab_anomaly.infer.known_event_runtime import KnownEventRuntime
    except ImportError as e:
        logger.error("无法导入 ViT 模块，请确认 Vit 目录与依赖已就绪: %s", e)
        raise
    vit_runtime = KnownEventRuntime(
        known_checkpoint=VIT_KNOWN_CHECKPOINT,
        device=None,
        clip_len=VIT_CLIP_LEN,
        frame_stride=VIT_FRAME_STRIDE,
        window_stride=VIT_WINDOW_STRIDE,
        encoder_model_name=VIT_ENCODER_MODEL,
        use_half=VIT_USE_HALF,
    )

    def on_frame_after_yolo(stream_id: str, frame_bgr, ts: float, state: dict):
        """每帧 YOLO/规则处理后：送入 ViT 缓冲（传 BGR 引用，ViT 内部做一次 cvtColor 入缓冲，避免重复 copy）。"""
        vit_runtime.add_frame(stream_id, frame_bgr, state=state)

    stop = threading.Event()
    run_pipeline(
        sources=sources,
        stream_cfg=stream_cfg,
        sys_cfg=sys_cfg,
        tracker_cfg=tracker_cfg,
        yolo=yolo,
        class_names=class_names,
        save_duration=save_duration,
        log_stream_end=lambda sid: logger.info("Stream ended: %s", sid),
        stop_event=stop,
        on_frame_after_yolo=on_frame_after_yolo,
    )


if __name__ == "__main__":
    main()
