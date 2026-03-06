# 实时多路视频分析管道（RTSP / 本地视频 + ByteTrack + ROI 停留 + 报警）

本模块在**不修改现有 YOLO 模型**的前提下，基于其检测输出（boxes / classes / scores），实现：**统一输入源（RTSP 或本地视频文件，可混合多路）→ 批量推理 → 每路独立 ByteTrack 跟踪 → 多多边形 ROI 停留时间统计 → 指定类别报警（声音 + 日志）**。适用于 10+ 路 RTSP 或本地视频的实时分析。

---

## 快速开始（一句话）

- **推荐**：在 **moniter 项目根目录** 执行 `python predict.py`，可同时启用 YOLO 检测、规则引擎（禁区/滞留/禁现物品）与 ViT 动态事件分类；所有可调参数与规则均在 `moniter/predict.py` 顶部配置区修改。
- 仅用 YOLO+规则（不用 ViT）时，可在 yolo 目录执行：`python run_realtime.py`（需先修改 `run_realtime.py` 中的 `sources`、`stream_cfg` 等）。

---

## 与主工程关系

- **依赖**：复用根目录的 `yolo.py`（YOLO 类）、`nets/`、`utils/utils_bbox.py`（DecodeBox、NMS）、`utils/utils.py`（resize_image 等）、`Class/*.txt`（类别名）。不修改任何已有模型或训练代码。
- **入口**：主工程原有 `predict.py` 仍用于单图/视频/目录预测；本模块通过 `run_realtime.py` 或自定义脚本调用 `realtime.pipeline.run_pipeline`，用于多路流 + 跟踪 + ROI + 报警。
- **输出**：检测与跟踪结果通过回调（`save_duration`、报警日志）输出，不写回图像文件；若需可视化，可在回调或 pipeline 内自行取帧绘制后保存或推流。

---

## 目录

- [快速开始](#快速开始一句话)
- [与主工程关系](#与主工程关系)
1. [概述](#1-概述)
2. [架构与数据流](#2-架构与数据流)
3. [依赖与环境](#3-依赖与环境)
4. [目录与文件说明](#4-目录与文件说明)
5. [配置详解](#5-配置详解)
6. [输入源与事件](#6-输入源与事件)
7. [各模块说明](#7-各模块说明)
8. [运行方式](#8-运行方式)
9. [ROI 与坐标](#9-roi-与坐标)
10. [验证清单](#10-验证清单)
11. [回调与扩展](#11-回调与扩展)
12. [性能与调参](#12-性能与调参)
13. [常见问题与排查](#13-常见问题与排查)
14. [安全与注意](#14-安全与注意)
- [版本与兼容](#版本与兼容)
- [参考](#参考)

---

## 1. 概述

- **目标**：对多路视频流（RTSP 或本地文件）做目标检测（YOLO）、多目标跟踪（ByteTrack）、ROI 内停留时间统计、指定类别报警。
- **约束**：
  - 必须使用 **ByteTrack**（非 BoT-SORT），以在遮挡场景下保持 ID 稳定。
  - 每路流**独立**跟踪器与状态，`stream_id = source.uri`，ID 不跨流混淆。
  - 不修改 YOLO 模型结构或训练代码，仅复用其推理接口。
- **输出**：停留事件通过 `save_duration` 回调输出；报警通过 `trigger_sound` + `write_alarm_log`；流结束通过 `log_stream_end`。

---

## 2. 架构与数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 输入层 (Input Layer)                                                         │
│   RTSP 源 ──→ 解码线程 (断流重连、drop_old) ──┐                              │
│   文件源 ──→ 解码线程 (EOF→EndOfStream) ──────┼──→ InferQueue (stream_id,    │
│                                               │      frame, ts)              │
└───────────────────────────────────────────────┴──────────────────────────────┘
                                                          │
┌─────────────────────────────────────────────────────────┼────────────────────┐
│ 推理层 (Infer Layer)                                     ▼                    │
│   BatchAggregator(max_batch, max_wait_ms) ──→ YOLO 批量推理                   │
│   ──→ 每帧 DetectionResult(boxes_xyxy, class_id, score) 按 stream_id 分发     │
└─────────────────────────────────────────────────────────┼────────────────────┘
                                                          │
┌─────────────────────────────────────────────────────────┼────────────────────┐
│ 每流业务层 (Per-Stream Layer)                             ▼                    │
│   BYTETracker(stream_id).update(high, low, ts) ──→ list[Track]                 │
│   ──→ update_roi_dwell (进入/离开 ROI，save_duration)                          │
│   ──→ update_alarm (alarm_classes 过滤、冷却、声音+日志)                       │
│   ──→ garbage_collect (丢失超时结算、状态清理)                                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

- **采集线程**：每个 `SourceConfig` 一个线程，持续往 `infer_queue` 放入 `FrameEvent` 或（仅文件源）`EndOfStreamEvent`。
- **主线程**：从队列取批 → 先处理本批中的 EOS（flush + log）→ YOLO 批量推理 → 按 `stream_id` 调用 ByteTrack → 停留 → 报警 → GC。

---

## 3. 依赖与环境

- **Python**：3.6+（建议 3.8+）。
- **项目根依赖**：与本仓库主工程一致（`torch`, `torchvision`, `opencv-python`, `numpy`, `Pillow` 等），见根目录 `requirements.txt`。
- **本模块**：仅使用标准库 + 上述依赖，**无额外 pip 包**（ByteTrack 为自实现封装）。
- **运行路径**：必须在**项目根目录**下执行（如 `python run_realtime.py`），以便 `utils`、`yolo`、`nets` 等可被正确导入。

---

## 4. 目录与文件说明

| 文件 | 说明 |
|------|------|
| `README.md` | 本说明文档。 |
| `__init__.py` | 包入口，导出配置、事件、类型等。 |
| `config.py` | 配置契约：`SourceConfig`、`StreamConfig`、`TrackerConfig`、`SystemConfig`、`build_stream_config_map`。 |
| `events.py` | 帧事件 `FrameEvent(stream_id, frame, ts)`、`EndOfStreamEvent(stream_id)`、`EventTag`。 |
| `types.py` | `DetectionResult`（boxes_xyxy, class_id, score）、`Track`（id, bbox_xyxy, score, class_id, class_name）。 |
| `queue_utils.py` | 队列写入：`put_event_nonblocking`，支持 `drop_old` 策略，不丢弃 EOS。 |
| `capture_rtsp.py` | RTSP 采集循环：重连、单调时间戳、向队列推送帧。 |
| `capture_file.py` | 本地视频采集循环：EOF 时推送 `EndOfStreamEvent`，可选 `target_fps`。 |
| `batch_aggregator.py` | `collect_batch(infer_queue, max_items, max_wait_ms)`：收集帧事件与 EOS。 |
| `yolo_batch.py` | `yolo_infer_batch(yolo, frames)`：调用现有 YOLO，返回 `list[DetectionResult]`。 |
| `bytetrack_wrapper.py` | `BYTETracker(cfg, class_names)`、`update(high, low, ts)` → `list[Track]`，类信息来自匹配检测。 |
| `roi_geometry.py` | `point_in_polygon`、`bbox_anchor_point`（center/bottom_center）、`roi_membership`（多 ROI）。 |
| `state.py` | 每流状态结构 `make_stream_state()`、`keys_where_track_id`。 |
| `dwell.py` | `update_roi_dwell`：进入/几何离开 ROI 的计时与 `save_duration` 回调。 |
| `alarm.py` | `update_alarm`、`trigger_sound`（Windows winsound）、`write_alarm_log`。 |
| `gc_flush.py` | `garbage_collect`（丢失超时结算、状态清理）、`on_end_of_stream_flush`（EOF 时 flush ROI）。 |
| `pipeline.py` | `run_pipeline(...)`：启动采集线程、推理循环、按流 ByteTrack + 停留 + 报警 + GC。 |
| `validation.py` | 上线前验证清单常量 `VALIDATION_CHECKLIST`。 |

项目根目录：

| 文件 | 说明 |
|------|------|
| `run_realtime.py` | 示例入口：加载 YOLO、构造 sources/stream_cfg、调用 `run_pipeline`。 |

---

## 5. 配置详解

### 5.1 SourceConfig（输入源）

| 字段 | 类型 | 说明 |
|------|------|------|
| `uri` | str | 流唯一标识，同时作为 `stream_id`。RTSP 示例：`rtsp://user:pass@ip:554/stream1`；文件示例：`D:/videos/cam1.mp4` 或任意绝对/相对路径。 |
| `type` | str | `"rtsp"` 或 `"file"`，决定采集逻辑（重连 vs EOF）。 |
| `enabled` | bool | 为 False 时该源不参与采集与配置构建。 |

### 5.2 StreamConfig（每流业务配置）

由 `build_stream_config_map` 或手动构建，键为 `stream_id`（即 `source.uri`）。

| 字段 | 类型 | 说明 |
|------|------|------|
| `rois` | List[Polygon] | 该流的多边形 ROI 列表，每个 Polygon 为 `[(x,y), ...]`，像素坐标（与推理输入分辨率一致，见后文 ROI 小节）。 |
| `roi_point_mode` | str | 判定“是否在 ROI 内”的锚点：`"center"` 为框中心，`"bottom_center"` 为框底边中点（推荐监控场景）。 |
| `enable_dwell` | bool | 是否统计该流 ROI 停留。 |
| `dwell_classes` | Set[str] | 仅对这些类别统计停留（如 `{"person", "backpack"}`），类别名与 `yolo.class_names` 一致。 |
| `enable_alarm` | bool | 是否启用该流报警。 |
| `alarm_classes` | Set[str] | 触发报警的类别（如 `{"cat", "dog"}`）。 |

### 5.3 TrackerConfig（ByteTrack）

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `track_high_th` | float | 0.5 | 高置信度检测阈值，优先参与匹配。 |
| `track_low_th` | float | 0.1 | 低置信度检测阈值，用于补救匹配。 |
| `match_thresh` | float | 0.8 | IoU 匹配阈值，高于此值才认为 track 与 det 匹配。 |
| `min_box_area` | float | 10.0 | 预留，当前实现未使用。 |

### 5.4 SystemConfig（全局系统）

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `max_batch` | int | 8 | 单次推理最大帧数。 |
| `max_wait_ms` | float | 40.0 | 等待首帧的最长时间（毫秒）。 |
| `lost_timeout_sec` | float | 2.0 | 超过此时间未匹配到的 track 视为“丢失”，若仍在 ROI 内则按丢失离开结算。 |
| `max_state_sec` | float | 300.0 | `last_seen` / `alarm_last_ts` 超过此时间则清理，防止内存增长。 |
| `alarm_cooldown_sec` | float | 30.0 | 同一 track_id 两次报警的最小间隔。 |
| `drop_policy` | str | `"drop_old"` | 队列满时：丢弃最旧帧以放入新帧；EOS 永不丢弃。 |
| `infer_queue_maxsize` | int | 64 | 推断队列最大长度。 |
| `rtsp_reconnect_delay_sec` | float | 2.0 | RTSP 断流后重连前等待时间。 |
| `rtsp_max_reconnect_attempts` | int | 10 | RTSP 最大重连次数，超过后该源线程退出。 |

---

## 6. 输入源与事件

### 6.1 输入源类型

- **RTSP**：`type="rtsp"`，`uri` 为完整 RTSP URL（可含用户名密码）。使用 OpenCV `CAP_FFMPEG` 解码，缓冲设为 1 以降低延迟。断帧时自动重连，直到达到最大重连次数。
- **本地文件**：`type="file"`，`uri` 为本地视频文件路径。读到 EOF 后推送 `EndOfStreamEvent(stream_id)` 并结束该线程；其它流不受影响。

### 6.2 事件类型

- **FrameEvent**：`stream_id`（= source.uri）、`frame`（OpenCV BGR  numpy 数组 H×W×3）、`ts`（`time.monotonic()` 秒）。
- **EndOfStreamEvent**：`stream_id`，仅文件源在 EOF 时产生；用于触发该流的 ROI 结算与 `log_stream_end`。

### 6.3 检测与跟踪类型

- **DetectionResult**（YOLO 输出）：`boxes_xyxy` (N×4)、`class_id` (N, int)、`score` (N, float)，坐标为**原图像素**。
- **Track**（ByteTrack 输出）：`id`、`bbox_xyxy` (4,)、`score`、`class_id`、`class_name`；类别来自匹配到的检测或沿用上一帧。

---

## 7. 各模块说明

### 7.1 config.py

- 定义所有配置数据类及 `build_stream_config_map`。
- `Polygon` 为 `List[Tuple[float, float]]`，顶点顺序任意，闭合即可。

### 7.2 events.py / types.py

- 事件与数据类型，见第 6 节。无业务逻辑。

### 7.3 queue_utils.py

- `put_event_nonblocking(queue, event, drop_policy)`：队列满时，若为帧事件且 `drop_policy=="drop_old"` 则丢弃队首一帧再放入；EOS 不丢弃，必要时挤掉一帧以放入 EOS。

### 7.4 capture_rtsp.py

- `capture_loop(source, infer_queue, sys_cfg, stop_event=None, on_error=None)`：循环 `cap.read()`；失败则释放并重连，间隔 `rtsp_reconnect_delay_sec`，最多 `rtsp_max_reconnect_attempts` 次；成功则 `time.monotonic()` 作为 `ts` 推送 `FrameEvent`。

### 7.5 capture_file.py

- `capture_loop(source, infer_queue, drop_policy, stop_event=None, target_fps=None)`：循环读帧直至 EOF；EOF 时推送 `EndOfStreamEvent(source.uri)` 并退出。`target_fps` 非空时在每帧后 sleep 以近似回放帧率。

### 7.6 batch_aggregator.py

- `collect_batch(infer_queue, max_items, max_wait_ms)`：阻塞等待首条（最多 `max_wait_ms`），然后非阻塞取到最多 `max_items` 条；返回 `(frame_events, eos_events)`，EOS 单独列出以便先处理。

### 7.7 yolo_batch.py

- `yolo_infer_batch(yolo, frames, conf_thres=None, nms_thres=None)`：对每帧 BGR 做 resize（与现有 YOLO letterbox 一致）、归一化、堆叠后调用 `yolo.net`，再 decode + NMS；返回 `list[DetectionResult]`，框坐标为原图像素。需在**项目根目录**运行以正确导入 `utils.utils`。

### 7.8 bytetrack_wrapper.py

- `BYTETracker(cfg, class_names)`：每流一个实例，内部维护 `_tracks` 与 `_lost`。
- `update(high, low, ts)`：`high`/`low` 为 `[(bbox_xyxy, score, class_id), ...]`；先高置信度与现有 track IoU 匹配，再低置信度与剩余 track 匹配，未匹配的高置信度生成新 track；返回本帧活跃的 `list[Track]`，class/score 来自匹配到的 det 或沿用上一帧。

### 7.9 roi_geometry.py

- `point_in_polygon(point_xy, polygon)`：射线法判断点是否在多边形内。
- `bbox_anchor_point(bbox_xyxy, mode)`：返回框中心或底边中点。
- `roi_membership(bbox_xyxy, rois, mode)`：返回该锚点落在哪些 ROI 的索引列表（可多选）。

### 7.10 state.py

- `make_stream_state()`：返回 `{"roi_start": {}, "last_seen": {}, "alarm_last_ts": {}}`。
- `keys_where_track_id(roi_start, track_id)`：在 `roi_start` 中找出所有第二元为 `track_id` 的 key。

### 7.11 dwell.py

- `update_roi_dwell(stream_id, ts, tracks, stream_cfg, state, save_duration)`：仅处理 `dwell_classes` 内类别；更新 `last_seen`；对每个 track 计算 `roi_membership`；若进入新 ROI 则写入 `roi_start[(roi_id, track_id)]=ts`；若曾在该 ROI 而当前不在则结算时长并调用 `save_duration(stream_id, roi_id, track_id, cls, enter_ts, ts, duration)` 并删除对应 key。丢失离开由 GC 处理。

### 7.12 alarm.py

- `update_alarm(stream_id, ts, tracks, stream_cfg, state, alarm_cooldown_sec)`：对 `alarm_classes` 内类别，若该 track 距上次报警已超过冷却时间，则调用 `trigger_sound(stream_id, class_name)` 与 `write_alarm_log(...)` 并更新 `alarm_last_ts[track_id]=ts`。
- `trigger_sound`：Windows 下使用 `winsound.Beep(1000, 300)`，其它平台可扩展。
- `write_alarm_log`：使用 `logging.warning` 输出一条带 stream_id、ts、track_id、类别、分数、框的日志。

### 7.13 gc_flush.py

- `garbage_collect(stream_id, ts, state, sys_cfg, save_duration)`：对 `roi_start` 中若某 track 的 `last_seen` 超过 `lost_timeout_sec`，则按丢失离开结算并删除 key；对 `last_seen` / `alarm_last_ts` 中超过 `max_state_sec` 的项删除。
- `on_end_of_stream_flush(stream_id, ts, state, save_duration, log_stream_end)`：对该流所有 `roi_start` 按 `last_seen`（无则用 ts）结算，清空 `roi_start`，再调用 `log_stream_end(stream_id)`。

### 7.14 pipeline.py

- `run_pipeline(sources, stream_cfg, sys_cfg, tracker_cfg, yolo, class_names, save_duration, log_stream_end=None, stop_event=None)`：
  - 为每个 enabled 源起一采集线程（RTSP 或 File）。
  - 主循环：`collect_batch` → 处理 EOS（flush + log）→ `yolo_infer_batch` → 对每帧按 stream_id 执行 ByteTrack → dwell → alarm → GC。
  - 所有状态与 tracker 均按 `stream_id` 隔离。

### 7.15 validation.py

- `VALIDATION_CHECKLIST`：五条验证项（跨流 ID 隔离、遮挡 ID 稳定、ROI 正确性、报警去重/冷却、文件 EOF 仅影响该流），用于上线前自检。

---

## 8. 运行方式

### 8.1 使用示例入口（项目根目录）

```bash
cd /path/to/yolov8-pytorch-master
python run_realtime.py
```

在 `run_realtime.py` 中修改：

- `sources`：RTSP 与文件可混合，例如：
  ```python
  sources = [
      SourceConfig(uri="rtsp://admin:pass@192.168.1.81:554/stream1", type="rtsp", enabled=True),
      SourceConfig(uri="D:/videos/test.mp4", type="file", enabled=True),
  ]
  ```
- `roi_config`：每个 `stream_id` 对应一列多边形，例如：
  ```python
  roi_config = {
      "rtsp://...": [[(100,100), (500,100), (500,400), (100,400)]],
      "D:/videos/test.mp4": [],
  }
  ```
- `dwell_classes` / `alarm_classes`：与 `yolo.class_names` 一致（如 COCO 80 类 + fire + smoke 则用对应名称）。

### 8.2 自定义入口

可自行编写脚本：加载 YOLO、构造 `sources` / `stream_cfg` / `sys_cfg` / `tracker_cfg`、实现 `save_duration` 与可选 `log_stream_end`，然后调用 `realtime.pipeline.run_pipeline(...)`。需保证在**项目根目录**下执行或已正确设置 `PYTHONPATH`。

### 8.3 停止

当前示例使用 `stop_event=threading.Event()` 但未在脚本中设置；若要优雅退出，可在主线程中 `stop_event.set()`，采集线程与主循环会随 daemon 线程和循环条件退出。如需更强可控性，可增加信号处理或其它停止逻辑。

---

## 9. ROI 与坐标

### 9.1 多边形格式

- 每个 ROI 为顶点列表 `[(x0,y0), (x1,y1), ...]`，单位与**送入 YOLO 前的原图**一致（即 OpenCV 读出的帧尺寸）。
- 顶点顺序任意，闭合即可；射线法会自动处理内外判定。
- 同一流可有多个 ROI；一个目标可同时落在多个 ROI 内，每个 ROI 独立计时。

### 9.2 锚点模式

- `bottom_center`：框底边中点 `((x1+x2)/2, y2)`，适合监控中“脚在区域内即算进入”。
- `center`：框中心，适合小目标或不在意脚部的情况。

### 9.3 坐标一致性

- 检测框来自 `yolo_infer_batch`，已通过 `yolo_correct_boxes` 映射回原图坐标，故 ROI 顶点应使用**与原图相同分辨率**下的像素坐标（即 `frame.shape[:2]` 对应的宽高）。

---

## 10. 验证清单

上线前建议逐项确认（见 `realtime/validation.py`）：

1. **跨流 ID 隔离**：`track_id` 仅在同一 `stream_id` 内有效；所有日志与回调均带 `stream_id`。
2. **遮挡稳定性**：调节 `TRACK_HIGH_TH`、`TRACK_LOW_TH`、`LOST_TIMEOUT_SEC`，在遮挡片段上验证 ID 是否保持稳定。
3. **ROI 正确性**：用录像回放抽查进入/离开事件，确认 `roi_point_mode` 与业务一致。
4. **报警不刷屏**：同一 track 不每帧报警；冷却生效；不同 track（如多只动物）可分别报警。
5. **文件 EOF**：文件流结束时仅该流做 flush 并 log，其它 RTSP/文件流继续运行。

---

## 11. 回调与扩展

### 11.1 save_duration

签名：`save_duration(stream_id, roi_id, track_id, cls, enter_ts, end_ts, duration)`。

- 调用时机：目标离开某 ROI（几何离开或丢失超时）或文件 EOF 时对该流未结算 ROI 做一次性结算。
- 可在此将停留记录写入数据库、文件或仅打印。

### 11.2 log_stream_end

签名：`log_stream_end(stream_id)`。

- 调用时机：收到该流的 `EndOfStreamEvent` 并完成 ROI flush 后。
- 可用于统计、告警或清理该流相关资源。

### 11.3 扩展报警

- 修改 `alarm.py` 中 `trigger_sound`（如播放自定义 WAV）或 `write_alarm_log`（如写数据库、推送到消息队列）。
- 报警逻辑仅在 `update_alarm` 中，按 `stream_cfg[stream_id].alarm_classes` 与冷却时间执行。

---

## 12. 性能与调参

- **max_batch**：在 GPU 显存允许下适当增大可提高吞吐；过大可能增加单批延迟。
- **max_wait_ms**：过小易导致 batch 常为 1，过大则延迟上升；建议 20–50 ms。
- **infer_queue_maxsize**：过小易丢帧（drop_old），过大会占用更多内存。
- **track_high_th / track_low_th**：高阈值过大会漏跟，低阈值过小会引入噪声；典型 0.5 / 0.1。
- **match_thresh**：ByteTrack IoU 匹配阈值，过小 ID 易切换，过大易丢跟。
- **lost_timeout_sec**：过小会过早结算停留，过大会延迟释放状态；监控场景常用 1–3 s。
- **max_state_sec**：长期运行必设，避免 `last_seen` / `alarm_last_ts` 无限增长。

---

## 13. 常见问题与排查

| 现象 | 可能原因 | 建议 |
|------|----------|------|
| 导入报错 `No module named 'utils'` | 未在项目根目录运行 | 在根目录执行 `python run_realtime.py` 或设置 `PYTHONPATH` 包含根目录。 |
| RTSP 连不上或频繁断 | 网络、鉴权、编码格式 | 检查 URL、账号密码、摄像头编码；可尝试 `CAP_FFMPEG` 与 `buffersize=1`。 |
| 队列满、大量丢帧 | 推理慢或源过多 | 增大 `infer_queue_maxsize`、适当增大 `max_batch` 或减少源数量；或降低分辨率/帧率。 |
| GPU OOM | batch 或分辨率过大 | 减小 `max_batch`、降低 YOLO 输入尺寸或减少并发流。 |
| 跟踪 ID 频繁切换 | 遮挡或阈值不当 | 调大 `match_thresh`、适当降低 `track_high_th` 或增加 `lost_timeout_sec`。 |
| 停留时间偏短/偏长 | ROI 锚点或超时不符业务 | 检查 `roi_point_mode` 与 `lost_timeout_sec`，用录像回放核对进入/离开时刻。 |
| 报警不响或每帧都响 | 类别名不一致或冷却未生效 | 确认 `alarm_classes` 与 `yolo.class_names` 完全一致（大小写、空格）；确认 `alarm_cooldown_sec` 已传入并大于 0。 |
| 文件播完其它流也停 | 逻辑错误 | 本实现中 EOS 仅影响对应 stream_id 的 flush，其它流应继续；若仍出现请检查是否误用同一 stop_event 或异常未捕获。 |

---

## 14. 安全与注意

- **RTSP 密码**：URL 中可能包含明文密码，注意日志与配置存储权限，避免泄露。
- **资源释放**：采集线程为 daemon，主进程退出时会被直接终止；若有写文件/数据库，尽量在 `log_stream_end` 或退出前做 flush。
- **线程安全**：每流状态与 tracker 仅被主线程访问，采集线程仅向队列投递事件，无需额外锁。若在回调中访问共享资源，需自行加锁或使用线程安全接口。
- **模型与类别**：YOLO 模型与 `class_names` 由项目根目录的 `yolo.py` 与 `Class/coco_classes.txt`（或你使用的类别文件）决定；本模块不修改模型，仅使用其推理结果与类别名。

---

## 版本与兼容

- 本模块随主工程一起使用，无独立版本号；若主工程升级 PyTorch / OpenCV，本模块通常无需改动。
- Python 3.6+ 可用；推荐 3.8+。Windows 下 `winsound.Beep` 可用；Linux/macOS 需自行在 `alarm.trigger_sound` 中扩展（如 `os.system('beep')` 或播放 WAV）。

---

## 参考

- 本仓库主 README：目标检测训练与预测流程。
- ByteTrack 论文与思路：高/低置信度两阶段关联，提升遮挡下 ID 稳定性。
- 计划文档：`unified-stream_bytetrack_roi_alarm_*.plan.md`（若存在）中的接口与流程说明。
