# Moniter — 多路视频智能监控系统

基于 **YOLOv8 + ByteTrack + VideoMAE-ViT** 的实时视频智能监控系统，支持多路 RTSP / 本地视频混合输入，集成目标检测、多目标跟踪、禁区/滞留/禁现物品/车辆超时等规则引擎，以及基于视频片段的动态事件分类（正常 / 盗窃 / 暴力冲突）。
此项目用了bullliiiing的yolov8-pytorch的全部代码，并且对数据处理部分进行了增删。其余代码均为本人撰写

## 功能概览

| 功能 | 说明 |
|------|------|
| 目标检测 | YOLOv8-L (COCO 80 类)，FP16 批量推理 |
| 多目标跟踪 | ByteTrack，人 / 车辆独立 ID 分配 |
| 禁区入侵 | 任意多边形 ROI，支持"框中心"/"框底中点"锚点模式 |
| 滞留检测 | 指定类别在 ROI 内停留超时告警 |
| 禁现物品 | 指定类别（如猫、狗）出现即刻告警 |
| 车辆超时 | 车辆在禁区内超过阈值（默认 10s）触发告警 |
| 动态事件 | VideoMAE + MIL Head 对视频片段分类：normal / steal / violent |
| 多路支持 | RTSP 与本地视频混合，每路独立配置规则与显示参数 |
| 实时显示 | OpenCV 窗口，带中文横幅（PIL 绘制）、禁区线、轨迹线、ID 标注 |

## 系统架构

```
采集线程 (RTSP/File)
    │  frame.copy()
    ▼
infer_queue (64)
    │
    ▼
主线程: collect_batch
    │
    ├─► YOLO 批量推理 (FP16, 真 batch forward)
    │       │
    │       ▼
    │   ByteTrack 跟踪
    │       │
    │       ▼
    │   规则引擎 (禁区/滞留/禁现/车辆)
    │       │
    │       ▼
    │   on_frame_after_yolo ──► ViT 异步线程 (缓冲 → 编码 → 分类)
    │       │
    │       ▼
    │   draw_frame (单次 PIL 转换)
    │       │
    │       ▼
    └── cv2.imshow + waitKey
```

## 目录结构

```
moniter/
├── predict.py                     # 主入口: YOLO + 规则 + ViT
├── README.md
│
├── yolo/
│   ├── yolo.py                    # YOLO 推理封装
│   ├── train.py                   # YOLO 训练脚本 (VOC 格式)
│   ├── run_realtime.py            # 仅 YOLO + 规则 (无 ViT) 入口
│   ├── nets/
│   │   └── yolo.py                # YOLOv8 模型定义 (n/s/m/l/x)
│   ├── utils/
│   │   ├── utils_bbox.py          # DecodeBox, NMS, 坐标变换
│   │   ├── utils_fit.py           # 训练 epoch 逻辑
│   │   └── callbacks.py           # 训练回调 (mAP, loss log)
│   ├── realtime/
│   │   ├── config.py              # SourceConfig, StreamConfig, SystemConfig, TrackerConfig
│   │   ├── pipeline.py            # 采集 → 推理 → 规则 → 显示 主循环
│   │   ├── yolo_batch.py          # YOLO 批量推理 (cv2 预处理, 真 batch)
│   │   ├── batch_aggregator.py    # 队列批量收集
│   │   ├── bytetrack_wrapper.py   # ByteTrack 封装
│   │   ├── capture_rtsp.py        # RTSP 采集线程 (重连机制)
│   │   ├── capture_file.py        # 本地视频采集线程
│   │   ├── dwell.py               # 滞留检测
│   │   ├── alarm.py               # 禁现物品 & 车辆超时告警
│   │   ├── roi_geometry.py        # 多边形 ROI & 点在多边形内判定
│   │   ├── display.py             # 绘制 (bbox/轨迹/横幅/禁区线)
│   │   └── ...
│   ├── Class/                     # 类别文件 (coco_classes.txt 等)
│   ├── model_data/                # 字体文件 (simhei.ttf) 等
│   └── logs/                      # 训练权重 (best_epoch_weights.pth)
│
└── Vit/
    └── lab_anomaly/
        ├── configs/               # YAML 配置示例
        ├── data/
        │   ├── rtsp_record.py     # RTSP 录制 → 训练数据
        │   ├── clip_dataset.py    # 视频片段数据集
        │   └── video_labels.py    # 标签管理
        ├── models/
        │   ├── vit_video_encoder.py  # VideoMAE 编码器 (单流/双流)
        │   └── mil_head.py           # MIL 注意力分类头
        ├── train/
        │   ├── train_known_classifier.py  # 已知事件分类器训练
        │   ├── extract_embeddings.py      # 视频 embedding 提取
        │   └── precompute_optical_flow.py # 光流预计算
        └── infer/
            ├── known_event_runtime.py     # 实时推理 (异步线程)
            └── scoring.py                 # 分类器加载
```

## 快速开始

### 环境要求

- Python 3.8+
- NVIDIA GPU (推荐 RTX 3060 及以上, 已在 RTX 4090 上验证)
- CUDA 11.7+

### 安装依赖

```bash
# YOLO 相关
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy pillow tqdm scipy tensorboard

# ViT 相关
pip install transformers scikit-learn joblib pyyaml
```

### 准备模型

1. **YOLO 权重**：将训练好的权重放到 `yolo/logs/best_epoch_weights.pth`
2. **ViT 权重**：将已知分类器放到 `Vit/lab_dataset/derived/known_classifier/checkpoint_best.pt`
3. **类别文件**：确认 `yolo/Class/coco_classes.txt` 包含所需类别

### 运行

```bash
# 完整管线（YOLO + 规则 + ViT 动态事件）
python predict.py

# 仅 YOLO + 规则（无 ViT）
cd yolo && python run_realtime.py
```

## 配置说明

所有参数集中在 `predict.py` 的配置区修改：

### 输入源

```python
def _sources():
    return [
        SourceConfig(uri="rtsp://admin:password@192.168.1.100:554/stream", type="rtsp", enabled=True),
        SourceConfig(uri=r"D:\videos\test.mp4", type="file", enabled=True),
    ]
```

### 每路规则（禁区 / 禁现 / 滞留 / 车辆）

```python
StreamConfig(
    rois=[[(100, 100), (100, 300), (400, 300), (400, 100)]],  # 禁区多边形
    roi_point_mode="bottom_center",   # 锚点模式
    enable_dwell=True,                # 启用滞留检测
    dwell_classes={"person"},         # 滞留检测类别
    enable_alarm=True,                # 启用禁现物品
    alarm_classes={"cat", "dog"},     # 禁现类别
    vehicle_alarm_classes={"car", "truck", "bus"},
    vehicle_alarm_sec=10.0,           # 车辆超时秒数
)
```

### YOLO 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `YOLO_CONFIDENCE` | 0.3 | 检测置信度阈值 |
| `phi` | `'l'` | 模型规模 (n/s/m/l/x) |
| `input_shape` | [640, 640] | 输入分辨率 |

### ViT 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `VIT_CLIP_LEN` | 16 | 每个片段帧数 |
| `VIT_FRAME_STRIDE` | 2 | 帧采样步长 |
| `VIT_WINDOW_STRIDE` | 4 | 推理滑窗步长 |
| `VIT_USE_HALF` | True | FP16 推理 |

### 显示参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `display_width` | 1280 | 窗口宽度 |
| `display_height` | 720 | 窗口高度 |
| `vit_anomaly_threshold` | 0.55 | ViT 异常报警阈值 |
| `dwell_warning_sec` | 5.0 | 滞留告警秒数 |

## 训练

### YOLO 训练

1. 准备 VOC 格式数据集（图片 + XML 标注）
2. 修改 `yolo/train.py` 中的 `classes_path`、`model_path`（预训练权重）
3. 运行：

```bash
cd yolo && python train.py
```

训练输出保存在 `yolo/logs/`。

### ViT 动态事件训练

1. **录制视频**：用 `Vit/lab_anomaly/data/rtsp_record.py` 从 RTSP 录制视频片段
2. **标注**：按类别整理视频到 `raw_videos/` 子目录，编辑 `video_labels.csv`
3. **提取 embedding**：

```bash
cd Vit && python -m lab_anomaly.train.extract_embeddings --config lab_anomaly/configs/embedding_example.yaml
```

4. **训练分类器**：

```bash
python -m lab_anomaly.train.train_known_classifier \
    --embeddings_dir lab_dataset/derived/embeddings \
    --out_dir lab_dataset/derived/known_classifier
```

输出 `checkpoint_best.pt` 即可用于实时推理。

## 性能优化

本项目已针对实时推理进行系统级优化：

| 优化项 | 说明 |
|--------|------|
| 真批推理 | 多帧堆叠为 batch 一次 forward，而非逐帧循环 |
| FP16 | YOLO 模型 `.half()` + 输入 tensor `.half()` |
| cv2 预处理 | letterbox 使用 `cv2.resize` 替代 PIL |
| ViT 异步 | 推理在独立线程中执行，不阻塞主线程 |
| 单次 PIL | 所有中文横幅合并为一次 BGR→PIL→BGR 转换 |
| 字体缓存 | `ImageFont.truetype` 按 size 缓存 |
| GPU NMS | `torch.unique` 在 GPU 执行，消除 CPU-GPU 往返 |

在 RTX 4090 上，2 路 1080p 视频可达 40-80+ FPS。

## 操作快捷键

| 按键 | 功能 |
|------|------|
| ESC | 退出程序 |

## 许可证

本项目仅供学习和研究使用。YOLO 部分基于 [YOLOv8](https://github.com/ultralytics/ultralytics) 架构，ViT 部分使用 [VideoMAE](https://huggingface.co/MCG-NJU/videomae-base) 预训练模型。
