# Vit — 视频异常检测模块

> 负责"看一小段视频里发生了什么"。基于 VideoMAE v2 + MIL（Multiple Instance Learning）实现视频级异常行为识别。

---

## 这个模块能做什么

| 能力 | 说明 |
|------|------|
| **视频异常检测** | 对一段连续视频（如 16 帧）判断是"正常"还是"异常" |
| **端到端训练** | 从原始视频 → 预切 clip → 训练 VideoMAE + MIL → 输出 checkpoint |
| **异步实时推理** | 缓存最近帧组成滑动窗口，异步线程推理，不阻塞主循环 |
| **独立 RTSP 服务** | 可脱离主项目，单独作为视频流异常检测服务运行 |

---

## 在整个项目中的位置

```
yolo/（看见目标）
    ↓ 提供视频帧
Vit/（判断异常）
    ↓ 触发可疑时
vlm/（解释异常）
    ↓ 输出结论
web/（展示结果）
```

Vit 是**事件识别层**：YOLO 告诉系统"画面里有什么"，ViT 告诉系统"这段视频发生了什么"。

---

## 技术架构

### 模型组成

```
视频片段 (B, C, T, H, W)
    ↓
VideoMAE v2 Encoder
    - Patch Embedding
    - 12 层 Transformer
    - CLS Token / Mean Pooling
    ↓
特征向量 (B, D)  D=768
    ↓
MIL Head
    - Attention Pooling 或 Top-K 聚合
    - 异常分数分支（可选）
    ↓
分类结果 (normal / anomaly) + 异常概率 + ranking 分数
```

### 核心组件

| 组件 | 文件 | 作用 |
|------|------|------|
| **VideoMAE v2 编码器** | `lab_anomaly/models/vit_video_encoder.py` | 把视频 clip 编码成 768 维特征向量 |
| **MIL 分类头** | `lab_anomaly/models/mil_head.py` | 聚合多 clip 特征，输出视频级分类 |
| **排序损失** | `lab_anomaly/models/ranking_loss.py` | 训练时帮助区分正常/异常片段 |

---

## 目录结构

```
Vit/
├── README.md                  # 本文件（模块总览）
├── docs/                      # 补充学习笔记和历史设计思路
│
├── lab_anomaly/               # 核心代码（训练 + 推理）
│   ├── data/                  # 数据读取与预处理
│   ├── models/                # VideoMAE v2 + MIL 模型定义
│   ├── tool/                  # 训练前准备工具
│   ├── train/                 # 训练入口
│   ├── infer/                 # 推理入口
│   └── configs/               # YAML 配置文件
│
└── lab_dataset/               # 数据目录约定
    ├── raw_videos/            # 原始视频
    ├── labels/                # video_labels.csv 标签文件
    └── derived/               # 中间产物和训练输出
        ├── preclips/          # 离线预切 clip
        └── end2end_classifier/# 训练产物（checkpoint）
```

---

## 主流程（当前推荐）

### 第一步：准备视频和标签

```
lab_dataset/raw_videos/        ← 放入原始视频
lab_dataset/labels/video_labels.csv  ← 编辑标签
```

CSV 格式：
```csv
video_id,video_path,label,camera_id,start_time,end_time,note
```
- `label`：`normal` 或异常类名（如 `fall`、`violent`、`fire_smoke`）
- `video_path`：相对于 `lab_dataset/` 的路径

### 第二步：离线预切 clip

```bash
python Vit/lab_anomaly/tool/precompute_clips.py
```

- 读取 `video_labels.csv` 和原始视频
- 按配置参数切成固定长度的 clip（`.npz` 格式）
- 输出到 `lab_dataset/derived/preclips/`

### 第三步：端到端训练

```bash
python Vit/lab_anomaly/train/train_end2end.py
```

- 读取预切 clip
- 三阶段渐进解冻训练：
  1. **head_only**：30 epochs，lr=1e-3，冻结全部 backbone
  2. **unfreeze_2**：20 epochs，lr=5e-5，解冻最后 2 层
  3. **unfreeze_4**：15 epochs，lr=2e-5，解冻最后 4 层
- 损失：CE Loss + λ * MIL Ranking Loss
- 输出到 `lab_dataset/derived/end2end_classifier/`

### 第四步：实时推理

**接入主项目**：
- 被 `web/services/runtime_manager.py` 调用
- 使用 `lab_anomaly/infer/known_event_runtime.py`

**独立运行**：
```bash
python Vit/lab_anomaly/infer/rtsp_service.py
```

---

## 关键参数对齐表

| 参数 | 预切阶段 | 训练阶段 | 推理阶段 | 说明 |
|------|----------|----------|----------|------|
| `frames_per_clip` / `clip_len` | ✅ | ✅ | ✅ | 每片段多少帧，必须一致 |
| `encoder_model_name` | — | ✅ | ✅ | 编码器名称，必须一致 |
| `window_stride` | — | — | ✅ | 滑窗步长，控制推理频率 |

> ⚠️ 这些参数前后不一致，轻则效果变差，重则直接报错。

---

## 注意事项

1. **编码器模型名**：当前主线使用 `OpenGVLab/VideoMAEv2-Base`，首次运行会自动下载预训练权重
2. **Meta Tensor 修复**：某些 transformers 版本下 `pos_embed` 可能残留 meta tensor，代码中已包含自动修复（重建正弦位置编码）
3. **训练/推理帧数对齐**：`clip_len`、`frame_stride`、`window_stride` 在训练脚本和推理代码中必须一致
4. **数据路径**：`video_labels.csv` 中的路径建议使用相对于 `lab_dataset/` 的路径，便于迁移
