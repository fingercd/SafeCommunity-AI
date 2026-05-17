# Moniter —— AI 编码助手项目指南

> 本文档供 AI 编码助手阅读。如果你第一次接触本项目，请先阅读根目录 `README.md` 获得整体概念，再阅读本文件获得开发、修改、运行所需的精确信息。

---

## 项目概览

`moniter` 是一个多路视频监控系统，把**目标检测**、**目标跟踪**、**区域规则告警**、**视频异常事件识别**、**大模型复核**、**网页展示**串成一条完整链路。

项目不是单一模型工程，而是 4 条互相配合的主线：

| 模块 | 职责 | 关键技术 |
|------|------|----------|
| `yolo/` | 目标检测 + 多目标跟踪 + 区域规则 | YOLOv8（PyTorch 自研实现）、ByteTrack |
| `Vit/` | 视频片段级异常事件识别 | VideoMAE v2 + MIL（Multiple Instance Learning） |
| `vlm/` | 大视觉语言模型复核，给异常片段做文字解释 | Qwen-VL + QLoRA 微调 |
| `web/` | 网页监控层，把上面三层串起来展示 | Flask + MJPEG 流 |

### 典型数据流

```text
视频源（RTSP / 本地文件）
    ↓
采集线程（每路独立）
    ↓
推理队列 → 批量聚合器
    ↓
YOLO 批量检测
    ↓
ByteTrack 跟踪
    ↓
规则判断（ROI / 滞留 / 禁现物品 / 车辆超时）
    ↓
ViT 异步事件识别（可选）
    ↓
VLM 二次复核（Web 端可选）
    ↓
绘制画面与状态
    ↓
OpenCV 本地窗口 或 Web MJPEG 页面
```

---

## 技术栈与依赖

- **语言**：Python >= 3.8（实际开发环境常用 Python 3.13 / 3.14）
- **深度学习框架**：PyTorch、torchvision、transformers（HuggingFace）
- **CV/视频**：OpenCV、Pillow、decord、numpy
- **Web**：Flask
- **跟踪**：ByteTrack（内嵌在 `yolo/realtime/`）
- **大模型微调**：PEFT（LoRA/QLoRA）、bitsandbytes、accelerate
- **训练可视化**：tensorboard、matplotlib、swanlab（可选）

### 各模块独立 requirements

| 模块 | 文件 |
|------|------|
| `yolo/` | `yolo/requirements.txt` |
| `Vit/` | `Vit/lab_anomaly/requirements.txt` |
| `vlm/` | `vlm/requirements.txt` |

> 注意：本项目没有单一的根级 `requirements.txt`，运行某一模块前请安装对应依赖。`web/` 的依赖是 `yolo/` + `Vit/` + `vlm/` 的并集 + `flask`。

---

## 项目结构

```text
moniter/
├── predict.py                 # 命令行版总入口（YOLO + ViT，本地窗口）
├── config.yaml                # 统一实时监控配置（供 predict.py 读取）
├── pyproject.toml             # 仅配置 black / ruff 代码格式化，无构建逻辑
├── yolo/
│   ├── yolo.py                # YOLO 推理核心类
│   ├── train.py               # YOLO 训练入口
│   ├── predict.py             # YOLO 离线推理入口（单图/视频/文件夹/FPS测试/ONNX导出）
│   ├── run_realtime.py        # 仅跑 YOLO + 规则的实时入口
│   ├── nets/                  # 模型结构（backbone、检测头、损失）
│   ├── utils/                 # 训练/推理工具（dataloader、NMS、mAP、fit 等）
│   ├── realtime/              # ⭐ 实时管线核心（采集、队列、批量推理、跟踪、规则、绘制）
│   ├── Class/                 # 类别文件（如 coco_classes.txt）
│   ├── model_data/            # 中文字体等显示资源
│   ├── logs/                  # 训练产物（best_epoch_weights.pth 等）
│   └── Datasets/              # 数据集与训练索引
├── Vit/
│   ├── lab_anomaly/           # ⭐ ViT 核心代码
│   │   ├── data/              # 数据读取、clip 数据集、视频标签
│   │   ├── models/            # VideoMAE v2 编码器、MIL 头、排序损失
│   │   ├── tool/              # 预切 clip 工具
│   │   ├── train/             # 端到端训练入口（train_end2end.py）
│   │   ├── infer/             # 实时推理运行时（known_event_runtime.py、rtsp_service.py）
│   │   └── configs/           # 训练/推理 YAML 配置
│   └── lab_dataset/           # 数据约定目录（raw_videos、labels、derived）
├── vlm/
│   ├── configs/default.yaml   # VLM 主配置（模型路径、LoRA 参数、训练超参）
│   ├── data/                  # 数据准备、切片、生成训练集
│   ├── train/                 # QLoRA 训练、LoRA 合并、评估
│   ├── infer/                 # 复核推理引擎（vlm_engine.py）
│   ├── pycharm/               # 分步脚本（元数据准备 → clip → 数据集 → 训练 → 评估）
│   ├── outputs/               # 训练产物（qlora/、merged/）
│   └── Qwen/、Qwen 3.5 2b/    # 本地基础模型目录（本机绝对路径）
└── web/
    ├── app.py                 # Flask 入口（路由、MJPEG、流 CRUD）
    ├── config/streams.json    # 流列表持久化
    ├── services/
    │   ├── runtime_manager.py # 后台总调度（启动 pipeline、加载 ViT/VLM）
    │   ├── frame_state.py     # 最新画面与状态缓存
    │   ├── stream_store.py    # 读写 streams.json
    │   └── vlm_review_runtime.py # VLM 复核线程
    ├── templates/index.html   # 主页面
    └── static/app.js          # 前端脚本
```

---

## 运行方式

本项目**没有正式的构建步骤**，直接以 Python 脚本方式运行。推荐在 PyCharm 中打开对应入口文件后点击运行，而不是强依赖命令行传参。

### 主要入口

| 场景 | 入口文件 | 说明 |
|------|----------|------|
| 本地窗口版整套监控 | `predict.py` | 读取 `config.yaml`，启动 YOLO + ViT + 规则 + OpenCV 显示 |
| 仅 YOLO + 规则（不接 ViT） | `yolo/run_realtime.py` | 调试检测、跟踪、规则链路 |
| 网页版监控 | `web/app.py` | Flask 服务，管理多路流，展示 ViT/VLM 结果 |
| YOLO 训练 | `yolo/train.py` | 修改文件顶部参数后直接运行 |
| YOLO 离线推理 | `yolo/predict.py` | 单图/视频/文件夹/FPS/热力图/ONNX |
| ViT 预切 clip | `Vit/lab_anomaly/tool/precompute_clips.py` | 离线切分训练 clip |
| ViT 端到端训练 | `Vit/lab_anomaly/train/train_end2end.py` | 当前 ViT 主训练入口 |
| ViT 实时推理 | `Vit/lab_anomaly/infer/known_event_runtime.py` | 供 `predict.py` 或 Web 调用 |
| VLM QLoRA 训练 | `vlm/train/train_qlora.py` | 读取 `configs/default.yaml` |
| VLM LoRA 合并 | `vlm/train/merge_lora.py` | 合并为 Web 可用的完整模型 |
| VLM 推理引擎 | `vlm/infer/vlm_engine.py` | 统一复核接口 |

### 常用命令示例

```bash
# 本地窗口版监控（优先读取 config.yaml）
python predict.py --config config.yaml

# 网页版监控
python -m web.app
# 或
python web/app.py

# YOLO 训练（修改 yolo/train.py 顶部参数后直接运行）
python yolo/train.py

# ViT 端到端训练
python Vit/lab_anomaly/train/train_end2end.py

# VLM QLoRA 训练
python vlm/train/train_qlora.py --config vlm/configs/default.yaml
```

---

## 配置系统

### 根级统一配置：`config.yaml`

`predict.py` 优先读取 `config.yaml`（通过 `--config` 指定），没有配置文件则回退到 `predict.py` 内建的 `_default_*()` 函数。

`config.yaml` 包含 6 大段：
1. `sources` —— 视频源列表（`id`、`uri`、`type`、`enabled`）
2. `streams` —— 每路规则配置（键 = `source.id`）
3. `yolo` —— 模型路径、类别文件、置信度
4. `vit` —— checkpoint 路径、clip 长度、帧步长、编码器模型名
5. `system` —— 批量大小、显示参数、告警冷却、跟踪阈值等
6. `tracker` —— ByteTrack 参数

### 各模块独立配置

| 文件 | 用途 |
|------|------|
| `Vit/lab_anomaly/configs/train_end2end.yaml` | ViT 训练参数覆盖（与 `train_end2end.py` 内默认值合并） |
| `Vit/lab_anomaly/configs/rtsp_service_example.yaml` | ViT 推理服务示例配置 |
| `vlm/configs/default.yaml` | VLM 全流程主配置（模型、数据、LoRA、训练、推理） |
| `web/config/streams.json` | Web 端流列表持久化（增删改查直接影响此文件） |

### 环境变量覆盖

`web/services/runtime_manager.py` 支持以下环境变量覆盖默认路径：

| 变量 | 说明 |
|------|------|
| `YOLO_WEIGHTS` | YOLO 权重路径 |
| `YOLO_CLASSES` | YOLO 类别文件路径 |
| `VIT_CHECKPOINT` | ViT checkpoint 路径 |
| `VLM_MERGED` | VLM 合并后模型目录 |
| `VLM_BASE` | VLM 基座模型目录 |

---

## 代码风格

- **格式化工具**：`black` + `ruff`
- **配置位置**：`pyproject.toml`
- **行宽**：88
- **目标 Python 版本**：3.8+
- **引号风格**：双引号（`ruff.format.quote-style = "double"`）
- **ruff 规则**：`E`、`F`、`I`、`W`，忽略 `E501`（行宽由 black 控制）

```toml
[tool.black]
line-length = 88
target-version = ["py38"]

[tool.ruff]
line-length = 88
target-version = "py38"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
ignore = ["E501"]
```

---

## 测试策略

> ⚠️ 本项目**没有自动化单元测试套件**。测试以“脚本验证 + 人工观察”为主。

现有测试/验证相关文件：

| 文件 | 作用 |
|------|------|
| `yolo/utils/test.py` | 训练标注格式检查器（检查图片是否存在、框格式、坐标越界、类别越界） |
| `vlm/train/stress_test_eval.py` | VLM 压力/批量评估脚本 |
| `yolo/check.py` | 模型/权重检查脚本 |
| `yolo/get_map.py` | mAP 计算验证 |

### 推荐的验证方式

1. **YOLO 检测效果验证**：先用 `yolo/predict.py` 跑单图/视频模式，确认画框和类别正确。
2. **实时链路验证**：用 `yolo/run_realtime.py` 跑一路视频，确认采集→检测→跟踪→规则→显示无报错。
3. **ViT 训练验证**：训练后查看 `lab_dataset/derived/end2end_classifier/eval_report/` 下的混淆矩阵、ROC 曲线、AUC。
4. **Web 端到端验证**：启动 `web/app.py` 后，在浏览器访问 `http://localhost:5000`，确认 MJPEG 画面、流增删改查、ViT/VLM 状态均正常。

---

## 部署与运行环境

### 开发环境特征

- **操作系统**：Windows（大量绝对路径写死为 `C:\Users\Administrator\Desktop\...`）
- **IDE**：PyCharm（项目中有大量 `.idea/` 目录，部分脚本顶部直接写死 PyCharm 工作目录说明）
- **虚拟环境**：`yolo/.venv/`（Python 3.14）、Conda env（如 `yolovv`）
- **显卡**：NVIDIA RTX 4090 24GB（VLM 训练说明中提及）

### 部署注意事项

1. **路径问题**：很多脚本内部写死了本机绝对路径，换机器后必须优先核对以下位置：
   - `predict.py` 的 `_default_sources()`
   - `config.yaml` 的 `sources.uri`
   - `vlm/configs/default.yaml` 的模型路径、数据路径
   - `web/services/runtime_manager.py` 的 `DEFAULT_*` 路径
   - 各 `requirements.txt` 中的版本号（部分较旧，如 `scipy==1.2.1`）

2. **模型权重位置约定**：

   | 类型 | 常见位置 |
   |------|----------|
   | YOLO 最佳权重 | `yolo/logs/best_epoch_weights.pth` |
   | ViT 训练产物 | `Vit/lab_dataset/derived/end2end_classifier/` |
   | Web 默认 ViT | `Vit/lab_dataset/derived/end2end_classifier/checkpoint_best.pt` |
   | VLM LoRA 输出 | `vlm/outputs/qlora/` |
   | VLM 合并后模型 | `vlm/outputs/merged/` |

3. **Web 不是独立系统**：`web/app.py` 依赖 `yolo/`、`Vit/`、`vlm/`（可选）。如果底层模型缺失，Web 页面会降级运行（如 YOLO 缺失则主功能不可用，ViT 缺失则无异常识别，VLM 缺失则无复核文字）。

4. **流变更会触发重启**：在 Web 端新增、删除、启停流时，`runtime_manager` 会 `stop()` 再 `start()` 整个 pipeline，不是无感热更新。

5. **ViT 训练与推理参数必须对齐**：
   - `frames_per_clip` / `clip_len`
   - `frame_stride`
   - `encoder_model_name`（当前主线为 `OpenGVLab/VideoMAEv2-Base`）
   这些值不一致轻则效果变差，重则直接报错。

---

## 安全与敏感信息

- `web/config/streams.json` 中可能包含**摄像头 RTSP 账号密码**（如 `rtsp://admin:密码@ip:554/...`）。
- 多个配置文件和脚本中写死了本地用户名和桌面路径，发布到公开仓库前需脱敏。
- `.gitignore` 当前仅排除了 `.gitnexus`，没有排除模型权重、数据集等超大文件，提交前请手动检查。

---

## 修改代码时的关键约定

1. **类别名一致性**：规则中使用的类别名（如 `person`、`car`、`fire`）必须与 `yolo/Class/coco_classes.txt` 中的行内容**完全一致**（包括大小写），否则会出现不统计、不报警、不显示轨迹的问题。

2. **ROI 坐标使用原图像素**：配置 `rois` 时按视频原始分辨率写坐标，不要按显示缩放后的尺寸写。

3. **配置双源**：很多脚本采用“Python 文件内默认值 + YAML 覆盖”模式。修改配置时要同时看 Python 文件顶部和对应 YAML，避免只改一边。

4. **线程安全**：实时管线大量使用 `threading.Queue`、`threading.Event`、`threading.Lock`。在 `realtime/`、`web/services/` 中修改共享状态时需注意线程安全。

5. **回调接口**：`run_pipeline()` 提供两个重要回调：
   - `on_frame_after_yolo(stream_id, frame_bgr, ts, state)` —— YOLO/规则之后、显示之前
   - `frame_sink(stream_id, drawn_frame, ts, state)` —— 绘制之后，用于 Web 输出
   Web 端正是靠这两个回调把 ViT/VLM 结果注入状态并缓存到 `frame_state`。

---

## 常见问题速查

| 现象 | 优先检查 |
|------|----------|
| YOLO 不画框 / 框类别不对 | `yolo/Class/coco_classes.txt` 是否对齐；`confidence` 是否过低 |
| 规则不触发 | 类别名大小写是否一致；ROI 坐标是否为原图像素 |
| ViT 加载失败 | checkpoint 路径是否存在；`clip_len` / `frame_stride` 是否与训练时一致 |
| VLM 不复核 | `vlm/outputs/merged/` 或 `vlm/Qwen/` 是否存在；transformers 版本是否匹配 |
| Web 页面无画面 | `runtime_manager` 是否已 `start()`；流 `enabled` 是否为 `true`；MJPEG 路由是否被防火墙拦截 |
| 训练报显存 OOM | 减小 `batch_size`、`max_seq_length`、`clip_len`；关闭 `tf32`；使用 `adamw_torch` 替代 `paged_adamw_8bit` |

---

## 阅读顺序建议

如果你是第一次修改本项目，建议按以下顺序理解代码：

1. 根目录 `README.md` —— 整体分层概念
2. `yolo/README.md` —— 基础检测与实时管线
3. `Vit/README.md` —— 当前 ViT 主线（VideoMAE v2 + MIL）
4. `web/README.md` —— 网页监控层
5. `vlm/README.md` —— 大模型复核层
6. 再深入到具体要改的脚本（如 `realtime/pipeline.py`、`web/services/runtime_manager.py` 等）
