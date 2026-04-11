# Moniter

一个把多路监控、目标检测、规则告警、视频事件识别、网页展示串起来的完整项目。

当前仓库不是单一模型工程，而是 4 条互相配合的主线：

- `yolo/`：目标检测、多目标跟踪、区域规则、实时显示。
- `Vit/`：视频片段级事件识别，当前真实主线是 `VideoMAE v2 + MIL`。
- `vlm/`：大视觉语言模型复核，用来给异常片段做二次说明。
- `web/`：网页监控层，把 YOLO、ViT、VLM 串起来做多路展示。

## 这套项目能做什么

| 能力 | 说明 |
|------|------|
| 多路视频接入 | 支持 RTSP 和本地视频混合输入。 |
| 目标检测 | 用 `YOLOv8` 检出人、车、物体等目标。 |
| 目标跟踪 | 用 `ByteTrack` 给同一路视频里的目标分配稳定 ID。 |
| 规则告警 | 支持禁区、滞留、禁现物品、车辆超时等规则。 |
| 视频事件识别 | 用 `VideoMAE v2 + MIL` 对一段视频片段判断 `normal / anomaly`。 |
| 网页监控 | 用 `Flask + MJPEG` 展示多路视频和状态。 |
| 大模型复核 | 当 ViT 觉得可疑时，再交给 `vlm/` 里的模型做文字解释。 |

## 先看懂整体关系

### 1. 根目录主入口

- `predict.py`
  - 这是命令行版总入口。
  - 它负责把 `yolo` 和 `Vit` 接起来。
  - 优先读取 `config.yaml`，如果你不传配置，就走文件里写好的默认参数。

- `config.yaml`
  - 这是整套实时监控的集中配置。
  - 包括视频源、每一路规则、YOLO 路径、ViT 路径、系统显示参数、跟踪参数。

### 2. 实时数据流

```text
视频源（RTSP / 文件）
    ↓
采集线程
    ↓
推理队列
    ↓
YOLO 批量检测
    ↓
ByteTrack 跟踪
    ↓
规则判断（ROI / 滞留 / 禁现 / 车辆）
    ↓
ViT 异步事件识别
    ↓
绘制画面与状态
    ↓
OpenCV 窗口 或 Web 页面
```

### 3. 网页版的额外流程

```text
YOLO 检测结果
    ↓
ViT 先判断当前片段是否可疑
    ↓
如果可疑并且超过阈值
    ↓
VLM 再做二次复核和文字解释
    ↓
Web 页面展示检测结果 + ViT 结果 + 大模型结论
```

## 顶层目录说明

```text
moniter/
├── predict.py
├── config.yaml
├── README.md
├── yolo/
├── Vit/
├── vlm/
└── web/
```

### `yolo/`

负责“看见画面里有什么”。

- 训练目标检测模型。
- 对单张图、视频、实时流做检测。
- 在 `realtime/` 里把检测、跟踪、规则、显示串成一条实时管线。

### `Vit/`

负责“看一小段视频里发生了什么”。

这一块最容易被旧文档误导，所以要特别说明：

- **当前仓库里真实存在、能对上的主线**：`VideoMAE v2 + MIL + 端到端训练`
- **旧文档里提到但当前仓库未完整保留的老流程**：光流、embedding、开放集边界等那一套

也就是说，现在你要把这个仓库介绍给别人时，应该把 `Vit` 的主线写成：

1. 先把视频离线切成 clip。
2. 再用 `train_end2end.py` 做端到端训练。
3. 训练好的 checkpoint 给 `known_event_runtime.py` 或 `rtsp_service.py` 使用。

### `vlm/`

负责“当系统怀疑异常时，再用大模型帮你解释一下到底像什么异常”。

这一块不是基础检测必须要有的模块，而是增强模块：

- 可以准备数据、做 QLoRA 微调。
- 可以把 LoRA 合并成完整模型。
- Web 监控会在 ViT 触发后调用它做复核。

### `web/`

负责“把整个系统变成网页监控工具”。

- 页面里可以添加、删除、启停视频流。
- 后台启动 `yolo.realtime.pipeline`。
- 读取 ViT 和 VLM 的判断结果并显示到页面上。

## 每一层建议怎么读

如果你是第一次接触这个项目，建议按这个顺序看：

1. 先看根目录 `README.md`，知道整个项目是怎么分层的。
2. 再看 `yolo/README.md`，先理解基础检测和实时管线。
3. 再看 `Vit/README.md`，确认当前最新 ViT 主线到底是什么。
4. 如果你要做网页监控，再看 `web/README.md`。
5. 如果你要做大模型复核，再看 `vlm/README.md`。

## 当前真实可用的主流程

### 主流程 A：本地窗口版监控

适合调试整条实时检测链路。

- 入口：`predict.py`
- 配置来源：
  - 优先 `config.yaml`
  - 没有的话就用 `predict.py` 里内置默认值
- 主要能力：
  - YOLO 检测
  - ByteTrack 跟踪
  - 规则告警
  - ViT 事件识别

### 主流程 B：网页版监控

适合做多路监控界面。

- 入口：`web/app.py`
- 后台调度：`web/services/runtime_manager.py`
- 主要能力：
  - 管理流列表
  - 启停监控
  - 页面拉取 MJPEG 画面
  - 显示 ViT 和 VLM 的结果

### 主流程 C：只跑 YOLO + 规则

适合暂时不接 ViT，只想先跑目标检测、跟踪和规则。

- 入口：`yolo/run_realtime.py`
- 重点目录：`yolo/realtime/`

## 配置从哪里改

这个项目比较适合在 PyCharm 里直接改代码和配置文件，不需要强依赖命令行参数。

### 你最常会改的地方

| 目标 | 建议修改位置 |
|------|--------------|
| 视频源 | `config.yaml` 的 `sources`，或者 `predict.py` 里的 `_default_sources()` |
| 每路规则 | `config.yaml` 的 `streams`，或者 `predict.py` 里的 `_default_stream_config_map()` |
| YOLO 权重 | `config.yaml` 的 `yolo.model_path` |
| YOLO 类别文件 | `config.yaml` 的 `yolo.classes_path` |
| ViT checkpoint | `config.yaml` 的 `vit.known_checkpoint` |
| 显示大小与阈值 | `config.yaml` 的 `system` |
| 跟踪参数 | `config.yaml` 的 `tracker` |
| Web 端模型路径 | `web/services/runtime_manager.py` 中的默认路径或环境变量读取位置 |
| VLM 训练参数 | `vlm/configs/default.yaml` |

## 根目录关键文件说明

| 文件 | 作用 |
|------|------|
| `predict.py` | 总入口，负责把 `yolo` 和 `Vit` 连起来。 |
| `config.yaml` | 统一配置文件。 |
| `README.md` | 项目总说明。 |

## `yolo/` 架构摘要

### 它负责什么

- 检测目标。
- 跟踪目标。
- 做区域规则判断。
- 把结果画到画面上。

### 它的核心文件

| 路径 | 作用 |
|------|------|
| `yolo/yolo.py` | YOLO 推理类。 |
| `yolo/train.py` | YOLO 训练入口。 |
| `yolo/run_realtime.py` | 只跑 YOLO 实时管线的入口。 |
| `yolo/realtime/pipeline.py` | 实时监控总循环。 |
| `yolo/realtime/yolo_batch.py` | 真批量推理。 |
| `yolo/realtime/bytetrack_wrapper.py` | 跟踪器封装。 |
| `yolo/realtime/display.py` | 画框、画轨迹、画文字。 |

## `Vit/` 架构摘要

### 最新 ViT 主线是什么

这里的“最新 ViT”建议你在 GitHub 上明确写成：

**当前仓库主用的是 `VideoMAE v2` 视频编码器，不是旧文档里那套双流光流主线。**

### 当前主流程

1. `Vit/lab_dataset/labels/video_labels.csv` 维护视频标签。
2. `Vit/lab_anomaly/tool/precompute_clips.py` 先把视频离线切成 clip。
3. `Vit/lab_anomaly/train/train_end2end.py` 训练 `VideoMAE v2 + MIL`。
4. 产物输出到 `Vit/lab_dataset/derived/end2end_classifier/` 一类目录。
5. `Vit/lab_anomaly/infer/known_event_runtime.py` 或 `rtsp_service.py` 用训练结果做实时推理。

### 需要特别提醒的地方

- `README.VIT.md` 里提到的一些脚本和配置，当前仓库里并不完整存在。
- 训练用的帧数、推理用的帧数一定要对齐。
- `predict.py` 里 ViT 默认参数名和 `known_event_runtime.py` 的真实构造参数存在不一致风险，后续如果你要真正跑通，建议专门再核对一遍。

## `vlm/` 架构摘要

### 它在项目里的位置

它不是基础检测模块，而是复核模块。

简单理解：

- YOLO 负责“看到目标”
- ViT 负责“判断一段视频是否异常”
- VLM 负责“把异常讲明白”

### 它的核心内容

| 路径 | 作用 |
|------|------|
| `vlm/data/` | 数据准备、切片、生成训练集。 |
| `vlm/train/` | QLoRA 训练、LoRA 合并、评估。 |
| `vlm/infer/` | 复核推理引擎。 |
| `vlm/configs/default.yaml` | 主配置文件。 |
| `vlm/pycharm/` | 给 PyCharm 用的分步脚本。 |

## `web/` 架构摘要

### 它负责什么

- 提供多路监控页面。
- 管理流配置。
- 把 YOLO、ViT、VLM 结果展示出来。

### 关键文件

| 路径 | 作用 |
|------|------|
| `web/app.py` | Flask 入口。 |
| `web/services/runtime_manager.py` | 后台总调度。 |
| `web/services/frame_state.py` | 保存最新画面和状态。 |
| `web/services/stream_store.py` | 读写流配置。 |
| `web/services/vlm_review_runtime.py` | 大模型复核线程。 |
| `web/config/streams.json` | 流列表持久化文件。 |

## 在 PyCharm 里怎么理解“运行”

考虑到这个项目里很多脚本都有自己的配置区，最适合的方式不是到处传参数，而是：

### 1. 固定解释器

- 解释器环境：`C:\ProgramData\anaconda3\envs\yolovv`

### 2. 按任务打开对应入口文件

| 想做的事 | 建议打开的文件 |
|----------|----------------|
| 跑整套实时监控 | `predict.py` |
| 改统一配置 | `config.yaml` |
| 训练 YOLO | `yolo/train.py` |
| 只调实时检测 | `yolo/run_realtime.py` |
| 预切 ViT 训练片段 | `Vit/lab_anomaly/tool/precompute_clips.py` |
| 训练最新 ViT | `Vit/lab_anomaly/train/train_end2end.py` |
| 跑 Web 页面 | `web/app.py` |
| 训练 VLM | `vlm/pycharm/` 下对应步骤脚本 |

### 3. 尽量在代码里改参数

本仓库很多脚本都已经按“直接改代码配置块”的思路写好了，比如：

- `predict.py`
- `yolo/train.py`
- `Vit/lab_anomaly/train/train_end2end.py`
- `Vit/lab_anomaly/tool/precompute_clips.py`
- `vlm/configs/default.yaml`

## 权重与产物一般放哪

| 类型 | 常见位置 |
|------|----------|
| YOLO 最佳权重 | `yolo/logs/best_epoch_weights.pth` |
| ViT 数据标签 | `Vit/lab_dataset/labels/video_labels.csv` |
| ViT 预切片 | `Vit/lab_dataset/derived/preclips/` |
| ViT 训练产物 | `Vit/lab_dataset/derived/end2end_classifier/` |
| Web 默认 ViT 路径 | `Vit/lab_dataset/derived/known_classifier/checkpoint_best.pt` |
| VLM 训练输出 | `vlm/outputs/qlora/` |
| VLM 合并后模型 | `vlm/outputs/merged/` |

## 已知容易踩坑的地方

### 1. 文档和代码并不完全同步

最明显的是 `Vit/`：

- 老文档写的是 embedding / 光流 / open-set 那条线。
- 当前代码里最完整的是 `precompute_clips + train_end2end + known_event_runtime` 这条线。

### 2. 很多路径是本机绝对路径

这说明仓库更偏本地开发状态，不是完全通用模板。

如果你要发 GitHub，建议别人优先看：

- 哪些路径可以在 YAML 改
- 哪些路径写死在 Python 文件里

### 3. Web 和命令行版不是一回事

- `predict.py` 主要是本地窗口版实时管线
- `web/app.py` 是网页监控版
- 两者都可能用到 `yolo` 和 `Vit`
- 但只有 Web 才会额外接上 `vlm`

### 4. ViT 的训练和推理参数必须对齐

尤其是：

- 每个 clip 多少帧
- 滑窗步长
- 编码器模型名

这些值如果前后不一致，轻则效果变差，重则直接报错。

