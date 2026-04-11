# Vit 模块说明

`Vit/` 这一层负责的不是“检测画面里有什么”，而是“看一小段视频里发生了什么”。

在整个项目里，它是事件识别层。

## 先说最重要的一句话

**当前仓库里真正完整、能和代码对应上的最新 ViT 主线，是 `VideoMAE v2 + MIL + 端到端训练`。**

这一点很重要，因为 `Vit/` 目录里还保留了一些历史说明文档，里面写的是另一套更老的流程。  
如果你以后把这个项目放到 GitHub，上来就应该把这句话写清楚，避免别人一开始就走错方向。

## 这个模块在整个项目里的位置

你可以这样理解：

- `yolo/` 先找出目标。
- `Vit/` 再看连续的一段视频片段，判断是不是异常事件。
- `vlm/` 在需要时再进一步做文字复核。

也就是说，`Vit/` 处理的是“时间上的变化”，不是单帧检测。

## 顶层结构

```text
Vit/
├── README.md
├── README.VIT.md
├── docs/
├── lab_anomaly/
└── lab_dataset/
```

## 每个子目录是干什么的

### `lab_anomaly/`

这是核心代码目录。

主要负责：

- 读取视频片段数据
- 定义 VideoMAE v2 编码器
- 定义 MIL 分类头
- 训练端到端事件分类模型
- 提供实时推理运行时

如果你只保留一个目录来代表“ViT 真正的实现”，那就是它。

### `lab_dataset/`

这是数据目录约定。

主要负责：

- 保存原始视频
- 保存 `video_labels.csv`
- 保存预切片结果
- 保存训练产物

### `docs/`

这是补充说明目录。

适合做：

- 学习笔记
- 代码功能总览
- 路径修改说明

但这里有一个现实问题：

**部分文档描述的是旧流程，不完全等于当前代码。**

所以在对外 README 里，应该把它定义为“补充资料”，而不是“唯一正确流程”。

### `README.VIT.md`

这是历史留下来的训练流程说明。

它有参考价值，但不能直接当成当前仓库的真实主线。

因为它写的是：

- 双流
- 光流预计算
- embedding 提取
- 开放集边界

而当前仓库里最完整的主线并不是这一套。

## 当前最新 ViT 主线

### 真实流程

当前建议在 GitHub 上这样介绍：

1. 准备视频和标签。
2. 用 `precompute_clips.py` 先离线切 clip。
3. 用 `train_end2end.py` 训练 `VideoMAE v2 + MIL`。
4. 用 `known_event_runtime.py` 或 `rtsp_service.py` 做实时推理。

### 真实对应文件

| 路径 | 作用 |
|------|------|
| `lab_anomaly/tool/precompute_clips.py` | 先把视频切成训练要用的 clip。 |
| `lab_anomaly/train/train_end2end.py` | 端到端训练主入口。 |
| `lab_anomaly/models/vit_video_encoder.py` | VideoMAE v2 编码器封装。 |
| `lab_anomaly/models/mil_head.py` | MIL 分类头。 |
| `lab_anomaly/infer/known_event_runtime.py` | 实时推理运行时。 |
| `lab_anomaly/infer/rtsp_service.py` | RTSP 服务版推理。 |
| `lab_anomaly/infer/scoring.py` | checkpoint 加载和打分。 |

## 为什么说这是“最新主线”

因为当前仓库里，这条链最完整：

- 有数据读法
- 有训练入口
- 有模型实现
- 有推理入口
- 有配置文件

而旧流程里提到的很多脚本，在当前仓库里已经不完整了，或者压根找不到对应文件。

## 目录深入说明

## `lab_anomaly/`

这是最重要的目录。

你可以把它再拆成 5 层来理解。

### 1. `data/`

负责把视频和标签变成模型可读的数据。

常见职责：

- 读取 `video_labels.csv`
- 读取视频帧
- 构造 clip 数据集
- 建立索引

### 2. `models/`

负责模型结构。

主要是两块：

- `vit_video_encoder.py`
  - 封装 `VideoMAE v2`
  - 负责把一段视频变成特征向量

- `mil_head.py`
  - 负责把多个 clip 的特征聚合起来
  - 最终输出正常/异常结果

### 3. `tool/`

主要是训练前准备工具。

当前最关键的是：

- `precompute_clips.py`

它不是训练本身，而是先把视频切成后续训练能直接复用的 `.npz` clip。

### 4. `train/`

负责训练。

当前最关键的是：

- `train_end2end.py`

它是目前最值得在 README 里重点讲的训练入口。

### 5. `infer/`

负责推理和部署。

常见入口：

- `known_event_runtime.py`
- `rtsp_service.py`
- `scoring.py`

如果你想把 ViT 挂进实时监控系统，这一层最关键。

## `lab_dataset/`

这里不是模型代码，而是数据约定。

### 建议理解方式

- `raw_videos/`：原始视频
- `labels/`：标签文件
- `derived/`：中间产物和训练产物

### 重点文件

| 路径 | 作用 |
|------|------|
| `lab_dataset/labels/video_labels.csv` | 视频级标签清单。 |
| `lab_dataset/derived/preclips/` | 离线预切后的 clip。 |
| `lab_dataset/derived/end2end_classifier/` | 当前主线训练产物。 |

## 当前训练主流程，按 PyCharm 思路解释

### 第一步：准备视频标签

你需要先确保：

- `lab_dataset/labels/video_labels.csv` 存在
- 每一行视频路径是对得上的
- `label` 字段能区分正常和异常

### 第二步：预切片

打开：

- `lab_anomaly/tool/precompute_clips.py`

在文件顶部配置区改好：

- 数据根目录
- 标签 CSV
- 输出目录
- 每个 clip 多少帧
- 采样间隔
- 每个视频最多切多少个 clip

它会把结果写到：

- `lab_dataset/derived/preclips/`

### 第三步：端到端训练

打开：

- `lab_anomaly/train/train_end2end.py`

这里是当前主训练入口。

你需要重点看这些参数：

- `dataset_root`
- `labels_csv`
- `preclip_root`
- `out_dir`
- `frames_per_clip`
- `encoder_model_name`
- `batch_size`
- `stages`
- `val_ratio`

### 第四步：用训练结果做推理

训练完成后，结果一般输出到：

- `lab_dataset/derived/end2end_classifier/`

然后可以由下面两种方式使用：

- `known_event_runtime.py`：嵌入到整个监控工程
- `rtsp_service.py`：单独作为 RTSP 推理服务

## 当前主模型是怎么组成的

### 1. 视频编码器

由 `vit_video_encoder.py` 提供。

它本质上是：

- 载入预训练的 `VideoMAE v2`
- 把一段视频 clip 编码成向量

### 2. MIL 头

由 `mil_head.py` 提供。

它负责：

- 把一个视频里的多个 clip 特征汇总
- 输出最终分类结果

### 3. 排序损失

由 `ranking_loss.py` 提供。

它主要是为了帮助模型更稳定地区分正常和异常片段。

## 当前配置文件有哪些

### `lab_anomaly/configs/train_end2end.yaml`

这是当前主训练配置文件。

它和 `train_end2end.py` 的关系是：

- Python 文件里先有一套默认值
- YAML 再去覆盖它

这意味着：

- 你不能只改一边不看另一边
- 如果 YAML 里没写某个值，就会回退到 Python 默认值

### `lab_anomaly/configs/rtsp_service_example.yaml`

这是推理服务示例配置。

适合查看：

- checkpoint 放哪
- 输出日志放哪
- RTSP 服务期望的字段有哪些

## 当前目录里最容易误解的地方

### 1. 旧文档还在，但不代表就是当前主线

比如 `README.VIT.md` 里说的是：

- 光流
- embedding
- 已知类分类
- KMeans + OCSVM

这套流程作为历史背景可以看，但不应该当作当前仓库唯一正确流程。

### 2. 文档中的某些脚本，当前仓库里并不存在

对外写 README 时，最好不要把这些脚本再当成核心入口。

否则别人一搜发现文件没有，会直接怀疑仓库不完整。

### 3. 训练和推理的帧数必须一致

比如：

- `frames_per_clip`
- `clip_len`
- 滑窗设置

这些值一旦前后不一致，很容易出问题。

### 4. 模型名要前后一致

当前实现更偏向：

- `OpenGVLab/VideoMAEv2-Base`

而不是旧文档里常出现的另一套名字。

README 里最好明确说明：

- 以当前训练脚本和 checkpoint 实际配置为准

## 你如果是小白，建议怎么读

最省事的顺序是：

1. 先看本文件，搞清楚当前主线不是旧文档那套。
2. 再看 `lab_anomaly/README.md`，知道代码怎么分层。
3. 再看 `lab_dataset/README.md`，知道数据该怎么摆。
4. 如果还想看历史背景，再去看 `README.VIT.md` 和 `docs/`。

## 推荐你在 GitHub 上怎么介绍“最新 ViT”

可以直接用下面这句话当摘要：

> 本项目当前的视频事件识别主线基于 `VideoMAE v2 + MIL`。训练流程采用“先离线预切 clip，再端到端训练，再挂入实时推理”的方式；仓库中保留的旧版双流光流文档仅作历史参考。

这句话非常重要，能帮读者少走很多弯路。
