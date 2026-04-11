# YOLO 模块说明

`yolo/` 是整个项目里最基础的一层，负责把“画面里有什么”先识别出来。

这一层做好之后，上层的跟踪、规则、ViT、网页展示才能继续工作。

## 这一层到底负责什么

简单说，它做 4 件事：

1. 训练目标检测模型。
2. 加载训练好的权重做推理。
3. 把多路视频送进实时检测管线。
4. 给上层提供稳定的检测结果、类别名和画框结果。

## 和主项目的关系

| 位置 | 作用 |
|------|------|
| `moniter/predict.py` | 会直接导入 `yolo.YOLO` 和 `yolo.realtime.pipeline`。 |
| `web/services/runtime_manager.py` | 启动网页监控时，也会调用这里的实时管线。 |
| `Vit/` | 不负责检测本身，而是接在 YOLO 之后看一段视频事件。 |

所以你可以把它理解成：

- `yolo/` 是“基础视觉层”
- `Vit/` 是“视频事件层”
- `vlm/` 是“解释层”
- `web/` 是“展示层”

## 目录结构

```text
yolo/
├── yolo.py
├── train.py
├── predict.py
├── run_realtime.py
├── coco_annotation.py
├── extra_annotation.py
├── mixture.py
├── get_map.py
├── summary.py
├── check.py
├── fix.py
├── Class/
├── logs/
├── model_data/
├── nets/
├── realtime/
├── utils/
└── Datasets/
```

## 顶层文件怎么分工

### `yolo.py`

这是检测推理核心类。

它主要做这些事：

- 加载模型权重。
- 读取类别文件。
- 处理图片输入。
- 调用网络前向。
- 解码检测框并做 NMS。

如果你想修改：

- 权重路径
- 类别文件
- 置信度
- 输入尺寸

通常第一眼应该看这里。

### `train.py`

这是训练入口。

适合在 PyCharm 中直接打开后修改顶部参数，比如：

- `phi`
- `input_shape`
- `Cuda`
- `fp16`
- `mosaic`
- 数据路径
- 预训练权重路径

训练完成后，产物一般会放到 `logs/`。

### `predict.py`

这是离线推理入口。

它支持多种模式，一般通过文件里的 `mode` 切换：

- 单图预测
- 视频预测
- 文件夹预测
- FPS 测试
- 热力图
- ONNX 导出

如果你只是想先快速验证一个模型能不能看懂图片，先看这个文件最合适。

### `run_realtime.py`

这是不接 ViT 的实时入口。

适合这种场景：

- 你先只想跑检测、跟踪、规则。
- 暂时不想接 `Vit/`。
- 想先把视频源、区域、多路逻辑跑通。

### `coco_annotation.py`、`extra_annotation.py`、`mixture.py`

这几个文件负责数据准备。

简单理解：

- `coco_annotation.py`：把 COCO 标注转成当前训练脚本认得的格式。
- `extra_annotation.py`：把自定义 XML 标注转成当前训练格式。
- `mixture.py`：把不同来源的数据合并成最终训练索引。

## 重要子文件夹说明

### `nets/`

放模型结构本体。

主要内容：

- `backbone.py`：骨干网络
- `yolo.py`：检测头和整网结构
- `yolo_training.py`：训练损失

这一层更偏“模型内部结构”。

如果你要改网络规模、特征融合、损失逻辑，这里最重要。

### `utils/`

放训练和推理会复用的工具。

主要内容：

- 数据读取
- 图片预处理
- 框解码
- NMS
- 坐标转换
- 单轮训练逻辑
- 训练日志与评估回调

如果你要查：

- 为什么框坐标不对
- 为什么训练 loss 异常
- 为什么 mAP 没记录

一般先看这里。

### `realtime/`

这是整个 `yolo/` 模块里最重要的业务层。

它把下面这些东西串起来：

- 多路采集
- 队列
- 批量检测
- ByteTrack 跟踪
- ROI 规则
- 滞留告警
- 车辆告警
- 画框和显示

如果你的目标是“把摄像头监控跑起来”，那实际最关键的目录就是这里。

### `Class/`

放类别文件和部分模型相关资源。

最常用的是：

- `coco_classes.txt`

这个文件的行号就是类别编号，所以训练和推理必须对齐。

### `model_data/`

主要放显示相关资源，比如中文字体。

当前常见用途：

- `simhei.ttf`

它主要影响中文横幅、中文标签显示。

### `logs/`

放训练结果。

常见内容：

- 最优权重
- 中间 checkpoint
- loss 记录
- 验证结果

### `Datasets/`

放数据集和训练索引。

这部分不是完全标准化的数据平台，而是更偏“本地工程式数据目录”，所以你在 GitHub README 里最好说明：

- 实际训练前要核对自己的图片路径
- 训练索引是否还是本机绝对路径

## 真实工作流程

### 场景 1：训练检测模型

在 PyCharm 里建议按这个顺序理解：

1. 先准备 `Datasets/` 下的数据和索引。
2. 核对 `Class/coco_classes.txt`。
3. 打开 `train.py` 修改训练参数。
4. 训练产物输出到 `logs/`。
5. 再到 `yolo.py` 或上层入口中切换到新权重。

### 场景 2：先离线测效果

适合先验证单张图、一个视频的检测质量。

建议看：

1. `predict.py`
2. `yolo.py`
3. `Class/coco_classes.txt`

### 场景 3：跑实时多路监控

适合摄像头、监控视频、区域告警。

建议看：

1. `run_realtime.py`
2. `realtime/config.py`
3. `realtime/pipeline.py`
4. `realtime/display.py`

如果你最终要跑的是根目录 `predict.py`，这里仍然是底层主干。

## 实时管线怎么串起来

```text
视频源
  ↓
采集线程
  ↓
推理队列
  ↓
YOLO 批量检测
  ↓
ByteTrack 跟踪
  ↓
规则判断
  ↓
画框与显示
```

对应的关键文件关系：

| 文件 | 作用 |
|------|------|
| `realtime/capture_rtsp.py` | 采集 RTSP。 |
| `realtime/capture_file.py` | 采集本地视频。 |
| `realtime/batch_aggregator.py` | 从队列收集一批帧。 |
| `realtime/yolo_batch.py` | 真批量推理。 |
| `realtime/bytetrack_wrapper.py` | 跟踪。 |
| `realtime/dwell.py` | 滞留统计。 |
| `realtime/alarm.py` | 报警逻辑。 |
| `realtime/display.py` | 画框和提示。 |
| `realtime/pipeline.py` | 总循环。 |

## 训练数据格式要点

当前训练脚本读的是文本索引，不是直接扫目录自动训练。

每一行大致长这样：

```text
图片路径 x1,y1,x2,y2,cls_id x1,y1,x2,y2,cls_id
```

这意味着：

- 第一段是图片路径。
- 后面每一段是一个框。
- `cls_id` 必须和 `Class/coco_classes.txt` 的行号一致。

## 你最可能要改的地方

| 目标 | 主要位置 |
|------|----------|
| 改检测权重 | `yolo.py` |
| 改类别文件 | `yolo.py`、`Class/coco_classes.txt` |
| 改训练参数 | `train.py` |
| 改数据增强 | `utils/dataloader.py`、`train.py` |
| 改检测后处理 | `utils/utils_bbox.py` |
| 改实时流配置 | `run_realtime.py` 或上层 `config.yaml` |
| 改实时显示 | `realtime/display.py` |
| 改 ROI 规则 | `realtime/dwell.py`、`realtime/alarm.py` |

## 这个模块里最容易踩坑的地方

### 1. 路径可能写死了

这个工程有比较明显的本地开发痕迹。

所以你在交给别人前，最好提醒：

- 某些图片路径可能是绝对路径
- 某些权重路径可能是绝对路径
- 某些脚本默认目录是按你本机写的

### 2. 类别名必须完全一致

规则里如果写：

- `person`
- `car`
- `dog`

那它们必须和 `Class/coco_classes.txt` 里的名字一致。

只要大小写或拼写不一样，就会出现：

- 不统计
- 不报警
- 不显示轨迹

### 3. ROI 坐标必须和原始帧一致

很多人会误以为 ROI 要按缩放后图片尺寸写，其实这里主要看原图像素坐标。

如果区域框得不准，第一优先去查：

- 视频原始分辨率
- ROI 坐标来源
- `display.py` 里画出来的位置是不是和想象一致

### 4. 实时入口和根目录入口不是一个东西

- `yolo/run_realtime.py`：只跑 YOLO + 规则
- `moniter/predict.py`：会再往上接 ViT

文档里一定要写清楚，否则很容易混淆。

## 推荐阅读顺序

如果你主要关心：

- 模型结构：先看 `nets/`
- 训练流程：先看 `train.py` + `utils/`
- 实时监控：先看 `realtime/`
- 纯推理：先看 `yolo.py` + `predict.py`

## 本目录下建议继续阅读的文档

- `realtime/README.md`
- `nets/README.md`
- `utils/README.md`

这三个 README 更像是“往里一层”的展开说明。
