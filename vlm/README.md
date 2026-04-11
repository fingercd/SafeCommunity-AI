# VLM 模块说明

`vlm/` 这一层负责的是“大模型复核”。

它不是整套项目里最底层的必要模块，而是增强模块。  
它的作用是：当 `Vit` 先判断某一段视频可能异常后，再交给大模型给出更详细的解释。

## 一句话理解它在整个项目里的位置

你可以这样看：

- `yolo/`：先看见目标。
- `Vit/`：先筛一段视频是否可疑。
- `vlm/`：再用大模型解释“这段异常更像什么、为什么”。

所以它更像“二次复核层”。

## 这个目录主要在做什么

这里包含一整套从数据准备到训练、再到推理复核的流程。

主要环节有：

1. 整理视频片段数据。
2. 生成训练集。
3. 用 QLoRA 微调大模型。
4. 把 LoRA 合并成完整模型。
5. 在网页监控中做异常片段复核。

## 顶层目录结构

```text
vlm/
├── README.md
├── requirements.txt
├── configs/
├── data/
├── infer/
├── outputs/
├── pycharm/
├── scripts/
├── train/
├── ECVA/
├── Qwen/
└── Qwen 3.5 2b/
```

## 各子目录作用

### `configs/`

放总配置。

其中最重要的是：

- `configs/default.yaml`

这个文件里会集中写：

- 模型路径
- 数据路径
- clip 相关参数
- LoRA 参数
- 训练超参
- 推理阈值

如果你打算用 PyCharm 跑这一整套流程，通常第一件事就是先改这里。

### `data/`

放数据准备相关代码。

它负责：

- 解析原始元数据
- 切分 clip
- 生成自动标注
- 最终整理出训练集 JSON

### `train/`

放训练相关代码。

主要是：

- `train_qlora.py`
- `merge_lora.py`
- `evaluate.py`
- `stress_test_eval.py`

### `infer/`

放推理和部署相关代码。

最关键的是：

- `vlm_engine.py`

它提供统一入口，把一组视频帧送进去，返回结构化分析结果。

### `pycharm/`

这是非常适合你使用的一层。

因为这里就是为“按步骤在 PyCharm 里点运行”准备的薄封装脚本。

简单说，它把完整流程拆成了几个可单独执行的步骤文件。

### `outputs/`

放训练产物。

常见内容：

- `qlora/`
- `merged/`
- 评估结果

### `ECVA/`

通常是数据集目录或实验数据目录。

### `Qwen/` 和 `Qwen 3.5 2b/`

放本地基础模型。

这两个目录在当前项目里是很典型的“本机资源目录”，发 GitHub 时一定要提醒别人改成本机路径。

## 推荐工作方式

考虑到你更适合在 PyCharm 里改配置、点运行，而不是背命令，这个模块建议按下面方式理解。

### 第一步：先改总配置

优先打开：

- `configs/default.yaml`

重点看这些内容：

- 本地大模型目录
- 数据根目录
- 训练输出目录
- clip 长度
- 训练 batch
- 推理阈值

### 第二步：按 PyCharm 步骤脚本走

优先看：

- `pycharm/`

这里通常会把流程拆成：

1. 元数据准备
2. clip 准备
3. 数据集构建
4. QLoRA 训练
5. 评估

这比直接从底层脚本硬跑更适合小白。

## 真实主流程

### 数据阶段

负责把原始视频和元数据整理成模型可训练的数据。

对应目录：

- `data/`

常见结果会落到：

- `data/processed/`

### 训练阶段

负责对大模型做 QLoRA 微调。

对应目录：

- `train/`

主入口：

- `train/train_qlora.py`

### 合并阶段

负责把 LoRA 适配器和基础模型合并。

主文件：

- `train/merge_lora.py`

这个步骤很关键，因为 Web 复核层通常更希望直接读取合并后的完整模型目录。

### 评估阶段

主文件：

- `train/evaluate.py`
- `train/stress_test_eval.py`

### 推理阶段

主文件：

- `infer/vlm_engine.py`
- `infer/rtsp_monitor.py`

## 它和 Web 是怎么接起来的

这一点非常重要。

当前仓库里，`vlm/` 不是由根目录 `predict.py` 直接调用的。  
真正把它接起来的是：

- `web/services/runtime_manager.py`
- `web/services/vlm_review_runtime.py`

工作方式大概是这样：

1. Web 后台先拿到 YOLO 和 ViT 的结果。
2. 如果 ViT 判断这段 clip 可疑，并且超过阈值。
3. 就把这段 clip 送到 `VLMEngine.analyze_clip()`。
4. 返回的结构化结果再显示到页面上。

## `vlm_engine.py` 值得重点说明

这是这个模块最核心的对外接口。

它的输入是：

- 一组视频帧

它的输出是：

- 一个结构化结果字典

结果里一般会包含：

- `classification`
- `reason`
- `result`
- `description`
- `key_sentences`

同时也会兼容一些旧字段，比如：

- `is_anomaly`
- `confidence`
- `reasoning`

这就是为什么 Web 层能比较方便地接它。

## 你最可能要改的地方

| 目标 | 主要位置 |
|------|----------|
| 改基础模型路径 | `configs/default.yaml` |
| 改数据路径 | `configs/default.yaml` |
| 改训练参数 | `configs/default.yaml`、`train/train_qlora.py` |
| 改输出目录 | `configs/default.yaml` |
| 改推理阈值 | `configs/default.yaml`、Web 调用层 |
| 改复核逻辑 | `infer/vlm_engine.py`、`web/services/vlm_review_runtime.py` |

## 小白最容易踩的坑

### 1. 路径基本都是本机路径

这个模块非常明显带有本地实验痕迹。

尤其是：

- 模型目录
- 数据目录
- 数据集根路径

换机器后要优先改配置。

### 2. Web 更偏向读取“合并后的完整模型”

也就是说，网页复核层更常期待：

- `outputs/merged/`

而不是只给一个 LoRA adapter 目录。

### 3. 显存压力会比较大

这里毕竟是大模型，不像纯 YOLO 那么轻。

真正影响显存的常见项包括：

- clip 长度
- 图像大小
- 量化方式
- batch 大小
- gradient accumulation

### 4. `predict.py` 不直接等于 `vlm`

很多人会误以为主入口已经把大模型也接好了。

实际上当前更清楚的关系是：

- 根目录 `predict.py`：YOLO + ViT
- `web/`：YOLO + ViT + 可选 VLM 复核

## 推荐阅读顺序

如果你只是想看懂这个模块，建议顺序：

1. `configs/default.yaml`
2. `pycharm/`
3. `data/README.md`
4. `train/README.md`
5. `infer/README.md`

这样最容易从“怎么用”一路看到“底层怎么实现”。
