## YOLOv8-PyTorch（本仓库“可实际运行版”说明）

本仓库基于 YOLOv8 的 PyTorch 实现，并在**代码层面做过较多本地化修改**（尤其是 Windows 绝对路径、COCO+自定义数据集混合、RTSP 推理等）。
因此，原始 README 中关于 `VOCdevkit` / `voc_annotation.py` 的流程**已不完全适用**。

这份 README 以“你当前仓库里的实际代码”为准，目标是：**按文档照做就能把训练、推理跑通**（不改代码也能跑，但需要按本文准备好目录/文件）。

---

## 目录
- [1. 仓库当前真实结构](#1-仓库当前真实结构)
- [2. 环境与依赖（Windows）](#2-环境与依赖windows)
- [3. 关键资源文件准备（必做）](#3-关键资源文件准备必做)
- [4. 数据集与标注 TXT 说明](#4-数据集与标注-txt-说明)
- [5. 生成标注 TXT（COCO / 自定义 XML）](#5-生成标注-txtcoco--自定义-xml)
- [6. 合并 COCO 与自定义数据（生成 train.txt/val.txt）](#6-合并-coco-与自定义数据生成-traintxtvaltxt)
- [7. 训练（train.py）](#7-训练trainpy)
- [8. 推理（predict.py）](#8-推理predictpy)
- [9. 导出 ONNX（predict.py export_onnx）](#9-导出-onnxpredictpy-export_onnx)
- [10. 评估（get_map.py，注意：需要 VOCdevkit 结构）](#10-评估get_mappy注意需要-vocdevkit-结构)
- [11. 常见问题](#11-常见问题)
- [12. 参考](#12-参考)

---

## 1. 仓库当前真实结构

你现在的仓库核心目录/文件如下（以仓库根目录为准）：

- **训练/推理入口**
  - `train.py`：训练入口（读取 `Datasets/Labels/train.txt` 与 `Datasets/Labels/val.txt`）
  - `yolo.py`：推理核心类（加载权重与类别文件）
  - `predict.py`：单图/视频/文件夹遍历/FPS/热力图/导出 ONNX 的统一入口
- **数据集（当前仓库实际使用的结构）**
  - `Datasets/JPEGImages/train/`、`Datasets/JPEGImages/val/`：图片
  - `Datasets/Annotations/coco/`：COCO JSON（train/val）
  - `Datasets/Annotations/custom/`：自定义 VOC-XML（未按 train/val 拆分，脚本会按文件名去图片目录匹配）
  - `Datasets/Labels/`：各种生成出来的索引/标注 TXT（最终训练用的是 `train.txt` 与 `val.txt`）
- **类别/字体/权重（本仓库实际放在 `Class/`）**
  - `Class/coco_classes.txt`：**82 类**（COCO80 + `fire` + `smoke`，最后两行是 `fire`、`smoke`）
  - `Class/simhei.ttf`：中文字体文件（推理画框用）
  - `Class/yolov8_l.pth`、`Class/yolov8_s.pth`：预训练权重（你本地已经放入）
- **输出**
  - `logs/`：训练日志与权重（`best_epoch_weights.pth` 等）

---

## 2. 环境与依赖（Windows）

### 2.1 先确认 Python 怎么运行

你机器上当前 `python` 命令**可能没有加入 PATH**（终端直接输入 `python` 会报“未识别”）。
但仓库里已经存在一个虚拟环境：`.venv/`，并且包含 `python.exe`。

- **推荐（不依赖系统 PATH）**：直接用 venv 的 Python 运行：

```bash
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip --version
```

- **可选**：激活虚拟环境（PowerShell）：

```bash
.\.venv\Scripts\Activate.ps1
python --version
```

> 如果 PowerShell 提示执行策略限制，可临时放开当前进程：
>
> ```bash
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```

### 2.2 安装依赖

仓库自带 `requirements.txt`（较精简、偏旧）：

```bash
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**重要提示（版本兼容）**
- 代码里启用了 `fp16`（AMP）与 `export_onnx`，通常需要 **PyTorch ≥ 1.7.1** 才比较稳。
- 你当前 `.venv` 的 Python 版本为 **3.14**（见 `.venv/pyvenv.cfg`），而 `requirements.txt` 里固定的 `scipy==1.2.1` 等老包可能在新 Python 上无法安装。

如果你遇到 “装不上/编译失败”，建议新建一个更常见的 Python 环境（例如 Python 3.8/3.9/3.10），再安装 PyTorch 与依赖；本文后续命令依然适用（把 `python` 换成你的解释器路径即可）。

---

## 3. 关键资源文件准备（必做）

### 3.1 `model_data/simhei.ttf`（不改代码也能跑的关键）

当前 `yolo.py` 画框时会读取：`model_data/simhei.ttf`。
但你的仓库里**没有** `model_data/` 目录，字体在 `Class/simhei.ttf`。

不修改代码的前提下，你需要手动准备：

```bash
mkdir model_data
copy .\Class\simhei.ttf .\model_data\simhei.ttf
```

### 3.2 权重与类别文件

当前默认配置（你的代码里已写死）使用：
- 权重：`logs/best_epoch_weights.pth`
- 类别：`Class/coco_classes.txt`（82 类）
- 模型规模：`phi = 'l'`（YOLOv8-L）

如果你的仓库路径不在 `C:\Users\Administrator\Desktop\yolov8-pytorch-master`，请阅读第 7/8 节的“硬编码路径”说明。

---

## 4. 数据集与标注 TXT 说明

### 4.1 图片目录（当前仓库实际使用）

- 训练图片：`Datasets/JPEGImages/train/`
- 验证图片：`Datasets/JPEGImages/val/`

### 4.2 训练/验证索引文件（train.py 实际读取）

`train.py` 当前写死读取：
- `Datasets/Labels/train.txt`
- `Datasets/Labels/val.txt`

这两个 TXT 的每一行格式（与 `utils/dataloader.py` 完全一致）：

```
图片绝对路径 x1,y1,x2,y2,cls_id x1,y1,x2,y2,cls_id ...
```

说明：
- **空格分隔**：第一个字段是图片路径，后面每个字段是一个目标框。
- 框坐标是**像素坐标**：左上角 \((x1,y1)\)，右下角 \((x2,y2)\)。
- `cls_id` 是类别下标，**从 0 开始**，对应 `Class/coco_classes.txt` 的行号。
  - 例如该文件最后两行是 `fire`、`smoke`，则：
    - `fire` 的 `cls_id = 80`
    - `smoke` 的 `cls_id = 81`

> 你当前的 `Datasets/Labels/train.txt` 与 `val.txt` 里既有 `C:/...` 也有 `C:\...` 风格的路径；代码用空格切分、直接 `Image.open(line[0])`，两种写法都可能工作，但**建议统一为一种**，便于排查问题。

---

## 5. 生成标注 TXT（COCO / 自定义 XML）

本仓库当前提供了两条生成标注 TXT 的脚本（都使用了**硬编码 Windows 路径**）：

### 5.1 COCO（`coco_annotation.py`）

用途：读取 COCO 2017 的 `instances_train2017.json` 与 `instances_val2017.json`，合并后按比例切分为 train/val/test，并生成：
- `Datasets/Labels/coco_train.txt`
- `Datasets/Labels/coco_val.txt`
- `Datasets/Labels/coco_test.txt`（仅图片路径）
- `Datasets/Labels/coco_test_images.txt`（仅图片路径）
- `Datasets/Labels/coco_trainval.txt`（train+val 合并）

运行：

```bash
.\.venv\Scripts\python.exe coco_annotation.py
```

注意点（来自脚本逻辑）：
- 脚本会过滤掉**文件名不是以 `"000"` 开头**的图片（非测试集），用于保证与 COCO2017 的命名风格一致。
- 每张图最多保留 `40` 个框，超过会跳过该图片。

### 5.2 自定义 VOC-XML（`extra_annotation.py`）

用途：读取 `Datasets/Annotations/custom/` 下的 XML（VOC 格式），按比例切分为 train/val/test，并生成：
- `Datasets/Labels/custom_train.txt`
- `Datasets/Labels/custom_val.txt`
- `Datasets/Labels/custom_test.txt`（仅图片路径）
- `Datasets/Labels/test_images.txt`（仅图片路径）
- `Datasets/Labels/custom_trainval.txt`（train+val 合并）

图片匹配方式（脚本逻辑）：
- XML 文件名（不带后缀）会去下面两个目录依次找同名图片：
  - `Datasets/JPEGImages/train/`
  - `Datasets/JPEGImages/val/`
- 支持后缀：`.jpg/.jpeg/.png/.bmp`

运行：

```bash
.\.venv\Scripts\python.exe extra_annotation.py
```

注意点：
- 脚本强制要求类别文件中必须存在 `fire` 与 `smoke`（你当前 `Class/coco_classes.txt` 已满足）。
- 数据集划分比例以脚本里的 `train_ratio/val_ratio/test_ratio` 为准（当前代码为 **7:2:1**）。
- 每张图最多保留 `40` 个框，超过会跳过该图片。

---

## 6. 合并 COCO 与自定义数据（生成 train.txt/val.txt）

你当前的训练入口 `train.py` 读取的是 `Datasets/Labels/train.txt` 与 `Datasets/Labels/val.txt`，
这两个文件由 `mixture.py` 合并生成（COCO + 自定义）：

输出：
- `Datasets/Labels/train.txt` = `coco_train.txt` + `custom_train.txt`（合并并打乱）
- `Datasets/Labels/val.txt`   = `coco_val.txt` + `custom_val.txt`（合并并打乱）

运行：

```bash
.\.venv\Scripts\python.exe mixture.py
```

---

## 7. 训练（train.py）

### 7.1 一键训练（按你当前代码默认配置）

在你已经完成：
- 第 3 节（字体 `model_data/simhei.ttf`）
- 第 6 节（生成 `Datasets/Labels/train.txt` 与 `val.txt`）

即可运行训练：

```bash
.\.venv\Scripts\python.exe train.py
```

训练输出：
- 权重与日志保存在 `logs/`
- 你当前代码默认会加载：`logs/best_epoch_weights.pth`（断点续训/继续训练）

### 7.2 你当前代码里写死的关键路径（迁移目录时必改）

你的 `train.py` 当前包含多处类似下面的硬编码（示例）：
- `classes_path`：`...\Class\coco_classes.txt`
- `model_path`：`...\logs\best_epoch_weights.pth`
- `train_annotation_path`：`...\Datasets\Labels\train.txt`
- `val_annotation_path`：`...\Datasets\Labels\val.txt`

如果你把工程放到别的目录（不是 `C:\Users\Administrator\Desktop\yolov8-pytorch-master`），
而又**不想改代码**，那就必须保持工程路径一致；
否则就需要把这些路径改成你机器上的真实路径。

### 7.3 GPU/CPU 与 fp16

训练脚本中：
- `Cuda = True`：没有 GPU 需要改为 `False`
- `fp16 = True`：开启混合精度（需要较新的 PyTorch）

---

## 8. 推理（predict.py）

`predict.py` 把多种推理方式集成在一个脚本里，通过 `mode` 切换：
- `predict`：单张图片路径输入
- `video`：视频文件/摄像头/RTSP
- `fps`：FPS 测试
- `dir_predict`：遍历文件夹并保存结果（你当前默认就是这个）
- `heatmap`：热力图
- `export_onnx`：导出 ONNX

运行：

```bash
.\.venv\Scripts\python.exe predict.py
```

### 8.1 推理前必须确认 `yolo.py` 的配置

`yolo.py` 当前默认使用（硬编码绝对路径）：
- `model_path = ...\\logs\\best_epoch_weights.pth`
- `classes_path = ...\\Class\\coco_classes.txt`
- `phi = 'l'`
- `confidence`、`nms_iou` 等阈值

如果你的路径变化，推理会直接找不到权重/类别文件。

### 8.2 关于 RTSP 地址/账号密码（安全提醒）

你当前的 `predict.py` 在 `mode="video"` 时把 `video_path` 写死为 RTSP 地址（包含账号密码）。
建议你本地使用时自行替换，并且**不要把含密码的地址提交到公开仓库**。

### 8.3 文件夹遍历推理（dir_predict）

你当前默认：
- `mode = "dir_predict"`
- `dir_origin_path` 指向 `Datasets/JPEGImages/val`
- `dir_save_path` 指向 `Datasets/Pre of JPEGImages`

运行后会把推理结果（画框后的图片）保存到 `dir_save_path`。

---

## 9. 导出 ONNX（predict.py export_onnx）

切换：
- `mode = "export_onnx"`
- `onnx_save_path = "model_data/models.onnx"`

运行：

```bash
.\.venv\Scripts\python.exe predict.py
```

注意：
- 需要安装 `onnx`，若开启 `simplify=True` 还需要 `onnxsim`。
- 对 PyTorch 版本通常也有要求（建议 PyTorch ≥ 1.7.1）。

---

## 10. 评估（get_map.py，注意：需要 VOCdevkit 结构）

仓库里提供了 `get_map.py`，但它默认读取：

```
VOCdevkit/VOC2007/ImageSets/Main/test.txt
VOCdevkit/VOC2007/JPEGImages/
VOCdevkit/VOC2007/Annotations/
```

而你的当前仓库根目录下**并不存在 `VOCdevkit/`**。
因此：
- **如果你没有按 VOCdevkit 组织数据集**：`get_map.py` 不能直接用。
- 若你确实需要用 `get_map.py`，请先准备 VOCdevkit 目录结构并放入对应文件，再按脚本要求配置 `classes_path` 等参数。

---

## 11. 常见问题

更多通用排查思路见：`常见问题汇总.md`。

另外，本仓库最常见的“跑不起来”原因来自：
- **硬编码绝对路径**：工程移动目录后，脚本仍指向旧路径。
- **缺失 `model_data/simhei.ttf`**：推理画框时报错。
- **Python/依赖版本不匹配**：尤其是新 Python + 老版本 `scipy/numpy` 组合。

---

## 12. 参考

- `ultralytics/ultralytics`：`https://github.com/ultralytics/ultralytics`
