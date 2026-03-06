# ByteTrack 实时视频分析系统 — PyCharm 操作指南

本文档面向**不使用命令行**的 PyCharm 用户，按步骤说明如何从打开项目到成功运行实时分析系统。每步包含「操作」和「预期结果」。

---

## 一、项目结构设置

### 步骤 1.1：在 PyCharm 中打开项目

| 操作 | 预期结果 |
|------|----------|
| 1. 打开 PyCharm。 | PyCharm 主窗口出现。 |
| 2. 点击菜单 **File → Open**（或欢迎界面的 **Open**）。 | 弹出文件夹选择对话框。 |
| 3. 选择本项目的**根目录**（即包含 `run_realtime.py`、`yolo.py`、`realtime` 文件夹、`nets` 文件夹的那一层），点击「确定」。 | 项目以根目录为根在 PyCharm 中打开，左侧项目树显示所有文件。 |

**重要**：必须打开的是「项目根目录」，而不是其子文件夹（例如不要只打开 `realtime`）。

---

### 步骤 1.2：确认项目文件结构

| 操作 | 预期结果 |
|------|----------|
| 在左侧 **Project** 面板中展开目录树，确认结构大致如下。 | 能看到下表所列关键文件/文件夹。 |

**必须包含的关键内容：**

| 路径 | 说明 |
|------|------|
| `run_realtime.py` | 运行入口，在**项目根目录**下。 |
| `yolo.py` | 项目自带的 YOLO 推理，在根目录。 |
| `realtime/` 文件夹 | 实时分析模块（ByteTrack、ROI、报警等）。 |
| `realtime/config.py` | 配置类定义。 |
| `realtime/pipeline.py` | 主流程。 |
| `nets/` | YOLO 网络定义。 |
| `utils/` | 工具函数（图像、解码等）。 |
| `Class/` | 类别名文件（如 `coco_classes.txt`）。 |
| `logs/` 或 `model_data/` | 权重或模型相关（按你当前项目实际为准）。 |

若缺少 `realtime` 或 `run_realtime.py`，请先补全再继续。

---

### 步骤 1.3：设置运行工作目录为项目根目录（关键）

| 操作 | 预期结果 |
|------|----------|
| 1. 在 Project 面板中**右键点击** `run_realtime.py`。 | 弹出右键菜单。 |
| 2. 选择 **Run 'run_realtime'**（或 **Modify Run Configuration...** 若已运行过）。 | 若为首次运行，会直接运行；若选择修改，则进入运行配置界面。 |
| 3. 若要修改配置：点击顶部菜单 **Run → Edit Configurations...**，在左侧选中 **Python** 下与 `run_realtime` 对应的项（或点击 **+** → **Python** 新建，**Script path** 选为 `run_realtime.py`）。 | 右侧出现该运行配置的详细选项。 |
| 4. 找到 **Working directory**（工作目录）。若为空白或不是项目根，点击右侧文件夹图标，选择**本项目根目录**（与步骤 1.1 中打开的同一文件夹）。 | **Working directory** 显示为项目根目录的完整路径。 |
| 5. 点击 **Apply**，再点击 **OK**。 | 配置保存；之后运行 `run_realtime.py` 时，当前工作目录均为项目根目录。 |

**说明**：本系统在运行时会执行 `from yolo import YOLO`、`from realtime.config import ...` 等，只有在「项目根目录」下运行时，Python 才能正确找到 `yolo`、`realtime`、`utils` 等模块。工作目录设错会导致 `ModuleNotFoundError`。

---

## 二、依赖安装

本系统依赖项目根目录 `requirements.txt` 中的包（如 `torch`、`opencv-python`、`numpy`、`Pillow` 等），**不依赖** `ultralytics`。请在 PyCharm 中为当前项目安装这些包。

### 步骤 2.1：打开 Python 解释器设置

| 操作 | 预期结果 |
|------|----------|
| 1. 点击菜单 **File → Settings**（Windows/Linux）或 **PyCharm → Preferences**（macOS）。 | 打开设置窗口。 |
| 2. 在左侧展开 **Project: 你的项目名**，点击 **Python Interpreter**。 | 右侧显示当前项目使用的 Python 解释器及已安装包列表。 |

**界面描述**：右侧会看到「Project Interpreter」下拉框、解释器路径、以及一个包列表（表格式），列表下方有 **+**、**-**、**↻** 等按钮。

---

### 步骤 2.2：安装所需包

| 操作 | 预期结果 |
|------|----------|
| 1. 点击包列表上方的 **+**（Install New Package）。 | 弹出「Available Packages」窗口。 |
| 2. 在搜索框中输入 **torch**，在结果中选中 **torch**，勾选 **Specify version** 可按需选择版本，点击 **Install Package**。 | 开始安装；若已安装则显示已存在。 |
| 3. 同上，依次搜索并安装：**torchvision**、**opencv-python**、**numpy**、**Pillow**、**scipy**、**tqdm**、**matplotlib**、**h5py**、**tensorboard**（与 `requirements.txt` 一致即可）。 | 每个包安装完成后，在「Python Interpreter」页的包列表中能看到对应包名及版本。 |
| 4. 关闭「Available Packages」窗口；在 Settings 中点击 **OK**。 | 设置保存。 |

若你已用命令行或其它方式装过依赖，只要在 PyCharm 的 **Project → Python Interpreter** 里能看到上述包即可，无需重复安装。

---

### 步骤 2.3：确认安装成功

| 操作 | 预期结果 |
|------|----------|
| 在 **File → Settings → Project → Python Interpreter** 的包列表中，确认存在 **torch**、**opencv-python**、**numpy**、**Pillow** 等，且无红色报错。 | 列表正常显示包名和版本号，无「未安装」或错误标记。 |

---

## 三、配置修改（核心）

运行前必须在 **项目根目录下的 `run_realtime.py`** 中，按你的实际需求修改：**视频源（RTSP/本地文件）**、**ROI 坐标**、**停留统计类别**、**报警类别**。下面按项说明并给出可直接参考的示例。

---

### 3.1 修改视频源（RTSP 与本地视频）

**含义**：  
- 每个视频源用 `SourceConfig(uri=..., type=..., enabled=...)` 表示。  
- `uri`：RTSP 填完整地址（可含用户名密码）；本地视频填本机路径。  
- `type`：`"rtsp"` 或 `"file"`。  
- `enabled=True` 表示启用，`False` 表示暂时不拉该路。

**操作**：在 PyCharm 中打开 **项目根目录** 下的 `run_realtime.py`，找到 `main()` 里的 `sources = [...]`，改成你的源。

**示例（按需保留/删除/复制多路）：**

```python
sources = [
    # RTSP：用户名 admin，密码 password，IP 192.168.1.100，端口 554，子码流 stream1
    SourceConfig(
        uri="rtsp://admin:password@192.168.1.100:554/stream1",
        type="rtsp",
        enabled=True,
    ),
    # 本地视频文件（Windows 路径示例）
    SourceConfig(
        uri="D:/videos/test.mp4",
        type="file",
        enabled=True,
    ),
]
```

**注意**：  
- RTSP 的 `uri` 必须和后面 **ROI 配置里用的 key 完全一致**（见 3.2）。  
- 本地路径用正斜杠 `/` 或双反斜杠 `\\` 均可。

---

### 3.2 修改 ROI 坐标

**含义**：  
- ROI（感兴趣区域）用「多边形」表示，用于统计目标在区域内的停留时间。  
- 每个视频源可以有一个或多个 ROI；每个 ROI 是一串 **(x, y)** 像素点，顺序首尾闭合即可。  
- 坐标基于**该路视频的原始分辨率**（例如 1920×1080 或 1280×720），与 YOLO 内部 resize 无关。

**如何得到坐标：**  
1. 用播放器或 OpenCV 打开该路视频，看分辨率（宽 W、高 H）。  
2. 在画面里想划定的区域上，用「左上 → 右上 → 右下 → 左下」取 4 个点，记下像素 (x,y)。  
3. 若暂时不做停留统计，可把该源对应的 ROI 设为**空列表** `[]`。

**操作**：在 `run_realtime.py` 里找到 `roi_config = {...}`，按 `stream_id`（即上面的 `uri`）配置。

**示例：**

```python
roi_config = {
    # 与 sources 里的 uri 完全一致
    "rtsp://admin:password@192.168.1.100:554/stream1": [
        # 第一个 ROI：矩形，分辨率 1920x1080 下的 (100,100)-(500,400)
        [(100, 100), (500, 100), (500, 400), (100, 400)],
        # 可再加第二个 ROI：[(x1,y1), (x2,y2), ...]
    ],
    "D:/videos/test.mp4": [],  # 不统计该路的 ROI 停留，留空
}
```

**预期结果**：保存文件后，运行时会只对配置了非空 ROI 的流做停留统计。

---

### 3.3 修改停留统计类别与报警类别

**含义**：  
- **停留统计**：只有类别名在 `dwell_classes` 里的目标（如人、包）才会计入 ROI 停留时间。  
- **报警**：只有类别名在 `alarm_classes` 里的目标（如猫、狗）出现时才会触发声音和日志。  
- 类别名**必须**与项目里 YOLO 使用的类别文件一致（如 `Class/coco_classes.txt`），**区分大小写**。

**操作**：在 `run_realtime.py` 里找到 `dwell_classes`、`alarm_classes`，改成你需要的类别名集合。

**示例（COCO 80 类 + fire + smoke 时）：**

```python
# 只对「人」「背包」统计在 ROI 内的停留时间
dwell_classes = {"person", "backpack"}

# 检测到「猫」「狗」时触发报警（声音 + 日志）
alarm_classes = {"cat", "dog"}
```

若你的类别文件是自定义的（例如只有 `good`、`bad`），则这里应写：

```python
dwell_classes = {"good"}   # 只统计 good 的停留
alarm_classes = {"bad"}   # 检测到 bad 就报警
```

**预期结果**：保存后运行，只有这些类别会参与停留统计和报警。

---

### 3.4 完整配置示例（复制参考）

下面是一段可直接替换进 `run_realtime.py` 的 `main()` 配置示例（仅作格式参考，请按你的 RTSP、路径、类别修改）：

```python
def main():
    # ---------- 1. 视频源（RTSP / 本地文件） ----------
    sources = [
        SourceConfig(
            uri="rtsp://admin:password@192.168.1.100:554/stream1",
            type="rtsp",
            enabled=True,
        ),
        SourceConfig(
            uri="D:/videos/test.mp4",
            type="file",
            enabled=True,
        ),
    ]

    # ---------- 2. ROI：每个流的 uri 对应一组多边形，空列表表示不统计 ----------
    roi_config = {
        "rtsp://admin:password@192.168.1.100:554/stream1": [
            [(100, 100), (500, 100), (500, 400), (100, 400)],
        ],
        "D:/videos/test.mp4": [],
    }

    # ---------- 3. 停留统计类别 / 报警类别（名字要和 Class/*.txt 里一致） ----------
    dwell_classes = {"person", "backpack"}
    alarm_classes = {"cat", "dog"}

    stream_cfg = build_stream_config_map(
        sources, roi_config, dwell_classes, alarm_classes,
    )
    sys_cfg = SystemConfig(max_batch=8, max_wait_ms=40.0)
    tracker_cfg = TrackerConfig(track_high_th=0.5, track_low_th=0.1)

    yolo = YOLO()
    class_names = yolo.class_names

    stop = threading.Event()
    run_pipeline(
        sources=sources,
        stream_cfg=stream_cfg,
        sys_cfg=sys_cfg,
        tracker_cfg=tracker_cfg,
        yolo=yolo,
        class_names=class_names,
        save_duration=save_duration,
        log_stream_end=lambda sid: logging.info("Stream ended: %s", sid),
        stop_event=stop,
    )
```

---

## 四、运行与停止

### 步骤 4.1：运行

| 操作 | 预期结果 |
|------|----------|
| 1. 确认 **Run → Edit Configurations** 里，运行 `run_realtime.py` 的 **Working directory** 为项目根目录（见 1.3）。 | 避免出现 `ModuleNotFoundError`。 |
| 2. 在 Project 面板中右键 **run_realtime.py**，选择 **Run 'run_realtime'**；或打开该文件后点击右上角绿色运行按钮。 | 控制台（Run 窗口）开始输出日志；若有 RTSP/视频，会开始拉流、检测、跟踪、停留统计与报警。 |
| 3. 观察控制台是否有 `INFO` 或报警相关日志。 | 正常时无报错；有报警类别出现时会触发声音并打印报警日志。 |

---

### 步骤 4.2：停止

| 操作 | 预期结果 |
|------|----------|
| 在 Run 窗口左侧点击**红色方块**停止按钮，或使用 **Run → Stop**。 | 进程结束，采集与推理线程退出。 |

---

## 五、常见问题速查

| 现象 | 处理 |
|------|------|
| `ModuleNotFoundError: No module named 'yolo'` 或 `No module named 'realtime'` | 运行配置里的 **Working directory** 未设为项目根目录，按 **1.3** 重新设置。 |
| RTSP 连不上 | 检查 IP、端口、用户名密码、网络；RTSP URL 与 `roi_config` 的 key 是否完全一致。 |
| 没有报警 / 没有停留统计 | 检查 `alarm_classes` / `dwell_classes` 是否与 `Class/*.txt` 中类别名**完全一致**（大小写、空格）。 |
| 报错找不到 `utils` | 同样确认工作目录是**项目根目录**（包含 `utils` 文件夹的那一层）。 |

---

## 六、小结

1. **项目**：用 PyCharm 打开**项目根目录**，保证 `run_realtime.py` 与 `realtime/` 在同一层级。  
2. **工作目录**：运行 `run_realtime.py` 时，**Working directory** 必须为项目根目录。  
3. **依赖**：在 **File → Settings → Project → Python Interpreter** 中安装 `torch`、`opencv-python`、`numpy`、`Pillow` 等（见 `requirements.txt`），无需 `ultralytics`。  
4. **配置**：在 `run_realtime.py` 里改 `sources`（RTSP/本地路径）、`roi_config`（ROI 多边形）、`dwell_classes` 与 `alarm_classes`（与类别文件一致）。  
5. **运行**：右键 `run_realtime.py` → **Run 'run_realtime'**，在 Run 窗口查看日志与报警。

按上述步骤操作后，即可在**不使用命令行**的情况下，用 PyCharm 正常运行 ByteTrack 实时视频分析系统。
