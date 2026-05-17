# web — Web 监控模块

> 把整个 AI 监控系统变成可在浏览器中访问的多路监控面板。用 Flask 提供 REST API 和 MJPEG 实时视频流，前端展示检测结果、异常状态与大模型复核结论。

---

## 这个模块能做什么

| 能力 | 说明 |
|------|------|
| **多路监控** | 同时展示多路视频的实时画面（MJPEG 流）|
| **流管理** | Web 端添加、删除、启停视频流，配置持久化 |
| **状态看板** | 每路流显示运行状态、YOLO 检测、ViT 异常结果、VLM 复核结论 |
| **ROI 绘制** | 在视频帧快照上绘制多边形禁区 |
| **实时配置** | 修改阈值、类别、ROI 后保存，后端自动重启管线实时生效 |
| **精致 UI** | Glassmorphism 玻璃拟态设计，卡片动画、渐变按钮、可折叠 VLM 面板 |

---

## 技术架构

```
浏览器
    ↓ HTTP
Flask (app.py)
    ↓
MonitorRuntimeManager (runtime_manager.py)
    ├─→ YOLO 实时管线（多路视频检测 + 跟踪 + 规则）
    ├─→ ViT 异步异常检测（滑动窗口推理）
    ├─→ VLM 复核引擎（异常触发 / 周期性自动触发）
    └─→ FrameStateCache (frame_state.py) ← 缓存最新帧和状态
    ↓
浏览器轮询 / MJPEG 流
```

---

## 目录结构

```
web/
├── README.md                  # 本文件
├── app.py                     # Flask 入口（REST API + MJPEG 流）
│
├── config/
│   ├── README.md              # 配置说明
│   └── streams.json           # 流列表持久化（运行时自动生成）
│
├── services/                  # 后台核心逻辑
│   ├── runtime_manager.py     # 总调度：串起 YOLO + ViT + VLM
│   ├── frame_state.py         # 帧状态缓存（最新 JPEG + 状态 JSON）
│   ├── stream_store.py        # 流配置读写（CRUD + 持久化）
│   └── vlm_review_runtime.py  # VLM 复核线程管理
│
├── static/                    # 前端资源
│   ├── app.js                 # 前端核心：卡片渲染、轮询、弹窗、ROI 绘制
│   └── style.css              # Glassmorphism 主题（CSS 变量 + 动画）
│
└── templates/
    └── index.html             # 监控面板单页应用
```

---

## 核心后台文件

### `runtime_manager.py` — 总调度器

- 读取 `streams.json` 中启用的流列表
- 启动 `yolo.realtime.pipeline` 实时管线
- 加载 ViT 异常检测运行时（`known_event_runtime.py`）
- 加载 VLM 复核引擎（优先 `vlm/outputs/merged/`，回退 `vlm/Qwen/`）
- 管理回调：`on_frame_after_yolo`（送帧给 ViT）、`frame_sink`（缓存结果）
- 流变更时自动 `stop()` + `start()` 重启管线

### `frame_state.py` — 帧状态缓存

- 保存每路流的**最新 JPEG 画面**
- 保存每路流的**状态 JSON**（运行状态、ViT 结果、VLM 结论、错误信息）
- MJPEG 端点从这里取帧
- 前端状态轮询从这里取数据

### `stream_store.py` — 流配置管理

- 提供流的增删改查（CRUD）
- 配置持久化到 `web/config/streams.json`
- 每个流包含：名称、RTSP 地址、启用状态、阈值、ROI、类别等

### `vlm_review_runtime.py` — VLM 复核线程

- 管理 VLM 推理任务队列
- 双触发机制：
  1. **ViT 异常触发**：当 ViT 判定异常且超过阈值时提交 clip
  2. **周期性自动触发**：按 `vlm_auto_interval_sec` 间隔自动提交，独立于 ViT
- 15 秒冷却时间，避免重复提交

---

## API 端点

### 流管理

| 方法 | 端点 | 说明 |
|------|------|------|
| `GET` | `/api/streams` | 列出所有流 + ViT/VLM 加载状态 |
| `POST` | `/api/streams` | 添加新流 |
| `DELETE` | `/api/streams/<id>` | 删除流 |
| `POST` | `/api/streams/<id>/start` | 启动流 |
| `POST` | `/api/streams/<id>/stop` | 停止流 |
| `PATCH` | `/api/streams/<id>` | 更新配置（阈值、ROI、类别等）|
| `GET` | `/api/streams/<id>/status` | 获取流状态 + 检测结果 |

### 视频与数据

| 方法 | 端点 | 说明 |
|------|------|------|
| `GET` | `/video/<id>` | MJPEG 实时视频流 |
| `GET` | `/api/streams/<id>/snapshot` | 最新帧 JPEG（用于 ROI 绘制）|
| `GET` | `/api/classes` | YOLO 可检测的所有类别 |

---

## 默认模型依赖

启动时会自动尝试加载：

| 模型 | 默认路径 | 环境变量覆盖 |
|------|----------|--------------|
| YOLO 权重 | `yolo/logs/best_epoch_weights.pth` | `YOLO_WEIGHTS` |
| YOLO 类别 | `yolo/Class/coco_classes.txt` | `YOLO_CLASSES` |
| ViT checkpoint | `Vit/lab_dataset/derived/end2end_classifier/checkpoint_best.pt` | `VIT_CHECKPOINT` |
| VLM（优先） | `vlm/outputs/merged/` | `VLM_MERGED` |
| VLM（回退） | `vlm/Qwen/` | `VLM_BASE` |

如果模型缺失，会打印警告但继续运行（对应功能不可用）。

---

## 启动方式

### 方式 1：一键启动（推荐）

```bash
python launch.py
```

自动启动 Flask 服务并打开浏览器。

### 方式 2：手动启动

```bash
python -m web.app
```

然后手动访问 `http://127.0.0.1:5000`。

---

## 注意事项

1. **依赖底层模块**：`runtime_manager.py` 会 import `yolo`、`lab_anomaly`、`vlm`，如果环境没装好会直接报错
2. **流变更触发重启**：新增/删除/启停流、修改阈值或 ROI 等字段时，后端会自动 `stop()` + `start()` 整个管线，前端会短暂黑屏
3. **前后端阈值可能不一致**：Web 端显示的阈值和实际生效的阈值以 `runtime_manager.py` 启动时读取的为准，修改配置后保存才会同步
4. **浏览器缓存**：前端更新后按 `Ctrl + F5` 强制刷新
