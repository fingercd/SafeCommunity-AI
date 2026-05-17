# services — 后台核心目录

> Web 层的"调度中心"，负责读取配置、启停监控、缓存画面与状态、管理 VLM 复核。

---

## 四大职责

1. **读取配置** — 从 `streams.json` 加载流列表
2. **启停监控** — 启动/停止 YOLO + ViT + VLM 实时管线
3. **缓存画面与状态** — 保存每路流的最新帧和检测结果
4. **VLM 复核** — 管理大模型推理任务队列

---

## 核心文件

### `runtime_manager.py` — 总调度器

- 读取 `streams.json` 中启用的流
- 构建 `SourceConfig`、`StreamConfig`、`SystemConfig`、`TrackerConfig`
- 初始化 YOLO 检测器
- 初始化 ViT 异常检测运行时（`KnownEventRuntime`）
- 初始化 VLM 复核引擎（`VlmReviewRuntime`）
- 启动 `run_pipeline` 实时管线线程
- 管理回调链：`on_frame_after_yolo` → `frame_sink`

### `frame_state.py` — 帧状态缓存

- 每路流维护：最新 JPEG 画面、时间戳、状态字典
- MJPEG 端点 `/video/<id>` 从这里取帧
- 状态端点 `/api/streams/<id>/status` 从这里取 JSON
- 流删除时自动清理对应缓存

### `stream_store.py` — 流配置管理

- `add_stream()` — 添加流，自动生成 ID
- `delete_stream()` — 删除流
- `update_stream()` — 更新流配置
- `get_stream()` / `load_streams()` — 读取流信息
- 所有操作自动持久化到 `streams.json`

### `vlm_review_runtime.py` — VLM 复核线程

- 维护 VLM 推理任务队列
- `try_submit_if_triggered()` — ViT 异常触发时提交
- `submit_periodic()` — 周期性自动触发
- `pop_result()` — 获取已完成的推理结果
- 15 秒冷却 + 去重，避免重复分析同一段视频

---

## 数据流

```
YOLO pipeline (run_pipeline)
    ↓ on_frame_after_yolo callback
runtime_manager
    ├─→ vit_runtime.add_frame()      → 异步 ViT 推理
    ├─→ review_runtime.pop_result()  ← 获取 VLM 结果写入 state
    ├─→ review_runtime.set_last_clip() + try_submit_if_triggered()
    │       ↓ ViT 异常 或 周期触发
    │   review_runtime.submit() → VLMEngine.analyze_clip()
    │       ↓
    └─→ frame_sink() → frame_state.update()
            ↓
    /video/<id>  (MJPEG)
    /api/streams/<id>/status  (JSON)
```

---

## 注意事项

- `runtime_manager.py` 会 import `yolo`、`lab_anomaly`、`vlm`，任一模块 import 失败都会打印警告但不会中断启动
- 流配置变更（阈值、ROI、类别等）会触发管线重启，期间画面会短暂中断
- VLM 推理是异步的，提交后可能需要数秒才能拿到结果
