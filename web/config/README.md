# config — 配置目录

> 存放 Web 层的持久化配置文件。

---

## 核心文件

### `streams.json`

**最重要的配置文件**，保存所有视频流的配置信息。

由 `stream_store.py` 自动读写，网页上的增删改操作都会实时反映到这个文件。

### 保存的字段

每个流对象包含：
- `id` — 流唯一标识
- `name` — 显示名称
- `rtsp_url` — 视频源地址
- `enabled` — 是否启用
- `resize_width` / `resize_height` — 显示分辨率
- `vit_threshold` — ViT 异常触发阈值
- `yolo_confidence` — YOLO 检测置信度
- `agent_enabled` — 是否启用 VLM 复核
- `vlm_auto_interval_sec` — VLM 自动触发间隔
- `rois` — ROI 禁区多边形
- `roi_alarm_classes` — ROI 闯入告警类别
- `global_alarm_classes` — 全局禁现类别

---

## 注意事项

1. **敏感信息**：RTSP 地址可能包含用户名密码，发 GitHub 前注意脱敏
2. **字段有效性**：部分字段可能是历史遗留，需对照 `runtime_manager.py` 确认是否被读取和使用
3. **自动持久化**：不需要手动编辑这个文件，Web 端操作会自动同步
