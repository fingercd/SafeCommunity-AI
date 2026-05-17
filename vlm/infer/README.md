# infer — 推理目录

> 负责 VLM 模块的推理与部署，是 Web 层直接调用的能力来源。

---

## 核心文件

### `vlm_engine.py`

**整个 VLM 模块最重要的对外接口。**

- **输入**：一组视频帧（RGB，list of PIL.Image 或 numpy array）
- **处理**：构建 Prompt → 送入 Qwen-VL → 解析输出
- **输出**：结构化结果字典

#### 输出字段

| 字段 | 说明 |
|------|------|
| `classification` | 异常分类，如 `"打架"`、`"火灾"`、`"正常"` |
| `reason` | 30 字以内中文说明 |
| `result` | 兼容字段（同 classification）|
| `description` | 兼容字段（详细描述）|
| `key_sentences` | 兼容字段（关键词句）|
| `is_abnormal` | bool，是否异常（`classification` 为 `"正常"` 时为 False）|
| `confidence` | 兼容字段（置信度）|

#### Prompt 约束

```
你是智能监控分析助手。请分析这段监控视频，判断是否有异常行为。
只输出一行纯 JSON，不要 markdown，不要换行，不要任何其他文字。
字段：classification（"正常"或异常类型）、reason（30字以内中文说明）。
示例：{"classification":"正常","reason":"画面中只有一人正常行走，无异常行为。"}
示例：{"classification":"打架","reason":"两人在画面中央发生肢体冲突，动作激烈。"}
必须用中文回答。
```

### `rtsp_monitor.py`

如果你想**不通过 Web**，单独做视频流分析，可以优先看这个文件。

---

## 与 Web 的关系

```
web/services/runtime_manager.py
    ↓ 调用
web/services/vlm_review_runtime.py
    ↓ 调用
vlm/infer/vlm_engine.py
    ↓ 返回结构化 JSON
Web 面板展示
```

当前项目里，网页端大模型复核并不是自己重新写一套推理逻辑，而是直接调用这里的能力。

- `web/` 负责**调度和展示**
- `vlm/infer/` 负责**真正分析 clip**
