# infer — 推理目录

> 负责把训练好的 ViT 模型真正用起来，提供实时推理能力。

---

## 核心文件

| 文件 | 作用 | 适用场景 |
|------|------|----------|
| `known_event_runtime.py` | 异步实时推理运行时，多流支持 | **接入主项目**（被 `web/services/runtime_manager.py` 调用）|
| `rtsp_service.py` | 独立 RTSP/本地视频推理服务 | **独立部署**，不依赖 moniter 主工程 |
| `scoring.py` | checkpoint 加载、MIL 分类、概率与 ranking 分数融合 | 被上述两个文件调用 |

---

## `known_event_runtime.py` 工作原理

1. **帧缓冲**：每路流维护一个固定长度的帧缓冲区
2. **滑窗触发**：每收到 `window_stride` 个新帧，从缓冲区均匀采样组成 clip
3. **异步推理**：推理任务通过 `queue.Queue` 提交到独立后台线程，不阻塞 YOLO 主循环
4. **结果回写**：推理完成后更新 `state.vit_event`，供上层读取

核心特性：
- 支持多路流（`stream_id` 隔离），每路独立缓冲与步进
- 接收 BGR 帧，内部转 RGB 后入缓冲
- 推理：clip 帧序列 → VideoMAE 编码 → MIL 分类 → 更新 `vit_event`

---

## `rtsp_service.py` 工作原理

1. **抽帧**：按 `sample_fps` 降采样（如 5 fps）
2. **缓冲区**：维护最近 `clip_len` 帧
3. **推理**：均匀取帧 → VideoMAE 编码 → MIL 分类 + ranking 分数
4. **融合决策**：`predict_fusion()` 结合 softmax 概率与 ranking 异常分数
5. **去抖**：`min_consecutive`（连续触发次数）+ `cooldown_sec`（冷却时间）
6. **输出**：`events.jsonl` 日志、可选截图、可选 HTTP POST JSON

---

## `scoring.py` 核心逻辑

```
checkpoint_best.pt
    ├── encoder_cfg    # VideoMAE v2 配置
    ├── encoder_state  # 编码器权重
    ├── mil_cfg        # MIL 头配置
    ├── mil_state      # MIL 头权重
    └── label2idx / idx2label  # 标签映射
```

加载后：
1. 重建编码器和 MIL 头
2. `model.float()` 强制 FP32，避免 BFloat16 与 LayerNorm 冲突
3. 推理时：embedding → logits → softmax → 分类结果
4. 若启用 `anomaly_branch`，输出每 clip 异常分数，取 max 得 `ranking_score`
5. 融合：概率最高类 + ranking 分数综合判断是否异常

---

## 注意事项

- 训练和推理的 `frames_per_clip` / `clip_len` 必须对齐
- checkpoint 中的编码器配置要与推理时使用的 `encoder_model_name` 一致
- 如果上层接入时报参数不匹配，先核对 `known_event_runtime.py` 的真实构造函数
