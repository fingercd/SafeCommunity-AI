# models — 模型定义目录

> 定义 Vit 当前主线的模型结构：VideoMAE v2 编码器 + MIL 分类头 + 排序损失。

---

## 核心文件

### `vit_video_encoder.py` — 视频编码器

把一段视频 clip 编码成特征向量。

- **输入**：`(B, C, T, H, W)` 或 `list[list[np.ndarray]]`（RGB uint8 帧列表）
- **默认配置**：`image_size=224`，`num_frames=16`
- **输出**：`(B, D)` embedding，`D=768`（Base 模型 hidden_size）
- **编码器**：`OpenGVLab/VideoMAEv2-Base` 预训练权重
- **Pooling 策略**：`auto` / `cls` / `mean` / `pooler`，默认优先取 `pooler_output`

**防御性代码**：
- transformers 5.x 兼容性修复
- meta tensor 自动修复：递归扫描并重建正弦位置编码和 cls_token
- 支持冻结 / 按层解冻（`freeze_backbone`、`unfreeze_last_n_blocks`）

### `mil_head.py` — MIL 分类头

把多个 clip 的特征聚合起来，输出视频级分类结果。

- **输入**：多个 clip 的 embedding `(B, N, D)`
- **输出**：视频级 logits `(B, C)`，当前为二分类（normal / anomaly）

**两种聚合方式**：
1. **Attention Pooling**：`tanh(Wx)` → `w^T` → softmax → 加权求和
2. **Top-K**：按 clip 分类分数选 top-k 再平均 logits

**异常分数分支**（可选）：`D → 512 → 32 → 1 + Sigmoid`，输出每 clip 异常分数 `(B, N)`

### `ranking_loss.py` — MIL 排序损失

帮助模型更好地区分正常和异常片段。

三项损失组合：
1. **排序项**：`max(0, 1 - max(scores_pos) + max(scores_neg))`
2. **稀疏约束**：`λ_sparse * sum(scores_pos)`（默认 `8e-5`）
3. **时间平滑约束**：`λ_smooth * sum((scores[i] - scores[i+1])^2)`（默认 `8e-5`）

> 参考：CVPR 2018 "Real-world Anomaly Detection in Surveillance Videos"

---

## 模型整体理解

```
视频 clip（多帧）
    ↓
[编码器] 把视频"看懂" → 768 维向量
    ↓
[分类头] 把多个片段"总结成结论" → normal / anomaly
```

如果你只是使用模型，不一定先看这里。但如果你要解释"ViT 到底是什么结构"，这个目录一定要看。
