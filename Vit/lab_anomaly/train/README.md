# train — 训练目录

> 负责 Vit 当前主线的模型训练。

---

## 当前核心文件

### `train_end2end.py`

**当前最新、最完整的训练入口。**

它做的事情：
1. 读取已经预切好的 clip（`.npz`）
2. 用 `VideoMAE v2` 编码这些 clip
3. 用 `MIL` 把多个 clip 的结果汇总
4. 输出正常/异常的分类模型

---

## 训练前置条件

在运行之前，先确认：
- [ ] `lab_dataset/labels/video_labels.csv` 已准备好
- [ ] `tool/precompute_clips.py` 已跑完
- [ ] `lab_dataset/derived/preclips/` 里已有切好的 clip

---

## 关键参数

建议优先看 `train_end2end.py` 顶部配置区和 `configs/train_end2end.yaml`：

| 参数 | 说明 |
|------|------|
| `preclip_root` | 预切 clip 目录 |
| `out_dir` | 训练输出目录 |
| `frames_per_clip` | 每片段帧数（与预切阶段对齐）|
| `encoder_model_name` | 编码器名称（如 `OpenGVLab/VideoMAEv2-Base`）|
| `batch_size` | 训练批次大小 |
| `stages` | 多阶段训练配置（解冻策略、学习率、epoch 数）|
| `val_ratio` | 验证集比例 |

---

## 三阶段渐进解冻训练

```
阶段 1：head_only
  - 30 epochs，lr=1e-3
  - 冻结全部 backbone，只训练 MIL 头

阶段 2：unfreeze_2
  - 20 epochs，lr=5e-5
  - 解冻最后 2 个 Transformer block

阶段 3：unfreeze_4
  - 15 epochs，lr=2e-5
  - 解冻最后 4 个 Transformer block
```

**损失函数**：`CE Loss + lambda_rank * MIL Ranking Loss`

**数据采样**：`BalancedVideoBatchSampler` — 保证每个 batch 至少包含 1 个 normal + 1 个 anomaly 视频

**混合精度**：支持 `torch.amp.autocast` + `GradScaler`

---

## 输出产物

训练完成后，结果输出到 `lab_dataset/derived/end2end_classifier/`：

| 文件 | 说明 |
|------|------|
| `checkpoint_best.pt` | 验证集最优 checkpoint |
| `checkpoint_last.pt` | 最后一个 epoch |
| `labels.json` | 标签映射 |
| `history.json` | 训练历史（loss、acc、auc）|
| `plot_loss_curves.png` | 损失曲线图 |
| `plot_acc.png` | 准确率曲线图 |
| `eval_report/` | 评估报告（混淆矩阵、ROC 曲线等）|

---

## 运行方式

```bash
python Vit/lab_anomaly/train/train_end2end.py
```

在文件顶部配置区修改参数后直接运行。

---

## 注意事项

- 如果在其它文档里看到 embedding、光流、开放集那类说法，请优先以当前真实存在的 `train_end2end.py` 为准
- `clip_len`、`encoder_model_name` 等参数在训练、预切、推理三阶段必须保持一致
