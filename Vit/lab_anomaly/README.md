# lab_anomaly — 视频异常检测核心代码

> Vit 模块最核心的代码目录，负责视频异常检测的**数据准备、模型定义、训练、推理**全流程。

---

## 目录分层

```
lab_anomaly/
├── data/           # 把视频和标签变成模型可读的数据
├── models/         # VideoMAE v2 编码器 + MIL 分类头 + 排序损失
├── tool/           # 训练前准备工具
├── train/          # 训练入口
├── infer/          # 推理入口
└── configs/        # YAML 配置文件
```

---

## 主流程

```
video_labels.csv
    ↓
tool/precompute_clips.py      （离线预切 clip → .npz + manifest.json）
    ↓
train/train_end2end.py        （端到端训练 VideoMAE v2 + MIL）
    ↓
derived/end2end_classifier/   （输出 checkpoint_best.pt 等）
    ↓
infer/known_event_runtime.py  （嵌入主项目实时推理）
或
infer/rtsp_service.py         （独立 RTSP 推理服务）
```

---

## 重点文件

| 路径 | 作用 |
|------|------|
| `tool/precompute_clips.py` | 离线预切 clip，是训练的前置步骤 |
| `train/train_end2end.py` | **当前最核心的训练脚本**，端到端训练 |
| `models/vit_video_encoder.py` | VideoMAE v2 编码器封装，含 meta tensor 自动修复 |
| `models/mil_head.py` | MIL 分类头，支持 Attention Pooling 和 Top-K 聚合 |
| `models/ranking_loss.py` | MIL 排序损失（排序项 + 稀疏约束 + 时间平滑）|
| `infer/known_event_runtime.py` | 异步实时推理运行时，多流支持 |
| `infer/rtsp_service.py` | 独立 RTSP/本地视频推理服务 |
| `infer/scoring.py` | checkpoint 加载、概率与 ranking 分数融合决策 |

---

## 建议阅读顺序

1. `tool/README.md` — 了解预切 clip 怎么做
2. `train/README.md` — 了解怎么训练
3. `infer/README.md` — 了解怎么推理
4. `models/README.md` — 了解模型内部结构
5. `configs/README.md` — 了解配置文件

这样最容易从"怎么跑"理解到"模型内部怎么工作"。
