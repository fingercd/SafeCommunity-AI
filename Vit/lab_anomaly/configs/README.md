# configs — 配置文件目录

> 存放 Vit 当前主线用到的 YAML 配置文件。

---

## 当前有效的配置文件

| 文件 | 作用 |
|------|------|
| `train_end2end.yaml` | **当前主线训练配置**，控制数据路径、模型参数、训练策略 |
| `rtsp_service_example.yaml` | 推理服务示例配置，控制 RTSP 地址、checkpoint 路径、输出目录 |

## 配置覆盖机制

```
train_end2end.py 内部默认值
    ↓ 被覆盖
train_end2end.yaml 中的配置
    ↓ 最终生效
```

- Python 文件里先有一套默认配置
- YAML 再去覆盖它
- **如果 YAML 里没写某个值，就回退到 Python 默认值**

> ⚠️ 不能只改 YAML 不看 Python 默认值。改了配置没效果时，先检查 YAML 是否真的覆盖了目标字段。

## 关键字段

### `train_end2end.yaml`
- `dataset_root` / `labels_csv` / `preclip_root`：数据路径
- `out_dir`：训练输出目录
- `frames_per_clip`：每片段帧数（与预切阶段对齐）
- `encoder_model_name`：编码器名称（如 `OpenGVLab/VideoMAEv2-Base`）
- `batch_size` / `stages` / `val_ratio`：训练参数

### `rtsp_service_example.yaml`
- `rtsp_url`：视频源地址
- `artifacts.known_checkpoint`：checkpoint 路径
- `output.events_dir` / `output.snapshots_dir`：结果输出目录

## 注意事项

1. 旧文档里提到的 `embedding_example.yaml`、`embedding_dual_stream.yaml` 等配置文件在当前仓库中可能不存在，以实际目录中的文件为准
2. 训练和推理配置要对齐，尤其是 `clip_len` 和 `encoder_model_name`
