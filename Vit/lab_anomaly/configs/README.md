## configs/ — YAML 配置文件

本目录包含 ViT 异常检测模块各阶段的示例配置文件。

### 文件说明

| 文件 | 用途 | 对应脚本 |
|------|------|----------|
| `embedding_example.yaml` | 提取 embedding 的基础配置（单流 RGB，关闭 SSIM 过滤） | `train/extract_embeddings.py` |
| `embedding_dual_stream.yaml` | 双流（RGB + 光流）embedding 配置 | `train/extract_embeddings.py` |
| `pseudo_label_iter_example.yaml` | 伪标签迭代训练配置 | `train/pseudo_label_iter.py` |
| `rtsp_service_example.yaml` | 实时 RTSP 推理服务配置 | `infer/rtsp_service.py` |

### 配置结构（以 embedding_example.yaml 为例）

```yaml
# 数据
dataset_root: lab_dataset
labels_csv: lab_dataset/labels/video_labels.csv

# 采样
sampling:
  clip_len: 16
  frame_stride: 2
  num_clips_per_video: 32

# 编码器
encoder:
  use_dual_stream: false    # true 时需要预计算光流
  fusion_method: concat     # 双流融合方式

# 过滤（建议关闭）
enable_filtering: false
enable_flow_filter: false

# 光流（双流时需要）
flows:
  flows_dir: lab_dataset/derived/optical_flows

# 运行时
runtime:
  batch_size: 32
  limit: 0                  # 0 = 不限制
```

### rtsp_service_example.yaml 关键字段

```yaml
rtsp_url: rtsp://user:pass@ip:554/stream
artifacts:
  known_checkpoint: lab_dataset/derived/known_classifier/checkpoint_best.pt
  open_set_dir: lab_dataset/derived/open_set
  labels_json: lab_dataset/derived/known_classifier/labels.json
output:
  events_jsonl: lab_dataset/derived/realtime/events.jsonl
  snapshots_dir: lab_dataset/derived/realtime/snapshots
```

### 使用方式

```bash
# 方式 A：通过 --config 参数
python -m lab_anomaly.train.extract_embeddings --config lab_anomaly/configs/embedding_example.yaml

# 方式 B：命令行参数覆盖 YAML
python -m lab_anomaly.train.extract_embeddings --config lab_anomaly/configs/embedding_example.yaml --batch_size 64
```
