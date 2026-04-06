# VLM 安防视频异常检测（Qwen3.5-9B）

从数据生成 → QLoRA 微调 → 实时推理的完整流水线，基座模型为 **Qwen3.5-9B-Instruct**。
## 环境

```bash
pip install -r requirements.txt
```

建议 Python 3.10+，GPU 显存 ≥24GB（如 RTX 4090）。

## 流程概览

1. **切分 clips**：UCF-Crime 视频 → 32 帧/clip、50% 重叠，每 clip 采样 8 帧保存。
2. **自动标注**：用 Qwen3.5-9B（INT4）对每个 clip 生成 JSON 标注（is_anomaly, reasoning 等）。
3. **构建训练集**：标注 → train/val/test JSON（70/15/15）。
4. **QLoRA 微调**：4bit 基座 + LoRA，训练 5 epoch。
5. **合并与评估**：合并 LoRA 到基座，在 test 上算 AUC / 准确率 / FPR@95%TPR。
6. **实时监控**：RTSP 或本地视频 → 滑动窗口组 clip → VLM 分析 → 时序平滑 + 置信度阈值 → 告警。

## 命令示例

```bash
# 1. 切分（需先准备 UCF-Crime 到 data/raw/ucf_crime，或指定 --video_root）
python -m vlm.data.prepare_clips --video_root data/raw/ucf_crime --clips_dir data/processed/clips --metadata_path data/processed/clips_metadata.jsonl

# 2. 自动标注（可选 --max_clips 做小规模测试）
python -m vlm.data.generate_annotations --config configs/default.yaml

# 3. 构建训练集
python -m vlm.data.build_dataset --config configs/default.yaml

# 4. QLoRA 微调
python -m vlm.train.train_qlora --config configs/default.yaml

# 5. 合并 LoRA
python -m vlm.train.merge_lora --adapter_path outputs/qlora/final --output_path outputs/merged

# 6. 评估
python -m vlm.train.evaluate --model_path outputs/qlora/final --test_json data/processed/test.json

# 7. 实时监控（RTSP 或本地视频）
python -m vlm.infer.rtsp_monitor --source "rtsp://..." --model_path outputs/qlora/final --log_path outputs/rtsp_alerts.jsonl
```

配置见 `configs/default.yaml`（模型路径、LoRA 参数、训练超参、推理阈值等）。
