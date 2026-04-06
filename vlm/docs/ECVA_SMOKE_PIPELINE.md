# ECVA 冒烟流水线（2B + 小数据）

工作目录：`vlm` 项目根（与 `configs/`、`data/` 同级）。

## 路径约定

- ECVA：`data.ecva_root` = `...\moniter\vlm\ECVA`（含 `Video_Annotation.xlsx` 与视频子目录）
- UCF Normal：推荐配置 `data.ucf_normal_root` = `...\raw_videos\normal`（**直接指向 normal**，其下多级子文件夹内 mp4 均会 `rglob`）；未配置时仍可用 `ucf_video_root` 父目录 + 子文件夹名 `normal` / `Normal`。
- 2B 权重：`model.name_or_path_2b` = `...\moniter\vlm\Qwen 3.5 2b`（路径含空格时 YAML 中保持引号）

## 显存与 OOM

- 2B + 4bit QLoRA + 16 帧 + `batch_size=1`：建议 GPU **空闲约 12～16 GB** 更稳；24G 单卡通常足够。
- 训练 OOM：将 `configs/quick_2b.yaml` 里 `training.max_seq_length` 改为 `1536`，并确认 `gradient_checkpointing: true`。
- 评测 OOM：`quick_2b.yaml` 已设 `inference.eval_batch_size: 1`；`evaluate.py` 在未传 `--batch_size` 时会读取该值。

## Smoke 限额（quick_2b）

当 `data.use_smoke_clip_limits: true` 时，`prepare_clips.py --mode ecva --config configs/quick_2b.yaml` 会应用：

- `smoke_ecva_max_records`（默认 40）：只处理前 N 条 ECVA 元数据
- `smoke_normal_ucf_max`（默认 16）：只抽 N 条 UCF Normal clip

命令行若显式传入 `--ecva_max_records` / `--normal_ucf_max`，将覆盖 YAML。

全量跑：`default.yaml` 里 `use_smoke_clip_limits: false`，或使用 `prepare_clips` 传入更大 `--normal_ucf_max`。

## 执行顺序

1. `python data/parse_ecva.py --config configs/quick_2b.yaml`
2. `python data/prepare_clips.py --mode ecva --config configs/quick_2b.yaml --clip_len 16`
3. `python data/build_dataset.py --source ecva --config configs/quick_2b.yaml`
4. `python train/train_qlora.py --config configs/quick_2b.yaml`
5. （可选）`python train/evaluate.py --config configs/quick_2b.yaml --model_path outputs/qlora_2b/final`

一键脚本（Windows）：`scripts/run_ecva_smoke.ps1`
