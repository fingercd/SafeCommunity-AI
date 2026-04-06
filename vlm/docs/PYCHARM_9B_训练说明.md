# 9B 大模型训练：PyCharm 一键顺序运行说明

## 环境

- **工作目录**（Working directory）：`vlm` 项目根（与 `configs/`、`pycharm/`、`data/` 同级）。
- **解释器**：与本机 GPU 训练一致的环境（例如 `conda env: yolovv`）。
- **主配置**：[`configs/default.yaml`](../configs/default.yaml)。下文参数均指该文件。

## 训练全流程（按顺序 Run）

| 顺序 | PyCharm 运行脚本 | 作用 |
|------|------------------|------|
| 1 | [`pycharm/step01_parse_ecva.py`](../pycharm/step01_parse_ecva.py) | 读 ECVA 的 `Video_Annotation.xlsx`，生成 `data/processed/ecva_metadata.jsonl` |
| 2 | [`pycharm/step02_prepare_clips_ecva.py`](../pycharm/step02_prepare_clips_ecva.py) | 按时间段导出帧目录 + 从 `ucf_normal_root` 抽 Normal，生成 `ecva_clips/` 与 `ecva_clips_metadata.jsonl` |
| 3 | [`pycharm/step03_build_dataset_ecva.py`](../pycharm/step03_build_dataset_ecva.py) | 按 **video_id** 划分 train/val/test，写出三份 JSON |
| 4 | [`pycharm/step04_train_qlora_9b.py`](../pycharm/step04_train_qlora_9b.py) | **9B + QLoRA**，读入上一步 JSON，checkpoint → `outputs/qlora/final` |
| 5 | [`pycharm/step05_evaluate.py`](../pycharm/step05_evaluate.py) | 加载 `outputs/qlora/final`，在 test 上出 `outputs/eval_report.json` |

## 核心参数说明（`configs/default.yaml`）

### `model`

- **`name_or_path`**：本地 9B VLM 目录（内含 `config.json` 等）。
- **`lora_small`**：必须为 **`false`** 才会走 9B；为 `true` 时训练脚本会改用 `name_or_path_2b`。
- **`use_4bit` + BitsAndBytes**：4bit 量化加载基座，显著省显存；**与「2B/9B」无简单倍数关系**，视频 token 仍占大头。

### `data`

- **`ecva_root`**：放 `Video_Annotation.xlsx` 与 ECVA 视频子目录的根路径。
- **`use_smoke_clip_limits`**：**`false`** = 全量 ECVA 元数据 + Normal 条数由 prepare 默认上限（800）；**`true`** 时用下面两个 smoke 数字做冒烟。
- **`ucf_video_root`**：UCF 父目录（保留兼容）。
- **`ucf_normal_root`**：直接指向 **`raw_videos/normal`**（递归子文件夹找 mp4）。
- **`train_ratio` / `val_ratio` / `test_ratio`**：按 **video_id** 分片，同一视频的多 clip 不会跨集。

### `clips`

- **`clip_len`**：每个 clip 多少帧；训练与评测读取帧数与此一致，**越大显存越高**。
- **`min_short_side`**：抽帧后 JPG 短边下限；**越大分辨率越高、显存越高**。

### `lora`

- **`r` / `lora_alpha`**：LoRA 秩与缩放；9B 常用 `r=64`、`alpha=128`。

### `training`

- **`output_dir`**：checkpoint 根目录，最终会保存到其下的 `final`（与 `step05` 默认路径一致）。
- **`num_train_epochs`**：训练轮数。
- **`per_device_train_batch_size`**：单卡 micro-batch，一般为 **1**（视频 VLM）。
- **`gradient_accumulation_steps`**：梯度累积，**增大不改变显存峰值**，只改变「等效 batch」与更新频率。
- **`learning_rate`**：学习率，9B QLoRA 常用约 **1e-4** 量级（需结合 loss 微调）。
- **`max_seq_length`**：最大序列长度；**过长易 OOM**，可先 **1536** 再试。

### `inference`

- **`eval_batch_size`**：`evaluate.py` 在未手动改 batch 时读取；9B 建议 **1～2**。

## 常见问题

- **decord 失败**：编辑 [`pycharm/step02_prepare_clips_ecva.py`](../pycharm/step02_prepare_clips_ecva.py)，将 `USE_OPENCV_ONLY = True`。
- **显存不足**：在 `default.yaml` 中降低 `clips.min_short_side`、`clips.clip_len`，或降低 `training.max_seq_length`；改 clip 相关后需 **重做 step02**。
- **`step05` 找不到适配器**：确认 `step04` 已完成且 `outputs/qlora/final` 内存在 `adapter_config.json` 等。
