# lab_anomaly 学习流程任务（从看懂到能改）

## 先建立三个核心概念

- **Clip**：一段短视频片段（例如 16 帧，每隔 2 帧取一帧），是模型的最小输入单位；一条视频会均匀采多个 clip。
- **Embedding**：每个 clip 经 ViT 编码后的固定维向量（如 768 维），后续分类和开放集检测都基于它，不直接用像素。
- **开放集**：训练时只用「正常」样本，推理时能发现「没见过」的异常；本仓库用 KMeans + One-Class SVM 在 normal 的 embedding 上建模实现。

---

## 阶段一：数据流——从 CSV 到 Clip 到磁盘 Embedding

**目标**：能说清楚「一条视频如何变成若干 clip，再变成若干向量存盘」。

### 任务 1.1：标注与索引

- **看**：[lab_anomaly/data/video_labels.py](../lab_anomaly/data/video_labels.py) 的 `CSV_HEADER`、`VideoLabelRow`、`read_video_labels_csv`。
- **懂**：每行 = 一条视频；`label` 只能是 `normal`、具体异常类名、或 `unknown`；`start_time`/`end_time` 可选，用于只取片段。
- **看**：[lab_anomaly/data/index_build.py](../lab_anomaly/data/index_build.py) 的 `scan_videos`、`build_or_update_csv`。
- **懂**：如何从 `raw_videos/` 扫出视频并生成/更新 `video_labels.csv`；新视频默认 `unknown`，需人工改 label。

### 任务 1.2：视频 → Clip 采样

- **看**：[lab_anomaly/data/clip_dataset.py](../lab_anomaly/data/clip_dataset.py) 的 `ClipSample`、`VideoClipDataset.__init__` 与 `__getitem__`（以及用到的 `uniform_clip_indices` 等）。
- **懂**：`clip_len`、`frame_stride`、`num_clips_per_video` 的含义；一条视频如何被切成 N 个 clip，每个 clip 是「若干帧的列表」；若有 `start_time`/`end_time` 只在该区间内采样。
- **可选**：看 [lab_anomaly/data/video_reader.py](../lab_anomaly/data/video_reader.py) 里如何按帧索引读帧（`read_frames_by_indices_cv2`）。

### 任务 1.3：Clip → Embedding 并落盘

- **看**：[lab_anomaly/train/extract_embeddings.py](../lab_anomaly/train/extract_embeddings.py) 的 `main`：读 YAML、构建 `VideoClipDataset`、加载 encoder、循环 batch 提 embedding、写 `out_dir` 和 `embeddings_meta.jsonl`。
- **看**：[lab_anomaly/configs/embedding_example.yaml](../lab_anomaly/configs/embedding_example.yaml)。
- **懂**：`save_format`（`npy_per_clip` vs `npz_per_video`）的区别；`embeddings_meta.jsonl` 里每行对应一个 clip 或一个视频，含 `embedding_path`、`label`、`video_id` 等，后面训练都读这个 meta。

**阶段一检验**：能画出「CSV → 按行读视频 → 每视频采 N 个 clip → 每个 clip 进 ViT → 得到 embedding 写盘 + meta」的流程图（可手画或 mermaid）。

---

## 阶段二：模型——ViT 编码器 + MIL 分类头

**目标**：能说清每个 clip 如何变成向量、多个 clip 如何变成一条视频的预测；知道换模型/改结构要动哪里。

### 任务 2.1：Clip → 单向量（Encoder）

- **看**：[lab_anomaly/models/vit_video_encoder.py](../lab_anomaly/models/vit_video_encoder.py) 的 `HfVideoEncoderConfig`、`HfVideoEncoder.__init__` 与 `forward`。
- **懂**：输入是 list of list of RGB 帧，或已归一化的 `(B,T,C,H,W)`；`processor` 做预处理；`model` 是 HuggingFace 的 VideoMAE；输出 `(B, D)` 的 embedding；`pooling`（auto/pooler/cls/mean）如何选最后一层表示。
- **扩展点**：换 backbone 只需改 `model_name`（或扩展 `HfVideoEncoder` 支持其他 HF 模型）；改输入分辨率/帧数看 `processor` 与 config。

### 任务 2.2：多 Clip → 视频级预测（MIL 头）

- **看**：[lab_anomaly/models/mil_head.py](../lab_anomaly/models/mil_head.py) 的 `MILHeadConfig`、`AttentionPooling`、`MILClassifier.forward`。
- **懂**：输入是 `(B, N, D)`（N 个 clip 的 embedding）；`attn` 是学习权重把 N 个向量加权合成一个再分类；`topk` 是取分数最高的 k 个 clip 再平均；输出是视频级 logits `(B, C)`。
- **扩展点**：改 `pooling`（attn/topk）、`topk`、`attn_hidden`、`dropout` 等都在 `MILHeadConfig`；加新 pooling 方式需在 `MILClassifier` 里加分支。

**阶段二检验**：能口头解释「一个视频 8 个 clip → 8 个 768 维向量 → MIL 聚成 1 个向量 → 线性层 → 得到 C 类 logits」。

---

## 阶段三：训练——三种产物（Embedding / 已知分类器 / 开放集）

**目标**：知道三个训练脚本的输入输出、关键参数，以及改参数/改逻辑要动哪里。

### 任务 3.1：提取 Embedding（已学）

- 回顾 [lab_anomaly/train/extract_embeddings.py](../lab_anomaly/train/extract_embeddings.py) 和 [lab_anomaly/configs/embedding_example.yaml](../lab_anomaly/configs/embedding_example.yaml)。
- **可改**：`clip_len`、`frame_stride`、`num_clips_per_video`、`model_name`、`batch_size`、`save_format`。

### 任务 3.2：已知异常分类器

- **看**：[lab_anomaly/train/train_known_classifier.py](../lab_anomaly/train/train_known_classifier.py) 的入口参数、如何读 `embeddings_meta.jsonl` 和 `embedding_path`、如何组 batch（同一视频的多个 clip 一起）、如何建 `MILClassifier` 和训练循环。
- **懂**：输入是已缓存的 embedding 路径（不是视频）；`--exclude_unknown` 会过滤掉 label=unknown；输出是 `checkpoint_best.pt`（含 `model_state`、`label2idx`、`idx2label`、`cfg`）。
- **可改**：`--pooling`、`--epochs`、学习率、优化器；改类别集合只需改 CSV 的 label 并重新训练。

### 任务 3.3：开放集（KMeans + OCSVM）

- **看**：[lab_anomaly/train/fit_kmeans_ocsvm.py](../lab_anomaly/train/fit_kmeans_ocsvm.py) 的完整流程：只取 `label==normal` 的 embedding → KMeans 聚类 → 每簇（或全局）训练 One-Class SVM → 在 normal 上算 `anomaly_score = -decision_function` → 按分位数得到每簇/全局阈值 → 写出 `kmeans.joblib`、`ocsvm_*.joblib`、`thresholds.json`。
- **懂**：KMeans 把「正常」分成多种模式，每模式一个 OCSVM 更紧的边界；小簇回退到 `ocsvm_global`；推理时先 KMeans 判簇，再用对应 OCSVM 和阈值判是否异常。
- **可改**：`--k`、`--min_cluster_size`、`--nu`、`--gamma`、`--quantile`；改这些会直接影响敏感度与误报。

**阶段三检验**：能说出「已知分类器用哪些 label」「开放集只用谁」「checkpoint 里存了什么」「thresholds.json 里有哪些 key」。

---

## 阶段四：推理——打分融合与 RTSP 实时服务

**目标**：能说清一次推理的链路；改配置或加输出（如新 API、新告警条件）知道改哪几处。

### 任务 4.1：开放集打分 + 已知分类 + 融合

- **看**：[lab_anomaly/infer/scoring.py](../lab_anomaly/infer/scoring.py) 的 `OpenSetScorer.score_embedding`（KMeans 判簇 → 选 OCSVM → decision_function → anomaly_score 与 threshold）、`load_known_classifier`、`fuse_known_and_open_set`。
- **懂**：`anomaly_score > threshold` 即开放集判异常；已知分类置信度低于 `min_known_prob` 可视为 `unknown`；融合结果同时包含 `predicted_label`、`predicted_prob`、`anomaly_score`、`is_anomaly`。

### 任务 4.2：RTSP 实时流水线

- **看**：[lab_anomaly/infer/rtsp_service.py](../lab_anomaly/infer/rtsp_service.py) 的 `ServiceConfig`、主循环：按 `sample_fps` 抽帧、滑窗组成 clip、调用 encoder、已知分类 + 开放集打分、融合、去抖（`min_consecutive`、`cooldown_sec`）、写 `events.jsonl`、截图、可选 HTTP POST。
- **看**：[lab_anomaly/configs/rtsp_service_example.yaml](../lab_anomaly/configs/rtsp_service_example.yaml)。
- **懂**：配置里 `artifacts`、`encoder`、`sampling`、`fusion`、`outputs`、`runtime` 各自管什么；去抖逻辑如何减少误触。
- **可改**：换 RTSP、改 checkpoint/open_set 路径、改抽帧与滑窗参数、改融合阈值、改 API URL、改冷却与连续帧数；加新告警条件在「融合后、写事件前」加分支。

**阶段四检验**：能画出「RTSP 帧 → 滑窗 clip → encoder → 已知 + 开放集 → 融合 → 去抖 → 输出」的流程图；能指出改 YAML 里哪几项会改变敏感度或延迟。

---

## 扩展时「该动谁」速查

| 想做的事 | 主要涉及文件 / 配置 |
|---------|---------------------|
| 改 clip 长度、帧步长、每视频 clip 数 | [embedding_example.yaml](../lab_anomaly/configs/embedding_example.yaml)、[rtsp_service_example.yaml](../lab_anomaly/configs/rtsp_service_example.yaml)、各脚本的 `--clip_len` 等 |
| 换视频编码模型（如另一 HF 模型） | [vit_video_encoder.py](../lab_anomaly/models/vit_video_encoder.py) 的 `model_name`；embedding 和 RTSP 的 config 里 `encoder.model_name` |
| 改已知分类的 pooling / 结构 | [mil_head.py](../lab_anomaly/models/mil_head.py)、[train_known_classifier.py](../lab_anomaly/train/train_known_classifier.py) 的 `MILHeadConfig` |
| 调开放集敏感度 / 误报 | [fit_kmeans_ocsvm.py](../lab_anomaly/train/fit_kmeans_ocsvm.py) 的 `--k`、`--quantile`、`--nu`；推理端无额外参数，阈值已写在 `thresholds.json` |
| 改融合策略（何时算 unknown/异常） | [scoring.py](../lab_anomaly/infer/scoring.py) 的 `fuse_known_and_open_set`；[rtsp_service_example.yaml](../lab_anomaly/configs/rtsp_service_example.yaml) 的 `fusion.*` |
| 加新的输出（如另一 API、数据库） | [rtsp_service.py](../lab_anomaly/infer/rtsp_service.py) 里在写 `events.jsonl` / 截图 / POST 的附近加逻辑 |

---

## 建议学习顺序与时间

1. **阶段一**（约 1～2 小时）：按 1.1 → 1.2 → 1.3 读代码，必要时在 CSV 和 `embeddings_meta.jsonl` 上打 print 或断点观察。
2. **阶段二**（约 1 小时）：看 ViT 与 MIL 的输入输出维度和 config；可单独写几行代码加载 encoder 和 MIL，喂假数据看 shape。
3. **阶段三**（约 1～2 小时）：先跑通 `extract_embeddings` → `train_known_classifier` → `fit_kmeans_ocsvm`（用少量数据），再带着「输入输出是什么」回头扫脚本逻辑。
4. **阶段四**（约 1 小时）：看 `scoring` 与 `rtsp_service` 的调用关系；若有 RTSP 可跑一次服务并对照 YAML 改几项观察行为。

每完成一个阶段，用该阶段的「检验」自测；全部做完后，用「扩展速查」试改一两处参数或配置，确认能对应到代码位置，你就达到「能改」的目标了。
