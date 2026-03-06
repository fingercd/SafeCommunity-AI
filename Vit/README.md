

# Vit 视频异常检测训练流程说明

本文档说明在 PyCharm 下（无需命令行）完成「双流 ViT（RGB + 光流）→ 先提 embedding → 已知类分类 + normal 聚类画边界」的完整操作流程。  
不使用 SSIM 与光流过滤，所有步骤均可通过 Run Configuration 一键运行。

---

## 一、流程概览

顺序  步骤                     脚本                                输入依赖                输出目录
1     预计算光流               precompute_optical_flow.py          视频 + video_labels.csv  lab_dataset/derived/optical_flows
2     提取 embedding           extract_embeddings.py               步骤 1 的 flows_dir      lab_dataset/derived/embeddings
3     已知类分类               train_known_classifier.py           步骤 2 的 embeddings    lab_dataset/derived/known_classifier
4     normal 聚类与边界        fit_kmeans_ocsvm.py                 步骤 2 的 embeddings    lab_dataset/derived/open_set
4b    评估边界                 eval_open_set.py                    步骤 4 的 open_set 产物  控制台报告
4c    按簇导出样本             export_clusters_to_folders.py       步骤 2 + 4             lab_dataset/derived/cluster_samples

重要：步骤 2 依赖步骤 1 的光流目录；步骤 3、4 依赖步骤 2 的 embedding 目录。请按顺序执行。

---

## 二、环境与依赖

- Python：建议 3.9+
- 工作目录：所有 Run Configuration 的 Working directory 请设为项目根目录，例如：
  C:\Users\Administrator\Desktop\Vit
- 依赖安装（在项目根或虚拟环境下）：
  pip install -r lab_anomaly/requirements.txt
  主要包含：torch、torchvision、opencv-python、numpy、transformers、PyYAML、scikit-learn、joblib、scikit-image

---

## 三、数据与目录约定

- 视频标签：lab_dataset/labels/video_labels.csv，需包含列：video_id、video_path、label（如 normal、Abuse、Shoplifting 等）
- 视频文件：video_path 为相对 lab_dataset 的路径（如 raw_videos/Abuse/Abuse001_x264.mp4），即实际路径为 lab_dataset/raw_videos/Abuse/Abuse001_x264.mp4
- 相对路径：脚本内 lab_dataset、lab_anomaly 等均相对于项目根目录解析；在 PyCharm 中设置好 Working directory 后即可直接使用相对路径

---

## 四、步骤 1：预计算光流

双流 ViT 需要预先算好的光流，与 extract_embeddings 使用相同的 clip 采样（clip_len、frame_stride、num_clips_per_video），保证帧索引一致。

### 4.1 脚本位置
lab_anomaly/train/precompute_optical_flow.py

### 4.2 PyCharm Run Configuration
Script path: 选择 lab_anomaly/train/precompute_optical_flow.py
Working directory: C:\Users\Administrator\Desktop\Vit（或你的项目根）
Parameters: 
--dataset_root lab_dataset --labels_csv lab_dataset/labels/video_labels.csv --out_dir lab_dataset/derived/optical_flows --clip_len 16 --frame_stride 2 --num_clips_per_video 32 --backend farneback --skip_existing

- --backend farneback：使用 OpenCV 在 CPU 上计算光流
- --skip_existing：若某视频已存在 flows.npz 则跳过

### 4.3 常用参数说明
参数                    默认值         说明
--dataset_root          lab_dataset    数据集根目录
--labels_csv            lab_dataset/labels/video_labels.csv  视频标签 CSV
--out_dir               lab_dataset/derived/optical_flows  光流输出目录
--clip_len              16             每个 clip 的帧数
--frame_stride          2              帧步长
--num_clips_per_video   32             每视频 clip 数
--backend               raft           farneback（CPU）或 raft（GPU）
--device                auto           光流计算设备：cpu/cuda/auto
--resize                384,384        光流计算分辨率 W,H
--skip_existing         无             已存在 flows.npz 的视频跳过

### 4.4 输出
- 每个视频一个子目录：lab_dataset/derived/optical_flows//flows.npz
- flows_meta.jsonl（记录视频与路径对应关系）
- 步骤 2 的 flows_dir 指向该 optical_flows 目录

---

## 五、步骤 2：提取 Embedding（双流 ViT，无过滤）

使用 RGB + 光流 双流 ViT 提取 clip 级 embedding，不启用 SSIM 与光流过滤。

### 5.1 脚本位置
lab_anomaly/train/extract_embeddings.py

### 5.2 方式 A：使用 YAML 配置（推荐）
配置文件：lab_anomaly/configs/embedding_example.yaml

PyCharm Run Configuration:
Script path: lab_anomaly/train/extract_embeddings.py
Working directory: 项目根目录
Parameters: --config lab_anomaly/configs/embedding_example.yaml

YAML 中已设置：
enable_filtering: false
enable_flow_filter: false
encoder.use_dual_stream: true
flows.flows_dir: lab_dataset/derived/optical_flows

### 5.3 方式 B：不用 YAML，纯命令行参数
Parameters:
--config "" --dataset_root lab_dataset --labels_csv lab_dataset/labels/video_labels.csv --out_dir lab_dataset/derived/embeddings --use_dual_stream --no_filtering --flows_dir lab_dataset/derived/optical_flows --clip_len 16 --frame_stride 2 --num_clips_per_video 32 --batch_size 32

- --no_filtering：关闭 SSIM 与光流过滤
- --use_dual_stream：启用双流（embedding 维度 1536）

### 5.4 常用参数说明
参数                    默认值         说明
--config                lab_anomaly/configs/embedding_example.yaml  YAML 配置文件
--out_dir               lab_dataset/derived/embeddings  embedding 输出目录
--use_dual_stream       由 YAML/默认      启用 RGB+光流双流
--no_filtering          无             关闭 SSIM 与光流过滤
--flows_dir             lab_dataset/derived/optical_flows  预计算光流目录
--clip_len              16             与步骤 1 一致
--frame_stride          2              与步骤 1 一致
--num_clips_per_video   32             与步骤 1 一致
--save_format           npy_per_clip   npy_per_clip 或 npz_per_video
--batch_size            32             批大小
--limit                 0              仅处理前 N 个 clip

### 5.5 输出
- 目录：lab_dataset/derived/embeddings
- embeddings_meta.jsonl：每行记录 video_id、label、embedding_path
- npy_per_clip：每个视频子目录含 clip_000.npy 等（长度 1536）
- npz_per_video：每个视频一个 xxx.npz（含 embeddings 数组）

---

## 六、步骤 3：已知类 MIL 分类训练

在步骤 2 的 clip embeddings 上训练「已知类」MIL 分类器（normal/Abuse/Shoplifting 等）。

### 6.1 脚本位置
lab_anomaly/train/train_known_classifier.py

### 6.2 PyCharm Run Configuration
Script path: lab_anomaly/train/train_known_classifier.py
Working directory: 项目根目录
Parameters: 
--embeddings_dir lab_dataset/derived/embeddings --out_dir lab_dataset/derived/known_classifier --pooling attn --epochs 200 --batch_size 16

### 6.3 常用参数说明
参数                    默认值         说明
--embeddings_dir        见脚本默认    步骤 2 的 embedding 目录
--out_dir               见脚本默认    分类器输出目录
--pooling               attn          聚合方式：attn 或 topk
--topk                  2             pooling=topk 时的 k
--epochs                200           训练轮数
--batch_size            16            批大小
--lr                    3e-4          学习率
--val_ratio             0.2           验证集比例
--expected_num_clips    0             每视频 clip 数
--resume                空            从 checkpoint 恢复
--use_anomaly_branch    无            启用异常分数分支
--normal_label          normal        正常类标签名

### 6.4 输出
- checkpoint_best.pt / checkpoint_last.pt：模型与优化器状态
- labels.json：label2idx、idx2label

---

## 七、步骤 4：Normal 聚类与边界（KMeans + OCSVM）

仅使用 label=normal 的 clip embedding，先做 KMeans 聚类，再对每个簇训练 One-Class SVM。

### 7.1 脚本位置
lab_anomaly/train/fit_kmeans_ocsvm.py

### 7.2 PyCharm Run Configuration
Script path: lab_anomaly/train/fit_kmeans_ocsvm.py
Working directory: 项目根目录
Parameters: 
--embeddings_dir lab_dataset/derived/embeddings --out_dir lab_dataset/derived/open_set --normal_label normal --k 16 --quantile 0.95 --min_cluster_size 20

### 7.3 常用参数说明
参数                    默认值         说明
--embeddings_dir        lab_dataset/derived/embeddings  步骤 2 的 embedding 目录
--out_dir               lab_dataset/derived/open_set  KMeans/OCSVM 输出目录
--normal_label          normal         视为「正常」的标签名
--k                     16             KMeans 簇数
--min_cluster_size      20             样本数少于此值的簇回退到全局
--nu                    0.05           OCSVM 的 nu 参数
--gamma                 scale          OCSVM 核参数：scale/auto/数值
--quantile              0.95           normal 上取 anomaly_score 的分位数
--limit                 0              仅用前 N 条 normal clip
--seed                  42             随机种子

### 7.4 输出与「画边界」含义
- kmeans.joblib：KMeans 模型
- ocsvm_global.joblib：全局 OCSVM
- ocsvm_cluster_*.joblib：每个簇的 OCSVM
- thresholds.json：global_threshold、cluster_thresholds

「画边界」指：决策边界由 OCSVM 的 decision_function 定义，anomaly_score = -decision_function(x)，大于阈值判为异常。

---

## 八、可选：评估边界与导出簇样本

### 8.1 评估开放集边界（eval_open_set.py）
Script: lab_anomaly/train/eval_open_set.py
Parameters: 
--open_set_dir lab_dataset/derived/open_set --embeddings_dir lab_dataset/derived/embeddings
输出：控制台报告（簇质量、边界有效性）

### 8.2 按簇导出样本（export_clusters_to_folders.py）
Script: lab_anomaly/train/export_clusters_to_folders.py
Parameters: 
--embeddings_dir lab_dataset/derived/embeddings --open_set_dir lab_dataset/derived/open_set --dataset_root lab_dataset --out_dir lab_dataset/derived/cluster_samples --max_per_cluster 50
输出：cluster_samples/cluster_* 文件夹（每簇导出的 clip 样本）

---

## 九、配置文件说明（YAML）

lab_anomaly/configs/embedding_example.yaml 关键配置：

配置项                            说明                              推荐（双流、无过滤）
enable_filtering                  是否启用 SSIM 过滤                false
enable_flow_filter                是否启用光流幅值过滤              false
encoder.use_dual_stream           是否使用 RGB+光流双流             true
encoder.fusion_method             双流融合方式                      concat
flows.flows_dir                   预计算光流目录                    lab_dataset/derived/optical_flows
sampling.clip_len                 与步骤 1 一致                     16
sampling.frame_stride             与步骤 1 一致                     2
sampling.num_clips_per_video      与步骤 1 一致                     32
runtime.batch_size                批大小                            32（按显存调整）
runtime.limit                     限制处理 clip 数，0 为不限制       0

---

## 十、输出文件一览

步骤  输出目录/文件                              说明
1     lab_dataset/derived/optical_flows//flows.npz  每视频每 clip 的光流数组
2     lab_dataset/derived/embeddings/embeddings_meta.jsonl    embedding 元信息索引
2     lab_dataset/derived/embeddings//clip_*.npy   clip 级 embedding 文件
3     lab_dataset/derived/known_classifier/checkpoint_best.pt 最佳分类模型
3     lab_dataset/derived/known_classifier/labels.json        类别映射
4     lab_dataset/derived/open_set/kmeans.joblib             KMeans 模型
4     lab_dataset/derived/open_set/ocsvm_global.joblib       全局 OCSVM
4     lab_dataset/derived/open_set/ocsvm_cluster_*.joblib     簇 OCSVM
4     lab_dataset/derived/open_set/thresholds.json           异常阈值
4c    lab_dataset/derived/cluster_samples/cluster_*/         每簇导出的 clip 样本

---

## 快速检查清单（PyCharm）

[ ] Working directory 设为项目根（如 C:\Users\Administrator\Desktop\Vit）
[ ] 步骤 1 已跑完，lab_dataset/derived/optical_flows 下有各视频的 flows.npz
[ ] 步骤 2 使用 embedding_example.yaml 或 --use_dual_stream --no_filtering --flows_dir ...，且未开启 SSIM/光流过滤
[ ] 步骤 2 输出目录与步骤 3、4 的 --embeddings_dir 一致
[ ] 步骤 4 的 --normal_label 与 CSV 中正常类标签一致（默认 normal）

按上述顺序在 PyCharm 中配置并运行各脚本即可完成全流程。