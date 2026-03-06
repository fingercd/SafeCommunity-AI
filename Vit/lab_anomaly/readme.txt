你应该怎么训练（最推荐的顺序）
下面是“最稳、最少走弯路”的训练流程。

第一步：准备并标注 video_labels.csv
把视频放进 lab_dataset/raw_videos/...
生成 CSV：
python -m lab_anomaly.data.index_build --dataset_root lab_dataset --videos_root lab_dataset/raw_videos --out_csv lab_dataset/labels/video_labels.csv
手工把 label 改好：至少要有一批 normal，以及你关心的已知异常类别。

第二步：先提 embedding（强烈建议先做）
python -m lab_anomaly.train.extract_embeddings --config lab_anomaly/configs/embedding_example.yaml --save_format npy_per_clip
产物在：lab_dataset/derived/embeddings/

第三步：训练“已知异常分类器”（你要输出类别就必须做）
python -m lab_anomaly.train.train_known_classifier --embeddings_dir lab_dataset/derived/embeddings --out_dir lab_dataset/derived/known_classifier --exclude_unknown --pooling attn --epochs 10
拿到：lab_dataset/derived/known_classifier/checkpoint_best.pt

第四步：训练“未知异常检测”（你要发现没见过的异常就做）
python -m lab_anomaly.train.fit_kmeans_ocsvm --embeddings_dir lab_dataset/derived/embeddings --out_dir lab_dataset/derived/open_set --normal_label normal --k 16 --quantile 0.95
拿到：lab_dataset/derived/open_set/thresholds.json 等文件。

第五步（可选）：伪标签迭代，让已知分类更抗背景
python -m lab_anomaly.train.pseudo_label_iter --embeddings_dir lab_dataset/derived/embeddings --ckpt lab_dataset/derived/known_classifier/checkpoint_best.pt --out_dir lab_dataset/derived/pseudo_iter --exclude_unknown --iters 3 --epochs_per_iter 2

训练完怎么用（实时 RTSP）
修改 lab_anomaly/configs/rtsp_service_example.yaml 里的：
rtsp_url
artifacts.known_checkpoint（指向你的 best 或 pseudo_iter 的 checkpoint）
artifacts.open_set_dir（指向 lab_dataset/derived/open_set）

启动：
python -m lab_anomaly.infer.rtsp_service --config lab_anomaly/configs/rtsp_service_example.yaml
输出：
事件：lab_dataset/derived/realtime/events.jsonl
截图：lab_dataset/derived/realtime/snapshots/...
你最容易踩坑的 3 点（提前提醒）
没 normal 类就没法做 open-set：fit_kmeans_ocsvm.py 只吃 label=normal
label=unknown 的视频默认不会提 embedding/训练：你需要把要用的数据改成 normal 或具体类别
第一次跑 encoder 要下载模型：离线环境要提前把 HF 模型缓存好，否则会卡在下载