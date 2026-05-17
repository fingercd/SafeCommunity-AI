【历史参考】旧流程训练指南

下面的流程是早期版本的训练路线（embedding + KMeans + OCSVM），
当前仓库的主线已演进为 VideoMAE v2 + MIL 端到端训练。
如需了解最新流程，请查看 lab_anomaly/train/train_end2end.py 和对应的 README.md。

---
旧流程步骤（仅供参考）：

第一步：准备并标注 video_labels.csv
把视频放进 lab_dataset/raw_videos/...
生成 CSV：
python -m lab_anomaly.data.index_build --dataset_root lab_dataset --videos_root lab_dataset/raw_videos --out_csv lab_dataset/labels/video_labels.csv
手工把 label 改好：至少要有一批 normal，以及你关心的已知异常类别。

第二步：先提 embedding（强烈建议先做）
python -m lab_anomaly.train.extract_embeddings --config lab_anomaly/configs/embedding_example.yaml --save_format npy_per_clip
产物在：lab_dataset/derived/embeddings/

第三步：训练"已知异常分类器"
python -m lab_anomaly.train.train_known_classifier --embeddings_dir lab_dataset/derived/embeddings --out_dir lab_dataset/derived/known_classifier --exclude_unknown --pooling attn --epochs 10
拿到：lab_dataset/derived/known_classifier/checkpoint_best.pt

第四步：训练"未知异常检测"
python -m lab_anomaly.train.fit_kmeans_ocsvm --embeddings_dir lab_dataset/derived/embeddings --out_dir lab_dataset/derived/open_set --normal_label normal --k 16 --quantile 0.95
拿到：lab_dataset/derived/open_set/thresholds.json

第五步（可选）：伪标签迭代
python -m lab_anomaly.train.pseudo_label_iter --embeddings_dir lab_dataset/derived/embeddings --ckpt lab_dataset/derived/known_classifier/checkpoint_best.pt --out_dir lab_dataset/derived/pseudo_iter --exclude_unknown --iters 3 --epochs_per_iter 2

训练完怎么用（实时 RTSP）
修改 lab_anomaly/configs/rtsp_service_example.yaml 里的 rtsp_url 和 artifacts 路径，然后：
python -m lab_anomaly.infer.rtsp_service --config lab_anomaly/configs/rtsp_service_example.yaml
