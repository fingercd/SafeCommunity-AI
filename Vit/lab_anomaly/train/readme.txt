【历史参考】旧训练脚本说明

下面的脚本是早期版本的训练路线（基于 embedding 提取 + KMeans/SVM），
当前仓库的主线已演进为 VideoMAE v2 + MIL 端到端训练（train_end2end.py）。

---
旧脚本说明（仅供参考）：

extract_embeddings.py
训练前的"必经步骤"（旧流程）。读取 video_labels.csv，对每条视频采样 clips，然后用 vit_video_encoder.py 提取 embedding 并缓存到磁盘。
这是为了后面训练更快、更省显存，也方便反复试 KMeans/SVM。

train_known_classifier.py
训练"已知异常分类器"（带类别）。输入不是视频帧，而是你缓存好的 embedding。
它会把同一个视频的多个 clip embedding 组成 (N,D)，用 MILClassifier 输出视频级类别。

fit_kmeans_ocsvm.py
训练"未知异常检测"。只用 label=normal 的 embedding：
- KMeans 把正常数据分成 K 个簇（正常的不同"模式"）
- 每个簇训练 One-Class SVM（或回退到全局一个 SVM）
- 在正常数据上统计阈值（比如 95% 分位），以后超过阈值就算异常

pseudo_label_iter.py
伪标签迭代增强（可选）。
用 KMeans 的"覆盖度" × 分类器对该类的分数，给 clip 打伪标签，然后继续训练分类器，让模型更少被背景干扰。
