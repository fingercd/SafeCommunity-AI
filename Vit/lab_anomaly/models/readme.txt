 模型相关（lab_anomaly/models/）
vit_video_encoder.py

“视频 ViT 特征提取器”。把一段 clip（多帧图像）变成一个向量 embedding。
默认用 HuggingFace 的 MCG-NJU/videomae-base。第一次会下载权重。

mil_head.py
“把多个 clip 的 embedding 汇总成一个视频预测”的分类头（MIL）。两种汇总方式：
pooling=attn：注意力加权，把更重要的 clip 权重变大
pooling=topk：挑最像目标的 top-k 个 clip 取平均（更简单粗暴）