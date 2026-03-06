"""
lab_anomaly
===========

面向“实验室/园区监控”的异常行为识别工程代码。

设计目标：
- 已知异常：视频级标签监督训练（MIL 聚合）。
- 未知异常：KMeans + One-Class SVM 产生 anomaly score。
- 支持离线视频与 RTSP 实时推理。
"""

from __future__ import annotations

