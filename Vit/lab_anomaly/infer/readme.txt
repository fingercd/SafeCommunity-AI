推理/上线相关（lab_anomaly/infer/）
scoring.py
加载开放集产物（KMeans/OCSVM/阈值）并计算 anomaly score
加载已知分类器 checkpoint
给出“融合输出”（低置信度可当 unknown；再结合异常分数）
rtsp_service.py
真正的“实时服务”。流程是：
RTSP抽帧 → 滑窗组成 clip → encoder 得 embedding → 已知分类概率 + 开放集异常分数 → 去抖动（连着几次才报警、报警冷却）→ 写 events.jsonl / 截图 / HTTP POST