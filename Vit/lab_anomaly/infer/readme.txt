推理/上线相关（lab_anomaly/infer/）

scoring.py
加载开放集产物（KMeans/OCSVM/阈值）并计算 anomaly score
加载已知分类器 checkpoint
给出"融合输出"（低置信度可当 unknown；再结合异常分数）

rtsp_service.py
真正的"实时服务"。流程是：
RTSP抽帧 → 滑窗组成 clip → encoder 得 embedding → 已知分类概率 + 开放集异常分数 → 去抖动（连着几次才报警、报警冷却）→ 写 events.jsonl / 截图 / HTTP POST

known_event_runtime.py
异步 ViT 动态事件分类运行时，由 moniter/predict.py 主管线调用。
核心特性：
- 维护每路流的帧缓冲，达到 clip 长度后触发推理
- 推理任务通过内部 queue.Queue 提交到独立后台线程（_vit_worker_thread），不阻塞主 YOLO 推理循环
- 接收 BGR 帧，内部转 RGB 后入缓冲，避免主线程重复做 cvtColor
- 推理流程：clip 帧序列 → VideoMAE ViT 编码 → MIL Head 分类 → 更新 state 中的 vit_event 字段
- 支持多路流（stream_id 隔离），每路独立缓冲与步进