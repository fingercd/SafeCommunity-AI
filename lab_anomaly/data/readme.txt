 数据/标注相关（lab_anomaly/data/）
rtsp_record.py
用来“录视频”。从 RTSP 拉流，把视频按固定时长（比如 30 秒）切成一段段 .mp4 保存到 lab_dataset/raw_videos/...。
你想做真实部署前的数据采集，就用它。

index_build.py
用来“做清单”。扫描 lab_dataset/raw_videos/ 下所有视频，生成/更新 lab_dataset/labels/video_labels.csv。
生成后你需要手工把每条视频的 label 改成 normal 或具体异常类别名（如 intrusion/fall/fire_smoke）。

video_labels.py
只是一套“CSV字段定义+读写工具”。让上面两个脚本读写 video_labels.csv 更稳定。

video_reader.py
只负责“从视频里取指定帧”。比如要第 10、20、30 帧，它负责用 OpenCV 读出来。

clip_dataset.py
负责“把一个视频切成 N 个 clip”。
你给它 clip_len=16、frame_stride=2、num_clips_per_video=8，它就会从每条视频均匀采 8 个片段，每个片段是 16 帧（中间隔 2 帧取一次）。
另外它支持 start_time/end_time：如果你在 CSV 里填了时间段，它只在这段里采样。

transforms.py
一些基础的视频 resize/归一化工具（你现在主要走 HuggingFace 的 processor，所以它不是核心）。