# data — 数据预处理目录

> 负责把原始视频和标签整理成模型能直接读取的数据格式。

---

## 核心文件

| 文件 | 作用 |
|------|------|
| `rtsp_record.py` | 从 RTSP 拉流录制视频，按固定时长切成 `.mp4` 保存到 `raw_videos/` |
| `index_build.py` | 扫描 `raw_videos/` 生成/更新 `video_labels.csv` 清单 |
| `video_labels.py` | CSV 字段定义与读写工具类 |
| `video_reader.py` | 从视频中读取指定帧（支持按索引或时间戳采样）|
| `clip_dataset.py` | 将视频切成 clip 数据集，支持均匀采样、时间范围过滤 |
| `transforms.py` | 基础视频预处理（resize / normalize），训练时优先用 HuggingFace processor |
| `preclip_manifest.py` | manifest.json 读写、npz 存取、参数一致性校验 |

## 使用流程

1. **录制视频**（可选）：`rtsp_record.py` 从摄像头录制
2. **生成标签清单**：`index_build.py` 扫描视频目录生成 CSV
3. **人工标注**：编辑 CSV 中的 `label` 字段（`normal` 或异常类名）
4. **预切 clip**：被 `tool/precompute_clips.py` 调用，将视频切成 `.npz`
5. **构建数据集**：被 `train/train_end2end.py` 调用，读取 clip 组织成训练批次
