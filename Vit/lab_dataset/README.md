## lab_dataset 目录约定（用于 `lab_anomaly`）

> 项目整体功能与数据流说明见 [docs/代码功能识别报告.md](../docs/代码功能识别报告.md)。

### 推荐结构

### 推荐结构
- `lab_dataset/raw_videos/{camera_id}/{date}/xxx.mp4`
- `lab_dataset/labels/video_labels.csv`
- `lab_dataset/derived/`（可选：缓存 clips/embeddings 等）

### `video_labels.csv` 字段
`video_id, video_path, label, camera_id, start_time, end_time, note`

- **label**：建议用字符串\n+  - `normal`：正常\n+  - 其他：已知异常类别名（例如 `intrusion` / `fall` / `fire_smoke` 等）\n+  - `unknown`：占位，等待你人工补标（索引脚本会生成）\n+
### 生成索引（先建 CSV）
在仓库根目录执行：

```bash
python -m lab_anomaly.data.index_build --dataset_root lab_dataset --videos_root lab_dataset/raw_videos --out_csv lab_dataset/labels/video_labels.csv
```

然后你把 `label` 批量改成 `normal` 或具体异常类别名即可开始训练。

