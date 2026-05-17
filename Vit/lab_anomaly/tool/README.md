# tool — 训练前工具目录

> 存放训练前的数据准备工具。

---

## 当前核心文件

### `precompute_clips.py`

在训练之前，先把原始视频切成模型能直接读取的 clip（`.npz` 格式）。

**为什么要预切？**
- 训练时不用每次都现读整段视频，提升 IO 效率
- 训练更稳定，数据准备和模型训练分开，排错更容易

**输入**：
- `lab_dataset/labels/video_labels.csv`
- `lab_dataset/raw_videos/` 下的原始视频

**输出**：
- `lab_dataset/derived/preclips/` — 切好的 clip（`.npz`）
- `manifest.json` — 索引信息

**常用配置参数**（文件顶部配置区）：
- 数据根目录
- 标签文件路径
- 输出目录
- 每个 clip 多少帧
- 间隔秒数
- 每个视频最多切多少个 clip

## 与后续训练的关系

```
precompute_clips.py  →  train_end2end.py
      ↓                        ↓
  preclips/            直接读取预切 clip
```

如果 `precompute_clips.py` 没跑完，`train_end2end.py` 就没有可直接训练的数据。

## 运行方式

```bash
python Vit/lab_anomaly/tool/precompute_clips.py
```

在文件顶部配置区修改参数后直接运行。
