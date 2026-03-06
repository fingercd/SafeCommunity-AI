# 添加异常数据 & 用 PyCharm 运行说明

## 〇、按文件夹名自动生成 CSV + 多类分类（推荐）

**当前逻辑：`raw_videos` 下的一级文件夹名 = 类别（label），直接生成 CSV，无需手改。**

### 文件夹约定

- **一级目录 = 类别名**（会写入 CSV 的 `label`）  
  例如：`raw_videos/normal/`、`raw_videos/Abuse/`、`raw_videos/Arrest/`、`raw_videos/Stealing/` 等。
- **normal 下可以有子文件夹**（如 `normal/01/`、`normal/02/`），子文件夹名会写入 `camera_id`，不影响 label（仍是 `normal`）。
- 异常类目录下可以直接放视频（如 `Abuse/Abuse001_x264.mp4`），也可以再建子目录。

示例结构：

```
lab_dataset/raw_videos/
├── normal/
│   ├── 01/          ← 视频的 label=normal，camera_id=01
│   ├── 02/
│   └── Testing_Normal_01/
├── Abuse/            ← label=Abuse，camera_id 为空
├── Arrest/
├── Stealing/
└── ...
```

### 流程（按文件夹名生成 CSV 并训练多类）

1. **按上面结构放好视频**（一级文件夹名 = 类别）。
2. **运行扫描**：`lab_anomaly/data/index_build.py`（Working directory = 项目根）。  
   → 会生成/覆盖 `lab_dataset/labels/video_labels.csv`，`label` 来自文件夹名。
3. **提取 embedding**：`lab_anomaly.train.extract_embeddings`（参数见下文 PyCharm 表）。  
   → 会处理所有非 `unknown` 的类别（normal + 各异常类）。
4. **训练多类分类器**：`lab_anomaly.train.train_known_classifier`，建议加 `--exclude_unknown`。  
   → 训练 normal + 所有异常类（Abuse、Arrest、Stealing 等），得到 `lab_dataset/derived/known_classifier/checkpoint_best.pt`。
5. **推理**：在 `rtsp_service` 的配置里指定 `artifacts.known_checkpoint` 指向上述 `checkpoint_best.pt` 即可做多类预测。

**不需要手改 CSV**；重新跑一次 index_build 会按当前文件夹结构刷新 CSV。

---

## 一、异常数据放哪个文件夹？（手动改 CSV 时的参考）

若你仍希望手动维护 CSV，可忽略「〇」中的约定，按下面方式放视频后再在 CSV 里改 `label`。

**结论：异常视频和正常视频都放在同一个根目录下即可，用 CSV 里的 `label` 区分。**

推荐两种放法（任选其一）：

### 方式 A：和正常视频混在同一目录（靠 CSV 区分）

- 把异常视频直接放进 `lab_dataset/raw_videos` 下的任意子目录。
- 例如你已有 `raw_videos/01/` 放正常视频，可以：
  - 在 `raw_videos/01/` 里再放异常视频（如 `raw_videos/01/Abnormal_xxx.mp4`），或
  - 新建 `raw_videos/02/` 专门放异常视频。

**文件夹约定（可选）：**  
`lab_dataset/raw_videos/{摄像头或场景名}/{可选日期}/xxx.mp4`  
例如：`raw_videos/01/20260211/fire_01.mp4`。  
代码不强制子文件夹名字，只要在 `raw_videos` 下面、能被扫描到即可。

### 方式 B：单独建一个“异常”子文件夹（方便管理）

- 例如：`lab_dataset/raw_videos/abnormal/` 或 `lab_dataset/raw_videos/02/`
- 把所有异常视频都放在这个文件夹里，之后在 CSV 里把这些行的 `label` 改成异常类名即可。

**若采用「按文件夹名生成 CSV」**（见上一节），则**一级文件夹名就是 label**，无需再改 CSV。

---

## 二、添加异常数据后，代码/流程还要做什么？

**不需要改任何代码**，按下面顺序做即可。

### 1. 把异常视频放进 `raw_videos` 下

- 放到 `lab_dataset/raw_videos` 里任意子目录（见上一节）。

### 2. 重新跑一遍“扫描视频”（更新 CSV）

- 跑脚本：`lab_anomaly/data/index_build.py`
- 作用：扫描 `raw_videos` 下所有视频，**新发现的视频**会在 CSV 里新增一行，`label` 默认为 `unknown`；已有行会保留你之前改过的 `label`。

### 3. 手动改 CSV：把异常视频的标签改掉

- 打开：`lab_dataset/labels/video_labels.csv`
- 找到刚加进来的异常视频行（新扫出来的通常是 `label=unknown`）。
- 把这几行的 **`label`** 改成你想要的异常类名，例如：
  - `abnormal`（通用异常）
  - `fire`、`intrusion`、`fall` 等（按你业务定）
- 保存 CSV。  
**注意：** 至少保留一批视频的 `label` 为 `normal`，训练时只会用 `normal` 的数据。

### 4. 后续流程照旧

- **步骤 3 提取 embedding**：会处理 CSV 里所有 `label != "unknown"` 的视频（包括 `normal` 和异常）。  
  异常数据也会被提取 embedding，可用于后续评估或可视化。
- **步骤 4 训练 KMeans + One-Class SVM**：**只用 `label=normal` 的 clip**，异常数据不会参与训练，不会破坏开放集设定。
- **步骤 5 推理**：无需改，用训练好的模型即可。

总结：**代码不用改，只需（1）放视频 →（2）跑 index_build →（3）改 CSV 里异常视频的 label。**

---

## 三、用 PyCharm 运行（不用命令行）

所有脚本的**工作目录必须为项目根目录**（即包含 `lab_anomaly` 和 `lab_dataset` 的那一层，例如 `C:\Users\Administrator\Desktop\Vit`），否则 `lab_dataset`、`lab_anomaly` 等相对路径会找不到。

### 1. 设置项目根为工作目录（必做一次）

1. 在 PyCharm 里用 **File → Open** 打开项目根目录（例如 `Vit` 文件夹）。
2. 打开 **Run → Edit Configurations**。
3. 如果没有配置过，先点 **+** → **Python**，新建一个配置。
4. 在配置里：
   - **Script path** 或 **Module name** 见下面各脚本说明。
   - **Working directory** 设为项目根，例如：  
     `C:\Users\Administrator\Desktop\Vit`  
     （或点右侧文件夹图标选到 `Vit` 目录。）
5. 点 **OK** 保存。

之后每新建一个“运行配置”，都**把 Working directory 设成同一项目根**。

### 2. 各步骤在 PyCharm 里怎么跑

| 步骤 | 脚本 | PyCharm 配置 |
|------|------|----------------|
| 扫描视频（按文件夹名生成 CSV） | `index_build.py` | **Script path** 选：`lab_anomaly/data/index_build.py`；**Working directory**：项目根（如 `Vit`）。 |
| 提取 embedding | `extract_embeddings.py` | **Module name** 填：`lab_anomaly.train.extract_embeddings`；**Parameters** 填：`--config lab_anomaly/configs/embedding_example.yaml --save_format npy_per_clip`；**Working directory**：项目根。 |
| **训练多类分类器**（normal + 各异常类） | `train_known_classifier.py` | **Module name** 填：`lab_anomaly.train.train_known_classifier`；**Parameters** 填：`--exclude_unknown`（可选：`--epochs 20 --batch_size 16`）；**Working directory**：项目根。 |
| 训练开放集模型（仅 normal） | `fit_kmeans_ocsvm.py` | **Module name** 填：`lab_anomaly.train.fit_kmeans_ocsvm`；**Parameters** 按需填（如 `--k 20 --quantile 0.95`）；**Working directory**：项目根。 |
| 推理（RTSP/本地视频） | `rtsp_service.py` | **Module name** 填：`lab_anomaly.infer.rtsp_service`；**Parameters** 填：`--config lab_anomaly/configs/rtsp_service_example.yaml`；**Working directory**：项目根。 |

### 3. 用“模块”方式运行（推荐）

对 `extract_embeddings`、`fit_kmeans_ocsvm`、`rtsp_service` 这些以 `python -m 包.模块` 方式跑的脚本，在 PyCharm 里：

- **Run → Edit Configurations → + → Python**。
- 选 **Module name**（不要选 Script path），在 **Module name** 里填：
  - `lab_anomaly.train.extract_embeddings`
  - `lab_anomaly.train.fit_kmeans_ocsvm`
  - `lab_anomaly.infer.rtsp_service`
- 在 **Parameters** 里填命令行参数（例如 `--config lab_anomaly/configs/embedding_example.yaml --save_format npy_per_clip`）。
- **Working directory** 一定要是项目根（`Vit`）。

这样效果和你在项目根目录下用命令行执行 `python -m lab_anomaly.train.extract_embeddings ...` 一致。

### 4. 扫描视频（index_build）在 PyCharm 里

- **Script path** 选：`lab_anomaly/data/index_build.py`。
- **Working directory**：项目根。
- 若视频不在默认的 `lab_dataset/raw_videos`，在 **Parameters** 里加：  
  `--videos_root 你的视频目录路径`。

---

## 小结

| 问题 | 答案 |
|------|------|
| 异常数据放哪？ | **推荐**：按类别建一级文件夹（`raw_videos/normal/`、`raw_videos/Abuse/` 等），`normal` 下可再建子文件夹（如 `01/`、`02/`）。 |
| CSV 怎么来？ | 运行 `index_build.py` 会**按一级文件夹名自动生成** CSV 的 `label`，无需手改。 |
| 训练多类（normal + 各异常）？ | 跑完 index_build → extract_embeddings 后，运行 **train_known_classifier**（加 `--exclude_unknown`），得到多类分类模型。 |
| 用 PyCharm 要注意什么？ | 每个运行配置的 **Working directory** 都设为项目根（包含 `lab_anomaly`、`lab_dataset` 的目录）；用 **Module name** 运行带 `-m` 的脚本时，在 **Parameters** 里填命令行参数。 |
