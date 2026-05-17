# VLM — 视觉语言模型复核模块

> 负责"当系统怀疑异常时，用大模型帮你解释到底像什么异常"。基于 Qwen-VL 实现监控视频异常片段的语义分析与结构化输出。

---

## 这个模块能做什么

| 能力 | 说明 |
|------|------|
| **语义复核** | 接收视频片段，输出"打架""火灾""盗窃"等结构化异常分类 |
| **中文解释** | 用 30 字以内中文说明异常原因 |
| **QLoRA 微调** | 在自有数据上微调 Qwen-VL，提升安防场景识别准确率 |
| **LoRA 合并** | 将微调后的 adapter 与基础模型合并，便于部署 |
| **独立推理** | 可作为独立服务运行，也可嵌入主项目 Web 层 |

---

## 在整个项目中的位置

```
yolo/（看见目标）→ Vit/（判断异常）→ vlm/（解释异常）→ web/（展示结论）
```

VLM 是**二次复核层**：不是基础检测的必需模块，而是增强模块。

---

## 技术架构

### 模型

- **基础模型**：Qwen3.5-VL（9B 或 2B，支持 image-text-to-text）
- **微调方式**：QLoRA 4-bit 量化微调
- **输入**：一组视频帧（RGB）+ Prompt
- **输出**：结构化 JSON

```json
{
  "classification": "打架",
  "reason": "两人在画面中央发生肢体冲突，动作激烈。"
}
```

### Prompt 设计

```
你是智能监控分析助手。请分析这段监控视频，判断是否有异常行为。
只输出一行纯 JSON，不要 markdown，不要换行，不要任何其他文字。
字段：classification（"正常"或异常类型）、reason（30字以内中文说明）。
必须用中文回答。
```

---

## 目录结构

```
vlm/
├── README.md              # 本文件（模块总览）
├── requirements.txt       # 依赖列表
│
├── configs/               # 总配置文件
│   └── default.yaml       # 模型路径、数据路径、LoRA 参数、训练/推理超参
│
├── data/                  # 数据准备
│   └── ...                # 解析元数据、切 clip、生成训练集 JSON
│
├── train/                 # 训练、合并、评估
│   ├── train_qlora.py     # QLoRA 微调入口
│   ├── merge_lora.py      # LoRA 与基础模型合并
│   ├── evaluate.py        # 常规评估
│   └── stress_test_eval.py# 压力场景评估
│
├── infer/                 # 推理
│   ├── vlm_engine.py      # 核心推理引擎（Web 层直接调用）
│   └── rtsp_monitor.py    # 独立视频流分析
│
├── pycharm/               # PyCharm 分步脚本（适合小白）
│
├── outputs/               # 训练产物
│   ├── qlora/             # QLoRA adapter 输出
│   └── merged/            # 合并后的完整模型（Web 层优先读取）
│
├── Qwen/                  # 本地基础模型（Qwen3.5-9B）
├── Qwen 3.5 2b/           # 本地基础模型（Qwen3.5-2B）
└── ECVA/                  # 数据集或实验数据
```

---

## 完整工作流程

### 阶段 1：数据准备

```bash
# 按 pycharm/ 中的步骤脚本，或直接用 data/ 下的脚本
# 输入：原始视频 + 元数据
# 输出：train.json / val.json / test.json（含视频帧路径和标注）
```

### 阶段 2：QLoRA 微调

```bash
python vlm/train/train_qlora.py
```

- 读取 `configs/default.yaml` 中的基础模型路径、数据路径、LoRA 参数
- 4-bit 量化加载 Qwen-VL，插入 LoRA adapter
- 训练完成后输出到 `outputs/qlora/`

### 阶段 3：合并模型

```bash
python vlm/train/merge_lora.py
```

- 将 LoRA adapter 与基础模型合并成完整模型
- 输出到 `outputs/merged/`
- **Web 层优先读取 `outputs/merged/`，而不是单独的 adapter**

### 阶段 4：评估

```bash
python vlm/train/evaluate.py
python vlm/train/stress_test_eval.py
```

### 阶段 5：推理（接入主项目）

Web 层通过以下文件调用 VLM：
- `web/services/runtime_manager.py` — 总调度
- `web/services/vlm_review_runtime.py` — VLM 复核线程管理

触发条件：
1. ViT 判定异常且概率超过 `vit_threshold`
2. 或周期性自动触发（`vlm_auto_interval_sec`，独立于 ViT）

---

## 与 Web 层的集成

```
Web 后台
    ↓
ViT 判断异常（概率 ≥ 阈值）
    ↓
VlmReviewRuntime.submit(stream_id, clip_frames)
    ↓
VLMEngine.analyze_clip(frames)  ← 在 vlm/infer/vlm_engine.py 中实现
    ↓
返回 JSON：{classification, reason, ...}
    ↓
Web 面板展示为可折叠卡片
```

---

## 你最可能要改的地方

| 目标 | 修改位置 |
|------|----------|
| 改基础模型路径 | `configs/default.yaml` |
| 改数据路径 | `configs/default.yaml` |
| 改训练参数 | `configs/default.yaml`、`train/train_qlora.py` |
| 改输出目录 | `configs/default.yaml` |
| 改推理阈值 | Web 端流级配置（`vit_threshold`、`agent_enabled`）|
| 改 Prompt 内容 | `infer/vlm_engine.py` 中的 `PROMPT` 常量 |
| 改复核逻辑 | `infer/vlm_engine.py`、`web/services/vlm_review_runtime.py` |

---

## 注意事项

1. **路径问题**：`configs/default.yaml` 中很多是本机绝对路径，换机器后优先修改这里
2. **Web 优先读取合并模型**：网页复核层期望 `outputs/merged/` 存在完整的合并模型，而不是单独的 LoRA adapter
3. **显存压力**：Qwen-VL 是大模型，显存占用远高于 YOLO。常见影响因素：
   - clip 长度（帧数）
   - 图像分辨率
   - 是否 4-bit 量化
   - batch 大小
4. **predict.py 不加载 VLM**：根目录 `predict.py` 只跑 YOLO + ViT，只有 Web 版才会额外接上 VLM
5. **模型加载回退**：优先加载 `outputs/merged/`（微调模型），不存在时回退到 `Qwen/`（基础模型）
