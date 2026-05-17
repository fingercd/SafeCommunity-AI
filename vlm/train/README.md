# train — 训练目录

> 负责 VLM 模块的 QLoRA 微调、LoRA 合并和模型评估。

---

## 核心文件

| 文件 | 作用 |
|------|------|
| `train_qlora.py` | **QLoRA 微调入口**。4-bit 量化加载 Qwen-VL，插入 LoRA adapter 训练 |
| `merge_lora.py` | **LoRA 合并**。将 adapter 与基础模型合并成完整模型 |
| `evaluate.py` | 常规评估。在验证集上测试微调效果 |
| `stress_test_eval.py` | 压力场景评估。模拟极端条件下的模型表现 |

---

## 当前推荐理解方式

### 第一步：训练

```bash
python vlm/train/train_qlora.py
```

核心入口。它会根据 `configs/default.yaml` 读取：
- 基础模型（如 `vlm/Qwen/`）
- 数据集（`data/` 生成的 JSON）
- LoRA 参数（rank、alpha、target_modules）
- 训练超参（学习率、batch、epoch）

**QLoRA 关键参数**：
- `load_in_4bit=True` — 4-bit 量化加载基础模型，大幅降低显存占用
- `r=16` — LoRA rank
- `lora_alpha=32` — LoRA scaling
- `target_modules` — 应用 LoRA 的模块（通常是 attention 的 q/v/o/proj）

### 第二步：合并

```bash
python vlm/train/merge_lora.py
```

训练结束后，不是所有上层模块都直接读取 adapter。很多场景更方便使用**合并后的完整模型**。

输入：`outputs/qlora/`（adapter 权重）+ `vlm/Qwen/`（基础模型）
输出：`outputs/merged/`（完整模型）

### 第三步：评估

```bash
python vlm/train/evaluate.py
python vlm/train/stress_test_eval.py
```

训练完后可以用这两个脚本验证效果。

---

## 最常改哪里

优先修改 `vlm/configs/default.yaml`，因为训练脚本很多参数都从那里来。

---

## 提醒

如果你是为了让 Web 里的大模型复核真正工作，通常不能只停留在"训练完成"，还要确认：
1. `merge_lora.py` 已成功运行
2. `outputs/merged/` 目录存在且包含有效模型文件
