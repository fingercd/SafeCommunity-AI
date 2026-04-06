"""
Qwen3.5-9B QLoRA 微调：安防视频异常检测。

功能概述：
- 从 config 或命令行读入数据路径与 LoRA/训练参数
- 加载 4bit 基座 + LoRA，使用 Hugging Face Trainer 训练
- 数据集：train.json，每条含 messages（含 video 指向 frames_dir）与 frames_dir
"""

# 用于保证 Python 2/3 类型提示的兼容性，现代代码可忽略，但保留以增强鲁棒性
from __future__ import annotations

import argparse  # 用于解析命令行参数
import json      # 用于读取 JSON 格式的数据集
import os
import sys
import traceback
from pathlib import Path  # 用于更优雅地处理文件路径（替代 os.path）
from typing import Any    # 类型提示通用类型

import torch
import yaml
from PIL import Image

# ===== [修改] Trainer 的回调必须继承此类，否则会缺方法导致 Trainer 启动崩溃 =====
try:
    from transformers import TrainerCallback as _TrainerCallback
except ImportError:
    _TrainerCallback = object

# ==============================================================================
# 延迟导入模块 (Lazy Import)
# 目的：
# 1. 加快脚本在无 GPU 环境下的启动速度（例如仅做语法检查时）
# 2. 避免在安装依赖前直接运行报错，便于友好提示
# ==============================================================================
def _import_train_deps():
    """
    内部函数：动态导入训练所需的重量级依赖库。
    主要是 transformers, peft, datasets 等。
    """
    # transformers 核心组件
    from transformers import (
        AutoProcessor,             # 自动加载模型对应的处理器（处理文本+图像/视频）
        Qwen3_5ForConditionalGeneration, # Qwen3.5 模型类
        TrainingArguments,         # 训练参数容器
        Trainer,                   # Hugging Face 训练器
        BitsAndBytesConfig,        # 4bit/8bit 量化配置
    )
    # PEFT (Parameter-Efficient Fine-Tuning) 库，用于 LoRA
    from peft import (
        LoraConfig,                # LoRA 配置
        get_peft_model,            # 将 LoRA 适配器挂载到基座模型
        prepare_model_for_kbit_training, # 为 4bit 量化训练准备模型（冻结层、Cast LayerNorm等）
    )
    from datasets import Dataset   # Hugging Face 数据集格式（可选，这里主要用了自定义 Map-style Dataset）
    
    return (
        AutoProcessor, Qwen3_5ForConditionalGeneration, TrainingArguments, Trainer,
        BitsAndBytesConfig, LoraConfig, get_peft_model, prepare_model_for_kbit_training,
        Dataset,
    )


def _transformers_version_at_least(major: int, minor: int) -> bool:
    """用于是否使用 TrainingArguments.report_to='swanlab'（transformers>=4.50）。"""
    try:
        import transformers

        parts = transformers.__version__.split(".")
        return (int(parts[0]), int(parts[1])) >= (major, minor)
    except Exception:
        return False


class EvalReadyCheckpointCallback(_TrainerCallback):
    """在 Trainer 自动存档后补齐 processor，使 checkpoint 可直接用于评估。"""

    def __init__(self, processor: Any):
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        #args 是 TrainingArguments 对象，state 是 TrainerState 对象，control 是 TrainerControl 对象
        #kwargs 是 kwargs 对象，包含模型、处理器等
        model = kwargs.get("model")
        if model is None:
            return control
            #control 用于控制训练流程，如果返回 control，则训练流程会停止
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_dir))
        self.processor.save_pretrained(str(ckpt_dir))
        print(f"[train] 已补全可评估 checkpoint: {ckpt_dir}", flush=True)
        return control


class VisionUnfreezeCallback(_TrainerCallback):
    """
    ===== [修改] 从第 (unfreeze_epoch+1) 个训练 epoch 开始解冻部分视觉编码器参数 =====
    unfreeze_epoch 为 0 基下标：1 表示第 2 个 epoch 开始时解冻。

    策略：只打开 float32 / fp16 / bf16 的参数（4bit 量化权重不可直接反传）：
    - visual.merger.*（视觉到文本的投影）
    - visual.blocks.24/25/26 中带 norm 的层（最后 3 个 ViT block 的归一化）
    """

    def __init__(self, model: Any, unfreeze_epoch: int = 1) -> None:
        self.model = model
        self.unfreeze_epoch = unfreeze_epoch
        self._done = False

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self._done or state.epoch < self.unfreeze_epoch:
            return control
        #如果已经解冻过了，就别再解冻第二次
        #如果现在训练轮数还没到指定时间，也先别动
        #解冻标志设为 True
        #解冻列表设为空
        #遍历模型参数
        self._done = True
        thawed: list[str] = []
        #遍历模型参数
        for name, p in self.model.named_parameters():
            if "lora" in name.lower():
                continue
                #如果参数名包含 lora，则跳过
            if p.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                continue
                #如果参数类型不是 float32、float16、bfloat16，则跳过
            if "visual.merger" in name:
                p.requires_grad = True
                thawed.append(name)
                continue
                #如果参数名包含 visual.merger，则解冻
            if any(f"blocks.{i}." in name for i in (24, 25, 26)) and "norm" in name.lower():
                p.requires_grad = True
                thawed.append(name)
                continue
                #如果参数名包含 blocks.24、blocks.25、blocks.26，则解冻
        print(
            f"[train] ===== [修改] 第 2 个 epoch 起：解冻部分视觉层，共 {len(thawed)} 个参数张量 =====",
            flush=True,
        )
        for n in thawed[:30]:
            print(f"  [修改] trainable visual: {n}", flush=True)
        if len(thawed) > 30:
            print(f"  ... 另有 {len(thawed) - 30} 项已解冻（省略打印）", flush=True)

        if hasattr(self.model, "print_trainable_parameters"):
            self.model.print_trainable_parameters()
        return control


# ==============================================================================
# 工具函数：视频帧加载
# ==============================================================================
def load_frames_from_dir(frames_dir: str | Path, num_frames: int = 16) -> list[Image.Image]:
    """
    从目录加载 frame_*.jpg 或按序的 jpg，返回 PIL 列表。
    逻辑细节：
    1. 优先寻找 "frame_0001.jpg" 这种格式
    2. 如果没有，则加载目录下所有 jpg
    3. 只取前 num_frames 张
    4. 如果帧数不足，重复最后一帧进行填充 (Padding)
    
    参数:
        frames_dir: 存放帧图片的目录路径
        num_frames: 需要返回的固定帧数
    返回:
        包含 PIL.Image 对象的列表，长度固定为 num_frames
    """
    # 将字符串路径转换为 Path 对象，方便操作
    frames_dir = Path(frames_dir)
    # 1. 查找文件：优先匹配 frame_*.jpg (如 frame_001.jpg)
    imgs = sorted(frames_dir.glob("frame_*.jpg")) 
    # 2. 如果没找到，退而求其次，匹配所有 *.jpg
    if not imgs:
        imgs = sorted(frames_dir.glob("*.jpg"))
    # 3. 截断：只保留前 num_frames 张
    imgs = imgs[:num_frames]
    out = []
    # 4. 读取图片
    for p in imgs:
        # 打开图片并强制转换为 RGB (排除灰度图或 RGBA 带来的通道问题)
        img = Image.open(p).convert("RGB")
        out.append(img)
    # 5. 填充：如果图片数量少于要求的 num_frames，且至少有一张图
    # 策略：重复最后一张图，直到填满
    while len(out) < num_frames and out:
        out.append(out[-1])
    # 再次截断确保长度严格一致（防止 while 逻辑漏洞）
    return out[:num_frames]


def _resolve_frames_dir(frames_dir: str | Path, project_root: Path) -> Path:
    """
    内部工具：智能解析 frames_dir 路径。
    为了适配不同的数据集存放结构，增加代码健壮性。
    
    查找逻辑优先级：
    1. 如果输入是绝对路径且存在，直接返回
    2. 在项目标准目录下查找 (data/processed/ecva_clips 或 clips)
    3. 如果相对路径直接存在，返回
    4. 最后默认拼接一个路径返回（即使不存在，交给上层报错）
    """
    p = Path(frames_dir)
    # 情况1：绝对路径且存在
    if p.is_absolute() and p.exists():
        return p
    # 提取文件夹名字 (例如输入是 "video_001", 提取 "video_001")
    name = p.name
    # 情况2：在项目的标准数据目录中查找
    # 尝试不同的子文件夹名称变体
    for sub in ("ecva_clips", "clips"):
        cand = project_root / "data" / "processed" / sub / name
        if cand.exists():
            return cand
    # 情况3：也许是相对于当前脚本的相对路径
    if p.exists():
        return p
    # 情况4：实在找不到，返回一个默认拼接的路径（让后续加载逻辑去报错）
    return project_root / "data" / "processed" / "clips" / name
# ==============================================================================
# 核心数据集类 (PyTorch Dataset)
# ==============================================================================
class VideoAnomalyDataset(torch.utils.data.Dataset):
    """
    自定义 PyTorch Dataset。
    每条样本处理流程：
    1. 从 JSON 读取 messages (对话历史) + frames_dir (视频路径)
    2. 加载视频帧 (Images)
    3. 用 Processor 将 文本+视频 编码为模型输入 (input_ids, pixel_values...)
    4. 构建 Labels (仅计算 Assistant 回复部分的 Loss)
    """
    def __init__(
        self,
        data: list[dict],          # 从 JSON 加载的原始数据列表
        processor: Any,             # Hugging Face Processor
        max_seq_length: int,        # 最大序列长度 (截断)
        project_root: Path,         # 项目根目录，用于找文件
        num_frames: int = 16,       # 每个视频取多少帧
    ):
        self.data = data
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.project_root = project_root
        self.num_frames = num_frames
        
        # 数据清洗：预检查所有样本，只保留同时有 frames_dir 和 messages 的样本
        self.valid_indices = []
        for i, item in enumerate(data):
            if item.get("frames_dir") and item.get("messages"):
                self.valid_indices.append(i)
                #有效的索引列表会被放到self.valid_indices列表中
        # 如果没有有效数据，直接报错终止程序
        if not self.valid_indices:
            raise ValueError("没有有效训练样本（需含 frames_dir 与 messages）")

    def __len__(self):
        """返回数据集大小（有效样本数）"""
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        核心逻辑：获取单条数据并处理为模型输入格式。
        这是 DataLoader 迭代时会调用的函数。
        """
        # 1. 获取原始数据
        # 注意：这里使用 valid_indices 来跳过脏数据
        real_idx = self.valid_indices[idx]
        #idx是数据集索引，real_idx是有效数据集索引
        item = self.data[real_idx]
        
        messages = item["messages"]       # 对话列表
        frames_dir = item["frames_dir"]   # 视频帧目录
        
        # 2. 解析路径并加载图片
        abs_frames = _resolve_frames_dir(frames_dir, self.project_root)
        frames = load_frames_from_dir(abs_frames, self.num_frames)
        
        # 3. 数据格式适配 (Message Injection)
        # Qwen/VL 模型通常需要在 messages 里通过特殊 dict 指明图片/视频位置
        # 这里我们遍历 messages，把 "video" 类型的占位符替换为实际加载的 frames list
        new_messages = []
        for m in messages:
            # m是一条一条json数据，写（‘content’），就返回当前content这个键的值赋给content 加上（‘m’）就是返回m这个键的值赋给content
            content = m.get("content", m)
            #content是消息内容，m是消息对象
            if isinstance(content, list):
                new_content = []
                for c in content:
                    # 如果发现是 video 类型的占位符
                    if isinstance(c, dict) and c.get("type") == "video":
                        # 替换为我们加载的 PIL 列表
                        new_content.append({"type": "video", "video": frames})
                    else:
                        new_content.append(c)
                new_messages.append({"role": m["role"], "content": new_content})
            else:
                # 如果只是纯文本，原样保留
                new_messages.append(m)
                
        # 4. 构建 Prompt 和 Full Text (用于 Label 计算)
        # 场景：messages = [User, Assistant, User, Assistant]
        # 我们需要让模型学习最后一个 Assistant 的回复。
        # prompt_messages: 包含到倒数第二条 (通常是最后一个 User 输入)
        prompt_messages = new_messages[:-1]
        
        # 使用 Processor 的 apply_chat_template 生成模型看得懂的文本格式
        # add_generation_prompt=True: 会在末尾加上 "<|im_start|>assistant\n" 引导生成
        text_prompt = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        
        # text_full: 包含所有内容 (User + Assistant)
        # 我们用这个来计算 Loss，但是会 Mask 掉 Prompt 部分
        text_full = self.processor.apply_chat_template(
            new_messages, tokenize=False, add_generation_prompt=False,
        )

        # 5. 编码 — 只调用一次 processor（原来调两次，视频编码开销翻倍）
        try:
            inputs = self.processor(
                text=[text_full],
                videos=[frames],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_seq_length,
            )
        except Exception:
            inputs = self.processor(
                text=[text_full],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_length,
            )

        # 6. 构建 Labels — 用纯 tokenizer 差值法算 prompt 长度，避免二次视频编码
        # prompt 和 full 共享相同的视频占位符，token 差值 = assistant 回复长度
        input_ids = inputs["input_ids"].squeeze(0)
        seq_len = int(input_ids.shape[0])

        tok_p = self.processor.tokenizer(
            text_prompt, return_tensors="pt",
            truncation=True, max_length=self.max_seq_length,
        )
        tok_f = self.processor.tokenizer(
            text_full, return_tensors="pt",
            truncation=True, max_length=self.max_seq_length,
        )
        assistant_token_len = max(
            0, tok_f["input_ids"].shape[1] - tok_p["input_ids"].shape[1]
        )
        prompt_len = max(0, seq_len - assistant_token_len)

        if prompt_len >= seq_len:
            prompt_len = max(0, seq_len - 1)

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        # 7. 组装返回字典
        out = {
            "input_ids": input_ids,
            # 如果没有 attention_mask，就创建全 1 的 (一般 Processor 都会返回)
            "attention_mask": inputs.get("attention_mask", torch.ones_like(input_ids)).squeeze(0),
            "labels": labels
        }
        # 如果是多模态模型，还需要把视频的像素值加进去
        if "pixel_values_videos" in inputs and inputs["pixel_values_videos"] is not None:
            out["pixel_values_videos"] = inputs["pixel_values_videos"].squeeze(0)
        if "video_grid_thw" in inputs and inputs["video_grid_thw"] is not None:
            out["video_grid_thw"] = inputs["video_grid_thw"].squeeze(0)
            
        return out


def build_train_dataset(
    train_path: Path,
    processor: Any,
    max_seq_length: int,
    project_root: Path,
    num_frames: int = 16,
) -> torch.utils.data.Dataset:
    """
    简单封装：从 JSON 文件路径构建 Dataset 对象。
    """
    with train_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return VideoAnomalyDataset(data, processor, max_seq_length, project_root, num_frames)


class VideoTrainDataCollator:
    """
    模块级 Collator，可被 multiprocessing pickle（Windows spawn 下必需）。
    仍建议 dataloader_num_workers=0，避免多进程重复占显存 / Processor 无法 pickle 的边界情况。
    """

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        pad_id = self.processor.tokenizer.pad_token_id or 0
        batch: dict[str, Any] = {}
        for k in ("input_ids", "attention_mask", "labels"):
            batch[k] = torch.nn.utils.rnn.pad_sequence(
                [f[k] if torch.is_tensor(f[k]) else torch.tensor(f[k]) for f in features],
                batch_first=True,
                padding_value=pad_id if k != "labels" else -100,
            )
        batch["labels"][batch["labels"] == pad_id] = -100
        if "pixel_values_videos" in features[0]:
            pv = [f["pixel_values_videos"] for f in features]
            if pv and torch.is_tensor(pv[0]):
                max_t = max(x.shape[0] for x in pv)
                padded = []
                for x in pv:
                    if x.shape[0] < max_t:
                        x = torch.cat(
                            [x, x[-1:].expand(max_t - x.shape[0], *x.shape[1:])],
                            dim=0,
                        )
                    padded.append(x)
                batch["pixel_values_videos"] = torch.stack(padded)
            if "video_grid_thw" in features[0]:
                batch["video_grid_thw"] = torch.stack([f["video_grid_thw"] for f in features])
        return batch


# ==============================================================================
# 主训练流程
# ==============================================================================
def main() -> None:
    # ==============================================================================
    # ==============================================================================
    #下面是都是传参数
    # --------------------------------------------------------------------------
    # 1. 参数解析 (命令行 + YAML Config)
    # 设计模式：命令行参数优先级 > YAML 配置文件 > 默认值
    # --------------------------------------------------------------------------
    ap = argparse.ArgumentParser(description="Qwen3.5-9B QLoRA 微调")
    
    # 配置文件路径（这是唯一一个有硬默认值的参数）
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    
    # 模型与数据路径 (覆盖 Config)
    ap.add_argument("--model_path", type=str, default="")
    ap.add_argument("--train_json", type=str, default="")
    ap.add_argument("--val_json", type=str, default="")
    ap.add_argument("--output_dir", type=str, default="")
    
    # 训练超参数 (覆盖 Config)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.0)
    
    args = ap.parse_args()

    # 确定项目根目录 (假设脚本在 project/src/ 下，根目录是 project/)
    project_root = Path(__file__).resolve().parent.parent

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
    # 加载 YAML 配置文件
    config_path = project_root / args.config
    # /就是路径相加
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        #用yaml函数的safe_load()把yaml文件转换成python对象
        #加上大括号 model：use_4bit: true training：learning_rate: 0.0001
        #就变成了{model: {use_4bit: True}, training: {learning_rate: 0.0001}}
        #这样就可以用cfg.get("model", {})来获取model的值 
    # 解析 Config 的各个子模块
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    lora_cfg = cfg.get("lora", {})
    train_cfg = cfg.get("training", {})
    clips_cfg = cfg.get("clips", {})
    num_frames = int(clips_cfg.get("clip_len", 16))
# ==============================================================================
# ==============================================================================
    # 逻辑：确定最终路径
    # 如果命令行传了 args.model_path 就用它，否则看 config，否则用 HuggingFace ID
    model_path = args.model_path or model_cfg.get("name_or_path", "Qwen/Qwen3.5-9B-Instruct")
    #name_or_path是model.fig的值
    #命令行 --model_path 非空 → 用命令行
    #否则 → 用 YAML 里 model.name_or_path
    #再否则 → 用 "Qwen/Qwen3.5-9B-Instruct"


    # 一个小的调试逻辑：如果 config 里设置了 lora_small (快速调试模式)，则换用小模型
    if (
        not args.model_path
        and model_cfg.get("lora_small")
        and model_cfg.get("name_or_path_2b")
    ):
        model_path = model_cfg["name_or_path_2b"]
        #这是快速调试的ai改的，目的是改小模型跑通流程，或许可以删掉config的参数配置，换模型直接改路径
# ==============================================================================
#下面这些很多or的东西是因为我写代码平常不用配置文件直接用文件的参数运行，后续选定了选一种就可以，不一定用配置文件
    # 同理处理其他路径
    train_json = args.train_json or data_cfg.get("train_json", "data/processed/train.json")
    output_dir = args.output_dir or train_cfg.get("output_dir", "outputs/qlora")
    
    # 相对路径转绝对路径（相对于项目根目录）
    train_json = project_root / train_json if not Path(train_json).is_absolute() else Path(train_json)
    output_dir = project_root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)

    # 同理处理超参数
    # ===== [修改] 按需求固定训练 2 个 epoch（不使用配置文件里的 num_train_epochs） =====
    num_epochs = 2
    batch_size = args.batch_size or train_cfg.get("per_device_train_batch_size", 2)
    lr = args.lr if args.lr > 0 else train_cfg.get("learning_rate", 1e-4)
    max_seq_length = train_cfg.get("max_seq_length", 2048)

    # SwanLab / 实验追踪（环境变量须在 TrainingArguments 构造前设置）
    sw_proj = str(train_cfg.get("swanlab_project") or "").strip()
    sw_ws = str(train_cfg.get("swanlab_workspace") or "").strip()
    if sw_proj:
        os.environ["SWANLAB_PROJ_NAME"] = sw_proj
    if sw_ws:
        os.environ["SWANLAB_WORKSPACE"] = sw_ws

    report_to_raw = train_cfg.get("report_to", [])
    if isinstance(report_to_raw, str):
        low = report_to_raw.strip().lower()
        if low in ("none", "null", ""):
            report_to_list: list[str] = []
        else:
            report_to_list = [report_to_raw]
    else:
        report_to_list = [str(x) for x in (report_to_raw or [])]

    run_name_cfg = str(train_cfg.get("run_name") or "").strip()
    hf_callbacks: list[Any] = []
    wants_swanlab = any(str(x).lower() == "swanlab" for x in report_to_list)
    if wants_swanlab and not _transformers_version_at_least(4, 50):
        try:
            from swanlab.integration.transformers import SwanLabCallback
        except ImportError as e:
            raise ImportError(
                "report_to 含 swanlab 且 transformers<4.50 时需 SwanLabCallback，请先 pip install swanlab"
            ) from e
        cb_kw: dict[str, Any] = {}
        if run_name_cfg:
            cb_kw["experiment_name"] = run_name_cfg
        if sw_proj:
            cb_kw["project"] = sw_proj
        hf_callbacks.append(SwanLabCallback(**cb_kw))
        report_to_eff: list[str] | str = [x for x in report_to_list if str(x).lower() != "swanlab"]
        if not report_to_eff:
            report_to_eff = "none"
    else:
        report_to_eff = report_to_list

# ==============================================================================
    # --------------------------------------------------------------------------
    # 2. 加载依赖与模型
    
    # --------------------------------------------------------------------------
    (
        AutoProcessor, Qwen3_5ForConditionalGeneration, TrainingArguments, Trainer,
        BitsAndBytesConfig, LoraConfig, get_peft_model, prepare_model_for_kbit_training,
        Dataset,
    ) = _import_train_deps()

    #AutoProcessor是处理器  #处理文本+视频
    #Qwen3_5ForConditionalGeneration是模型
    #TrainingArguments是训练参数 
    #Trainer是训练器
    #BitsAndBytesConfig是量化配置
    #LoraConfig是LoRA配置
    #get_peft_model是挂载LoRA
    #prepare_model_for_kbit_training是准备模型
    #Dataset是数据集

    # --- 4bit 量化配置 (BitsAndBytes) ---
    use_4bit = model_cfg.get("use_4bit", True)
    # 计算 dtype：通常是 bfloat16 (如果显卡支持)，这是 QLoRA 推荐的
    compute_dtype = getattr(torch, model_cfg.get("bnb_4bit_compute_dtype", "bfloat16"), torch.bfloat16)
    
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,                     # 开启 4bit 加载
            bnb_4bit_compute_dtype=compute_dtype,  # 计算时用的 dtype (bf16 加速)
            bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"), # 量化类型：nf4 (Normalized Float 4bit) 对 LLM 最好
            bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True), # 双重量化：对量化常数再量化，再省一点显存
        )

    print("加载 Processor 与模型...")
    
    # 加载 Processor，并限制视觉 token 预算以控制显存
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    _max_pix = int(clips_cfg.get("train_max_pixels", 129600))
    _min_pix = int(clips_cfg.get("train_min_pixels", 3136))
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = _max_pix
        processor.image_processor.min_pixels = _min_pix
        print(f"图像 processor: min_pixels={_min_pix}, max_pixels={_max_pix}")
    # 视频训练走 videos=...，需同步限制 video_processor，否则显存仍按默认上限走
    vp = getattr(processor, "video_processor", None)
    if vp is not None and hasattr(vp, "max_pixels"):
        vp.max_pixels = _max_pix
        vp.min_pixels = _min_pix
        print(f"视频 processor: min_pixels={_min_pix}, max_pixels={_max_pix}")
    hf_callbacks.append(EvalReadyCheckpointCallback(processor))
    
    # 加载基座模型 (Pre-trained Model)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=quantization_config, # 传入量化配置
        device_map="auto",                        # 自动分配模型层到 GPU/CPU
        trust_remote_code=True,                   # 允许执行模型仓里的自定义代码
    )
    
    # --- 准备 LoRA ---
    # 步骤1：将模型转为 "kbit 训练模式"
    # 这会冻结所有原始权重，并将 LayerNorm 等层转为 fp32 以保证训练稳定性
    model = prepare_model_for_kbit_training(model)
    
    # 步骤2：定义 LoRA 超参数
    lora_r = lora_cfg.get("r", 64)          # LoRA Rank (秩)，越大参数量越大，效果通常越好
    lora_alpha = lora_cfg.get("lora_alpha", 128) # LoRA Alpha (缩放因子)，通常设为 Rank 的 2倍
    
    # 调试模式覆盖
    if model_cfg.get("lora_small", False):
        lora_r = model_cfg.get("lora_r", 32)
        lora_alpha = model_cfg.get("lora_alpha_small", 64)
        
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_cfg.get("lora_dropout", 0.05), # LoRA 层的 Dropout
        # 目标模块：通常是 Attention 的 Q/K/V/O，以及 MLP 的 gate/up/down
        # 针对 Qwen，通常 target 这些模块以获得较好效果
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"), # 任务类型：因果语言模型
    )
    
    # 步骤3：将 LoRA Adapter 挂载到模型上
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数量，让用户心里有数
    model.print_trainable_parameters()

    # ===== [修改] 第 2 个 epoch 解冻部分视觉层（需在 model 存在后注册） =====
    hf_callbacks.append(VisionUnfreezeCallback(model, unfreeze_epoch=1))

    # --- 梯度检查点 (Gradient Checkpointing) ---
    # 这是另一个显存节省神器：不保存前向传播的中间激活值，而是在反向时重新计算
    # 代价是训练速度变慢 (约 20% - 30%)
    if train_cfg.get("gradient_checkpointing", True):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    # --------------------------------------------------------------------------
    # 3. 准备数据
    # --------------------------------------------------------------------------
    print("构建训练集...")
    train_dataset = build_train_dataset(train_json, processor, max_seq_length, project_root, num_frames)
    
    # ===== [修改] 不计算验证集 loss：不加载 val、不做 eval =====
    eval_dataset = None
    eval_strategy_eff = "no"

    # --------------------------------------------------------------------------
    # 4. 训练参数 (TrainingArguments)
    # --------------------------------------------------------------------------
    ta_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": train_cfg.get("gradient_accumulation_steps", 8),
        "learning_rate": lr,
        "lr_scheduler_type": train_cfg.get("lr_scheduler_type", "cosine"),
        "warmup_ratio": train_cfg.get("warmup_ratio", 0.05),
        "weight_decay": train_cfg.get("weight_decay", 0.01),
        "bf16": train_cfg.get("bf16", True),
        "max_grad_norm": train_cfg.get("max_grad_norm", 1.0),
        "logging_steps": train_cfg.get("logging_steps", 10),
        "logging_first_step": train_cfg.get("logging_first_step", True),
        "save_steps": train_cfg.get("save_steps", 100),
        "save_total_limit": train_cfg.get("save_total_limit", 3),
        "report_to": report_to_eff,
        "eval_strategy": eval_strategy_eff,
        # 必须为 0：main 内嵌 collator 已改为模块级类；多进程仍会 pickle collator，
        # 且子进程常额外吃显存。勿在 YAML 开大 workers。
        "dataloader_num_workers": 0,
        "optim": train_cfg.get("optim", "paged_adamw_8bit"),
        "tf32": train_cfg.get("tf32", True),
    }
    if eval_strategy_eff == "steps":
        ta_kwargs["eval_steps"] = int(train_cfg.get("eval_steps", 500))
    if eval_dataset is not None and eval_strategy_eff != "no":
        ta_kwargs["per_device_eval_batch_size"] = int(train_cfg.get("per_device_eval_batch_size", 1))
    if run_name_cfg:
        ta_kwargs["run_name"] = run_name_cfg
    training_args = TrainingArguments(**ta_kwargs)

    data_collator = VideoTrainDataCollator(processor)

    # --------------------------------------------------------------------------
    # 6. 开始训练
    # --------------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=hf_callbacks,
    )
    
    # 执行训练循环（显式打印异常栈，便于区分 OOM / 代码 bug / 被系统杀进程）
    try:
        train_out = trainer.train()
        print("[train] 训练结束，汇总指标:", train_out.metrics, flush=True)
    except Exception as e:
        print(f"\n[train] 训练异常: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        if torch.cuda.is_available():
            print(
                "[train] 若为显存不足：可减小 training.max_seq_length、clips.clip_len、"
                "per_device_train_batch_size；或设环境变量 "
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True（PyTorch2+）减轻碎片。",
                file=sys.stderr,
            )
        raise

    # --------------------------------------------------------------------------
    # 7. 保存最终模型
    # --------------------------------------------------------------------------
    final_dir = output_dir / "final"
    # 保存 Adapter (LoRA 权重很小，通常只有几十MB)
    trainer.save_model(str(final_dir))
    # 保存 Processor (方便推理时直接加载)
    processor.save_pretrained(str(final_dir))
    
    print("训练完成，已保存至", output_dir / "final")


if __name__ == "__main__":
    main()