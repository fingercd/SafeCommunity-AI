"""Qwen3.5-9B QLoRA fine-tuning for video anomaly (train.json: messages + frames_dir)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
import yaml

from vlm.utils import load_frames_from_dir, resolve_frames_dir

logger = logging.getLogger(__name__)

# Trainer callbacks must subclass TrainerCallback (or object fallback if unavailable).
try:
    from transformers import TrainerCallback as _TrainerCallback
except ImportError:
    _TrainerCallback = object

def _import_train_deps():
    """Lazy-import heavy training deps (transformers, peft)."""
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3_5ForConditionalGeneration,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    return (
        AutoProcessor,
        Qwen3_5ForConditionalGeneration,
        TrainingArguments,
        Trainer,
        BitsAndBytesConfig,
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
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
        model = kwargs.get("model")
        if model is None:
            return control
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_dir))
        self.processor.save_pretrained(str(ckpt_dir))
        print(f"[train] 已补全可评估 checkpoint: {ckpt_dir}", flush=True)
        return control


class VisionUnfreezeCallback(_TrainerCallback):
    """
    After epoch index `unfreeze_epoch` (0-based), unfreeze selected visual layers.
    Only float32/fp16/bf16 params are toggled (4bit weights stay frozen for backprop safety).
    Targets: visual.merger.* and norm layers in visual.blocks.24/25/26.
    """

    def __init__(self, model: Any, unfreeze_epoch: int = 1) -> None:
        self.model = model
        self.unfreeze_epoch = unfreeze_epoch
        self._done = False

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self._done or state.epoch < self.unfreeze_epoch:
            return control
        self._done = True
        thawed: list[str] = []
        for name, p in self.model.named_parameters():
            if "lora" in name.lower():
                continue
            if p.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                continue
            if "visual.merger" in name:
                p.requires_grad = True
                thawed.append(name)
                continue
            if any(f"blocks.{i}." in name for i in (24, 25, 26)) and "norm" in name.lower():
                p.requires_grad = True
                thawed.append(name)
                continue
        print(
            f"[train] Unfroze partial visual layers: {len(thawed)} parameter tensors",
            flush=True,
        )
        for n in thawed[:30]:
            print(f"  trainable visual: {n}", flush=True)
        if len(thawed) > 30:
            print(f"  ... 另有 {len(thawed) - 30} 项已解冻（省略打印）", flush=True)

        if hasattr(self.model, "print_trainable_parameters"):
            self.model.print_trainable_parameters()
        return control


class VideoAnomalyDataset(torch.utils.data.Dataset):
    """Loads frames, runs processor (video+text), labels only assistant tokens."""

    def __init__(
        self,
        data: list[dict],
        processor: Any,
        max_seq_length: int,
        project_root: Path,
        num_frames: int = 16,
    ):
        self.data = data
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.project_root = project_root
        self.num_frames = num_frames
        
        self.valid_indices = []
        for i, item in enumerate(data):
            if item.get("frames_dir") and item.get("messages"):
                self.valid_indices.append(i)
        if not self.valid_indices:
            raise ValueError("没有有效训练样本（需含 frames_dir 与 messages）")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        item = self.data[real_idx]

        messages = item["messages"]
        frames_dir = item["frames_dir"]

        abs_frames = resolve_frames_dir(frames_dir, self.project_root)
        frames = load_frames_from_dir(abs_frames, self.num_frames)

        new_messages = []
        for m in messages:
            content = m.get("content", m)
            if isinstance(content, list):
                new_content = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "video":
                        new_content.append({"type": "video", "video": frames})
                    else:
                        new_content.append(c)
                new_messages.append({"role": m["role"], "content": new_content})
            else:
                new_messages.append(m)

        prompt_messages = new_messages[:-1]

        text_prompt = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )

        text_full = self.processor.apply_chat_template(
            new_messages, tokenize=False, add_generation_prompt=False,
        )

        try:
            inputs = self.processor(
                text=[text_full],
                videos=[frames],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_seq_length,
            )
        except Exception as e:
            logger.warning(
                "Processor encode with videos failed, text-only fallback: %s",
                e,
            )
            inputs = self.processor(
                text=[text_full],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_length,
            )
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

        out = {
            "input_ids": input_ids,
            "attention_mask": inputs.get("attention_mask", torch.ones_like(input_ids)).squeeze(0),
            "labels": labels
        }
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
    with train_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return VideoAnomalyDataset(data, processor, max_seq_length, project_root, num_frames)


class VideoTrainDataCollator:
    """Pickle-safe collator for Windows spawn; keep dataloader_num_workers=0 if OOM."""

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


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen3.5-9B QLoRA 微调")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--model_path", type=str, default="")
    ap.add_argument("--train_json", type=str, default="")
    ap.add_argument("--val_json", type=str, default="")
    ap.add_argument("--output_dir", type=str, default="")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.0)

    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    config_path = project_root / args.config
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    lora_cfg = cfg.get("lora", {})
    train_cfg = cfg.get("training", {})
    clips_cfg = cfg.get("clips", {})
    num_frames = int(clips_cfg.get("clip_len", 16))

    model_path = args.model_path or model_cfg.get("name_or_path", "Qwen/Qwen3.5-9B-Instruct")

    if (
        not args.model_path
        and model_cfg.get("lora_small")
        and model_cfg.get("name_or_path_2b")
    ):
        model_path = model_cfg["name_or_path_2b"]

    train_json = args.train_json or data_cfg.get("train_json", "data/processed/train.json")
    output_dir = args.output_dir or train_cfg.get("output_dir", "outputs/qlora")

    train_json = project_root / train_json if not Path(train_json).is_absolute() else Path(train_json)
    output_dir = project_root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)

    num_epochs = (
        int(args.epochs)
        if args.epochs > 0
        else int(train_cfg.get("num_train_epochs", 2))
    )
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

    (
        AutoProcessor,
        Qwen3_5ForConditionalGeneration,
        TrainingArguments,
        Trainer,
        BitsAndBytesConfig,
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    ) = _import_train_deps()

    use_4bit = model_cfg.get("use_4bit", True)
    compute_dtype = getattr(torch, model_cfg.get("bnb_4bit_compute_dtype", "bfloat16"), torch.bfloat16)

    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True),
        )

    print("加载 Processor 与模型...")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    _max_pix = int(clips_cfg.get("train_max_pixels", 129600))
    _min_pix = int(clips_cfg.get("train_min_pixels", 3136))
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = _max_pix
        processor.image_processor.min_pixels = _min_pix
        print(f"图像 processor: min_pixels={_min_pix}, max_pixels={_max_pix}")
    vp = getattr(processor, "video_processor", None)
    if vp is not None and hasattr(vp, "max_pixels"):
        vp.max_pixels = _max_pix
        vp.min_pixels = _min_pix
        print(f"视频 processor: min_pixels={_min_pix}, max_pixels={_max_pix}")
    hf_callbacks.append(EvalReadyCheckpointCallback(processor))
    
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    lora_r = lora_cfg.get("r", 64)
    lora_alpha = lora_cfg.get("lora_alpha", 128)

    if model_cfg.get("lora_small", False):
        lora_r = model_cfg.get("lora_r", 32)
        lora_alpha = model_cfg.get("lora_alpha_small", 64)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    hf_callbacks.append(VisionUnfreezeCallback(model, unfreeze_epoch=1))

    if train_cfg.get("gradient_checkpointing", True):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    print("构建训练集...")
    train_dataset = build_train_dataset(train_json, processor, max_seq_length, project_root, num_frames)

    eval_dataset = None
    eval_strategy_eff = "no"

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=hf_callbacks,
    )
    
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

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    
    print("训练完成，已保存至", output_dir / "final")


if __name__ == "__main__":
    main()