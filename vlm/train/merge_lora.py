"""
将 LoRA adapter 合并回 Qwen3.5-9B 基座并保存完整模型。

用法:
  python -m vlm.train.merge_lora --base_model Qwen/Qwen3.5-9B-Instruct --adapter_path outputs/qlora/final --output_path outputs/merged
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml


def main() -> None:
    ap = argparse.ArgumentParser(description="合并 LoRA 权重到基座")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--base_model", type=str, default="")
    ap.add_argument("--adapter_path", type=str, default="outputs/qlora/final")
    ap.add_argument("--output_path", type=str, default="outputs/merged")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / args.config
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        model_cfg = cfg.get("model", {})
        base_model = args.base_model or model_cfg.get("name_or_path", "Qwen/Qwen3.5-9B-Instruct")
    else:
        base_model = args.base_model or "Qwen/Qwen3.5-9B-Instruct"

    adapter_path = Path(args.adapter_path)
    if not adapter_path.is_absolute():
        adapter_path = project_root / adapter_path
    output_path = Path(args.output_path)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from peft import PeftModel

    print("加载基座模型...")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("加载 LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    print("合并权重...")
    model = model.merge_and_unload()
    print("保存合并后模型...")
    model.save_pretrained(str(output_path))
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    processor.save_pretrained(str(output_path))
    print("已保存至:", output_path)


if __name__ == "__main__":
    main()
