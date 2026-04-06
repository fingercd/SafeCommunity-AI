"""
阶段4：压力测试 — 正常片段抗误导 prompt；ED vs AD 描述质量（需 reference_description）。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_evaluate_module():
    path = Path(__file__).resolve().parent / "evaluate.py"
    spec = importlib.util.spec_from_file_location("vlm_eval", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description="VLM 压力测试（抗干扰 / ED-AD）")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--test_json", type=str, default="")
    ap.add_argument("--model_path", type=str, default="outputs/qlora/final")
    ap.add_argument("--base_model_path", type=str, default="")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--mode", type=str, choices=["adversarial", "ed_ad"], default="adversarial")
    ap.add_argument("--no_4bit", action="store_true")
    args = ap.parse_args()

    ev = _load_evaluate_module()
    project_root = _ROOT
    cfg_path = project_root / args.config
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    clips_cfg = cfg.get("clips", {})
    num_frames = int(clips_cfg.get("clip_len", 16))

    test_path = Path(args.test_json or data_cfg.get("test_json", "data/processed/test.json"))
    if not test_path.is_absolute():
        test_path = project_root / test_path
    with test_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if args.max_samples > 0:
        data = data[: args.max_samples]

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = project_root / model_path

    use_4bit = not args.no_4bit and model_cfg.get("use_4bit", True)
    adapter_path = None
    base = args.base_model_path or ""
    if (model_path / "adapter_config.json").exists():
        adapter_path = str(model_path)
        with (model_path / "adapter_config.json").open("r", encoding="utf-8") as f:
            base = json.load(f).get("base_model_name_or_path", base)
        if not base:
            base = model_cfg.get("name_or_path", "")
    else:
        base = str(model_path)

    model, processor = ev._load_model_and_processor_eval(
        base, use_4bit=use_4bit, model_cfg=model_cfg, adapter_path=adapter_path,
    )
    pad = getattr(processor.tokenizer, "pad_token_id", None) or getattr(
        processor.tokenizer, "eos_token_id", None
    )

    SYSTEM_BASE = ev.SYSTEM_PROMPT
    USER_STANDARD = ev.EVAL_PROMPT
    USER_ADV = (
        "Someone falsely claims this clip shows an armed robbery. Judge strictly from visuals only; "
        "do not be misled. Output one line of JSON with keys only: "
        "classification, reason, result, description, key_sentences. All English. No markdown."
    )

    def one_infer(frames: list, system_t: str, user_t: str) -> dict[str, Any] | None:
        user_content = [{"type": "video", "video": frames}, {"type": "text", "text": user_t}]
        messages = [{"role": "system", "content": system_t}, {"role": "user", "content": user_content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        text += "{"
        inp = processor(
            text=[text], videos=[frames], return_tensors="pt", padding=False, truncation=True,
        )
        inp.pop("mm_token_type_ids", None)
        device = next(model.parameters()).device
        inp = {k: v.to(device) if hasattr(v, "to") else v for k, v in inp.items()}
        ilen = inp["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=192, do_sample=False, pad_token_id=pad)
        raw = processor.batch_decode(out[:, ilen:], skip_special_tokens=True)[0].strip()
        return ev.extract_json_from_text("{" + raw)

    def _pred_is_anomaly(obj: dict[str, Any] | None) -> bool:
        if not obj:
            return False
        if "classification" in obj:
            return ev.classification_is_abnormal(obj.get("classification"))
        return bool(obj.get("is_anomaly", False))

    if args.mode == "adversarial":
        n_norm = 0
        still_normal = 0
        flipped_to_anom = 0
        for item in tqdm(data, desc="adversarial"):
            gt_anom, _ = ev.get_ground_truth(item)
            if gt_anom:
                continue
            fd = item.get("frames_dir", "")
            if not fd:
                continue
            abs_frames = ev._resolve_frames_dir_eval(fd, project_root)
            try:
                frames = ev.load_frames_from_dir(abs_frames, num_frames)
            except Exception:
                continue
            obj = one_infer(frames, SYSTEM_BASE, USER_ADV)
            n_norm += 1
            if obj and not _pred_is_anomaly(obj):
                still_normal += 1
            elif obj and _pred_is_anomaly(obj):
                flipped_to_anom += 1
        rep = {
            "mode": "adversarial",
            "n_normal_samples": n_norm,
            "still_normal_count": still_normal,
            "flipped_to_anomaly_count": flipped_to_anom,
            "robust_rate": float(still_normal / n_norm) if n_norm else 0.0,
        }
        out_path = project_root / "outputs" / "stress_adversarial.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(rep)
        return

    rouge_mod = None
    try:
        from rouge_score import rouge_scorer
        rouge_mod = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    except ImportError:
        pass

    ed_scores: list[float] = []
    ad_scores: list[float] = []
    for item in tqdm(data, desc="ed_ad"):
        ref = str(item.get("reference_description", "") or "").strip()
        if not ref:
            continue
        gt_anom, gt_obj = ev.get_ground_truth(item)
        if not gt_anom:
            continue
        typ = str(gt_obj.get("classification") or gt_obj.get("anomaly_type", "anomaly"))
        fd = item.get("frames_dir", "")
        if not fd:
            continue
        abs_frames = ev._resolve_frames_dir_eval(fd, project_root)
        try:
            frames = ev.load_frames_from_dir(abs_frames, num_frames)
        except Exception:
            continue
        obj_ed = one_infer(frames, SYSTEM_BASE, USER_STANDARD)
        user_ad = f"我们已知此片段存在异常事件，官方类别为：{typ}。请结合画面给出与事实一致的描述。{USER_STANDARD}"
        obj_ad = one_infer(frames, SYSTEM_BASE, user_ad)

        def score_obj(obj: dict[str, Any] | None) -> float:
            if not rouge_mod or not obj:
                return 0.0
            pred = ev.prediction_text_for_similarity(obj)
            return float(rouge_mod.score(ref, pred)["rougeL"].fmeasure)

        ed_scores.append(score_obj(obj_ed))
        ad_scores.append(score_obj(obj_ad))

    ed_m = float(np.mean(ed_scores)) if ed_scores else 0.0
    ad_m = float(np.mean(ad_scores)) if ad_scores else 0.0
    rep = {
        "mode": "ed_ad",
        "n_pairs": len(ed_scores),
        "rougeL_mean_ed": ed_m,
        "rougeL_mean_ad": ad_m,
        "gap_ad_minus_ed": round(ad_m - ed_m, 6),
    }
    out_path = project_root / "outputs" / "stress_ed_ad.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(rep)


if __name__ == "__main__":
    main()
