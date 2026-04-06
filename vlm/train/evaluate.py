"""
在 test split 上评估 Qwen3.5-9B：AUC、准确率、JSON 合规率、FPR@95%TPR。

支持 4bit 量化、小批量推理、OOM 自动降级、断点续跑、
逐条完整预测 JSON 持久化（jsonl + 周期性 json 快照）、
单次运行先评基座再评微调（--dual_eval）。
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm import tqdm

from PIL import Image

SYSTEM_PROMPT = (
    "You are an expert in surveillance video analysis. Watch the clip and output exactly one "
    "single-line JSON. Fields: classification, reason, result, description, key_sentences. "
    "classification must be \"normal\" for routine activity, otherwise an English event type. "
    "All text must be English. Do not output Chinese. Do not use markdown."
)
EVAL_PROMPT = (
    "Analyze this surveillance clip. Output one line of JSON only: "
    '{"classification": "normal" or English event type, "reason": "...", "result": "...", '
    '"description": "...", "key_sentences": ["...","..."]}. '
    "No extra keys."
)


def classification_is_abnormal(classification: str | None) -> bool:
    """与训练标签一致：normal / Non-Anomaly / 空 视为正常，其余为异常。"""
    c = str(classification or "normal").strip().lower()
    return c not in ("normal", "non-anomaly", "")


def ground_truth_anomaly_from_obj(obj: dict[str, Any] | None) -> bool:
    if not obj:
        return False
    if "classification" in obj:
        return classification_is_abnormal(obj.get("classification"))
    return bool(obj.get("is_anomaly", False))


def _resolve_frames_dir_eval(frames_dir: str | Path, project_root: Path) -> Path:
    p = Path(frames_dir)
    if p.is_absolute() and p.exists():
        return p
    name = p.name
    for sub in ("ecva_clips", "clips"):
        cand = project_root / "data" / "processed" / sub / name
        if cand.exists():
            return cand
    if p.exists():
        return p
    return project_root / "data" / "processed" / "clips" / name


def load_frames_from_dir(frames_dir: str | Path, num_frames: int = 16) -> list:
    frames_dir = Path(frames_dir)
    imgs = sorted(frames_dir.glob("frame_*.jpg")) or sorted(frames_dir.glob("*.jpg"))
    imgs = imgs[:num_frames]
    out = []
    for p in imgs:
        out.append(Image.open(p).convert("RGB"))
    while len(out) < num_frames and out:
        out.append(out[-1])
    return out[:num_frames]


def extract_json_from_text(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except json.JSONDecodeError:
                continue
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def get_ground_truth(item: dict) -> tuple[bool, dict]:
    """从 test 条目的 messages 中取 assistant 的 JSON 作为 GT。返回 (is_anomaly, parsed_dict)。"""
    for m in item.get("messages", []):
        if m.get("role") == "assistant":
            raw = m.get("content", "")
            if isinstance(raw, dict):
                return (ground_truth_anomaly_from_obj(raw), raw)
            obj = extract_json_from_text(raw) if isinstance(raw, str) else None
            if obj is None:
                try:
                    obj = json.loads(raw)
                except Exception:
                    return (False, {})
            return (ground_truth_anomaly_from_obj(obj), obj)
    return (False, {})


def prediction_text_for_similarity(obj: dict[str, Any] | None) -> str:
    if not obj:
        return ""
    keys = obj.get("key_sentences")
    key_txt = ""
    if isinstance(keys, list):
        key_txt = " ".join(str(x) for x in keys if x)
    elif isinstance(keys, str) and keys.strip():
        key_txt = keys.strip()
    parts = [
        str(obj.get("reason", "") or ""),
        str(obj.get("result", "") or ""),
        str(obj.get("description", "") or ""),
        key_txt,
    ]
    if not any((p or "").strip() for p in parts):
        parts = [
            str(obj.get("scene_description", "") or ""),
            str(obj.get("action_description", "") or ""),
            str(obj.get("reasoning", "") or ""),
        ]
    return " ".join(p for p in parts if p).strip()


def ecva_text_for_similarity(item: dict[str, Any], gt_obj: dict[str, Any] | None = None) -> str:
    """拼出 ECVA 样本自身的说明文本，供向量相似度使用。"""
    parts: list[str] = []
    ref_desc = str(item.get("reference_description", "") or "").strip()
    if ref_desc:
        parts.append(ref_desc)
    key_sentences = item.get("reference_key_sentences", [])
    if isinstance(key_sentences, list):
        for sent in key_sentences:
            sent_t = str(sent or "").strip()
            if sent_t:
                parts.append(sent_t)

    # 兼容旧数据：如果样本里没有 reference_*，退回 GT JSON 里的解释字段。
    if not parts and gt_obj:
        for k in ("reason", "result", "description", "scene_description", "action_description", "reasoning"):
            txt = str(gt_obj.get(k, "") or "").strip()
            if txt:
                parts.append(txt)
        ks = gt_obj.get("key_sentences")
        if isinstance(ks, list):
            for sent in ks:
                st = str(sent or "").strip()
                if st:
                    parts.append(st)
    return " ".join(parts).strip()


def _cosine_similarity_from_text(text_a: str, text_b: str) -> float | None:
    text_a = (text_a or "").strip()
    text_b = (text_b or "").strip()
    if not text_a or not text_b:
        return None
    try:
        from sklearn.feature_extraction.text import HashingVectorizer
    except ImportError:
        return None

    vectorizer = HashingVectorizer(
        n_features=4096,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
    )
    vectors = vectorizer.transform([text_a, text_b])
    score = float(vectors[0].multiply(vectors[1]).sum())
    return max(0.0, min(1.0, score))


def _should_skip_ecva_similarity(
    item: dict[str, Any] | None = None,
    clip_id: str = "",
) -> bool:
    """UCF normal 没有真实参考解释，跳过 ECVA 解释相似度统计。"""
    row = item or {}
    cid = str(clip_id or row.get("clip_id", "") or "").strip().lower()
    source = str(row.get("source", "") or "").strip().lower()
    return source == "ucf_normal" or cid.startswith("ucf_normal_")


def _enrich_prediction_records(
    records: list[dict[str, Any]],
    item_by_clip_id: dict[str, dict[str, Any]],
) -> None:
    """给历史/新预测统一补齐 ECVA 文本与向量相似度字段。"""
    for rec in records:
        clip_id = str(rec.get("clip_id", "") or "")
        item = item_by_clip_id.get(clip_id, {})
        gt_obj = rec.get("ground_truth_json")
        if not isinstance(gt_obj, dict):
            gt_obj = {}

        rec.setdefault("frames_dir", str(item.get("frames_dir", "") or rec.get("frames_dir", "") or ""))
        rec.setdefault("reference_key_sentences", item.get("reference_key_sentences", []))
        pred_text = str(rec.get("prediction_text", "") or rec.get("pred_text", "") or "").strip()
        if pred_text and not rec.get("prediction_text"):
            rec["prediction_text"] = pred_text

        if _should_skip_ecva_similarity(item, clip_id):
            rec["ecva_reference_text"] = ""
            rec["explanation_ecva_similarity"] = None
            continue

        ecva_ref_text = str(rec.get("ecva_reference_text", "") or "").strip()
        if not ecva_ref_text:
            ecva_ref_text = ecva_text_for_similarity(item, gt_obj)
            rec["ecva_reference_text"] = ecva_ref_text

        similarity = _cosine_similarity_from_text(pred_text, ecva_ref_text)
        rec["explanation_ecva_similarity"] = similarity


def _mean_rouge_l(refs: list[str], preds: list[str]) -> float:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return 0.0
    if not refs or not preds or len(refs) != len(preds):
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for ref, pred in zip(refs, preds):
        ref_t = (ref or "").strip()
        pred_t = (pred or "").strip()
        if not ref_t and not pred_t:
            continue
        scores.append(scorer.score(ref_t, pred_t)["rougeL"].fmeasure)
    return float(sum(scores) / len(scores)) if scores else 0.0


def _mean_bertscore_f1(refs: list[str], preds: list[str], lang: str = "en") -> float:
    try:
        import bert_score
    except ImportError:
        return 0.0
    if not refs or not preds or len(refs) != len(preds):
        return 0.0
    filt_refs: list[str] = []
    filt_preds: list[str] = []
    for r, p in zip(refs, preds):
        rt = (r or "").strip()
        pt = (p or "").strip()
        if not rt:
            continue
        filt_refs.append(rt)
        filt_preds.append(pt or " ")
    if not filt_refs:
        return 0.0
    try:
        _, _, f1 = bert_score.score(
            filt_preds,
            filt_refs,
            lang=lang,
            rescale_with_baseline=False,
            verbose=False,
        )
        return float(f1.mean().item())
    except Exception:
        return 0.0


def _load_model_and_processor_eval(
    model_path: str,
    use_4bit: bool = True,
    model_cfg: dict | None = None,
    adapter_path: str | None = None,
):
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from transformers import BitsAndBytesConfig

    model_cfg = model_cfg or {}
    processor = AutoProcessor.from_pretrained(
        adapter_path or model_path, trust_remote_code=True
    )

    quantization_config = None
    if use_4bit:
        compute_dtype = getattr(
            torch,
            model_cfg.get("bnb_4bit_compute_dtype", "bfloat16"),
            torch.bfloat16,
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True),
        )
    device_map = "auto" if use_4bit else "cuda:0"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if not use_4bit else None,
        trust_remote_code=True,
    )

    if adapter_path:
        from peft import PeftModel

        print(f"挂载 LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, processor


def _normalize_prediction_record(r: dict[str, Any], model_tag: str) -> dict[str, Any]:
    """兼容旧版仅含 y_true/ref_text 的 jsonl 行。"""
    out = dict(r)
    out.setdefault("model_tag", model_tag)
    out.setdefault("frames_dir", "")
    out.setdefault("ground_truth_json", {})
    out.setdefault("ground_truth_is_anomaly", bool(r.get("y_true", 0)))
    out.setdefault("raw_response", "")
    out.setdefault("parsed_prediction_json", None)
    out.setdefault("predicted_is_anomaly", False)
    out.setdefault("confidence", 0.5)
    out.setdefault("json_ok", False)
    out.setdefault("reference_text", str(r.get("ref_text", "") or ""))
    out.setdefault("prediction_text", str(r.get("pred_text", "") or ""))
    if "ref_text" in out and "reference_text" not in r:
        out["reference_text"] = r.get("ref_text", "")
    if "pred_text" in out and "prediction_text" not in r:
        out["prediction_text"] = r.get("pred_text", "")
    out.setdefault("reference_key_sentences", [])
    out.setdefault("ecva_reference_text", "")
    out.setdefault("explanation_ecva_similarity", None)
    return out


def _load_predictions_jsonl(path: Path, model_tag: str) -> tuple[set[str], list[dict[str, Any]]]:
    done_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    if not path.exists():
        return done_ids, records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            rec = _normalize_prediction_record(raw, model_tag)
            done_ids.add(rec["clip_id"])
            records.append(rec)
    return done_ids, records


def _append_predictions_jsonl(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_predictions_snapshot(
    predictions_json: Path, model_tag: str, records: list[dict[str, Any]]
) -> None:
    predictions_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_tag": model_tag,
        "n_predictions": len(records),
        "predictions": records,
    }
    with predictions_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _pad_and_batch(inputs_list: list[dict]) -> dict:
    if len(inputs_list) == 1:
        return inputs_list[0]
    id_tensors = [inp["input_ids"].squeeze(0) for inp in inputs_list]
    max_len = max(t.shape[0] for t in id_tensors)
    pad_id = 0
    batched: dict[str, Any] = {}
    padded_ids = []
    padded_mask = []
    for ids in id_tensors:
        pad_len = max_len - ids.shape[0]
        padded_ids.append(torch.cat([torch.full((pad_len,), pad_id, dtype=ids.dtype), ids]))
        padded_mask.append(
            torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.ones(ids.shape[0], dtype=torch.long)])
        )
    batched["input_ids"] = torch.stack(padded_ids)
    batched["attention_mask"] = torch.stack(padded_mask)
    if all("pixel_values_videos" in inp for inp in inputs_list):
        batched["pixel_values_videos"] = torch.cat(
            [inp["pixel_values_videos"] for inp in inputs_list], dim=0
        )
    if all("video_grid_thw" in inp for inp in inputs_list):
        batched["video_grid_thw"] = torch.cat(
            [inp["video_grid_thw"] for inp in inputs_list], dim=0
        )
    if all("pixel_values" in inp for inp in inputs_list):
        batched["pixel_values"] = torch.cat(
            [inp["pixel_values"] for inp in inputs_list], dim=0
        )
    if all("image_grid_thw" in inp for inp in inputs_list):
        batched["image_grid_thw"] = torch.cat(
            [inp["image_grid_thw"] for inp in inputs_list], dim=0
        )
    return batched


def _reports_to_improvement(base: dict[str, Any], cur: dict[str, Any]) -> dict[str, Any]:
    bm = base.get("binary_metrics", base)
    cm = cur.get("binary_metrics", cur)
    tx_b = base.get("text_similarity", {})
    tx_c = cur.get("text_similarity", {})
    return {
        "accuracy_delta": round(
            float(cm.get("accuracy", 0.0)) - float(bm.get("accuracy", 0.0)), 6
        ),
        "macro_f1_delta": round(
            float(cm.get("macro_f1", 0.0)) - float(bm.get("macro_f1", 0.0)), 6
        ),
        "auc_delta": round(float(cm.get("auc", 0.0)) - float(bm.get("auc", 0.0)), 6),
        "rouge_l_delta": round(
            float(tx_c.get("rouge_l", 0.0)) - float(tx_b.get("rouge_l", 0.0)), 6
        ),
        "bertscore_delta": round(
            float(tx_c.get("bertscore_f1", 0.0)) - float(tx_b.get("bertscore_f1", 0.0)), 6
        ),
        "ecva_vector_similarity_delta": round(
            float(tx_c.get("ecva_explanation_vector_cosine_mean", 0.0))
            - float(tx_b.get("ecva_explanation_vector_cosine_mean", 0.0)),
            6,
        ),
        "json_compliance_delta": round(
            float(cur.get("json_compliance_rate", 0.0)) - float(base.get("json_compliance_rate", 0.0)),
            6,
        ),
    }


def evaluate_one_model(
    *,
    project_root: Path,
    test_data: list,
    num_frames: int,
    model_cfg: dict,
    inference_cfg: dict,
    base_model_path: str,
    adapter_path: str | None,
    output_report: Path,
    predictions_jsonl: Path,
    predictions_json: Path,
    model_tag: str,
    batch_size: int,
    max_new_tokens: int,
    use_4bit: bool,
    bertscore_lang: str,
    no_resume: bool,
    persist_every_n: int,
    npz_stem: str,
    roc_stem: str,
    baseline_report_for_merge: Path | None = None,
) -> dict[str, Any]:
    """单模型评测：写 report、predictions jsonl/json、npz、ROC。返回 report dict。"""
    output_report.parent.mkdir(parents=True, exist_ok=True)

    if no_resume:
        for p in (predictions_jsonl, predictions_json):
            if p.exists():
                p.unlink()
        print(f"[{model_tag}] 已清理旧预测文件，从头评估")

    done_ids, all_records = _load_predictions_jsonl(predictions_jsonl, model_tag)
    if done_ids:
        print(f"[{model_tag}] 从 jsonl 恢复：已完成 {len(done_ids)} clips")

    print(f"[{model_tag}] 加载模型（4bit={use_4bit}, adapter={adapter_path is not None}）...")
    model, processor = _load_model_and_processor_eval(
        base_model_path,
        use_4bit=use_4bit,
        model_cfg=model_cfg,
        adapter_path=adapter_path,
    )
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None) or getattr(
        processor.tokenizer, "eos_token_id", None
    )

    valid_items: list[tuple[dict, bool, dict, list]] = []
    total_valid = 0
    for item in test_data:
        gt_anomaly, gt_dict = get_ground_truth(item)
        frames_dir = item.get("frames_dir", "")
        if not frames_dir:
            continue
        abs_frames = _resolve_frames_dir_eval(frames_dir, project_root)
        try:
            frames = load_frames_from_dir(abs_frames, num_frames)
        except Exception:
            continue
        total_valid += 1
        clip_id = item.get("clip_id", "")
        if clip_id in done_ids:
            continue
        valid_items.append((item, gt_anomaly, gt_dict, frames))

    if done_ids:
        print(f"[{model_tag}] 总有效 clips: {total_valid}，剩余待评估: {len(valid_items)}")

    oom_fallback = False
    n_skipped = 0
    run_start = time.perf_counter()
    n_since_flush = len(all_records) % persist_every_n

    def run_batch(batch_quads: list[tuple[dict, bool, dict, list]]) -> list[dict]:
        inputs_list: list[dict] = []
        for item, _gt_anomaly, _gt_dict, frames in batch_quads:
            user_content = [
                {"type": "video", "video": frames},
                {"type": "text", "text": EVAL_PROMPT},
            ]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            text += "{"
            try:
                inp = processor(
                    text=[text],
                    videos=[frames],
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                )
            except Exception:
                inp = processor(text=[text], return_tensors="pt", padding=False, truncation=True)
            inp.pop("mm_token_type_ids", None)
            inputs_list.append(inp)

        batched = _pad_and_batch(inputs_list)
        device = next(model.parameters()).device
        batched_gpu = {k: v.to(device) if hasattr(v, "to") else v for k, v in batched.items()}
        input_len = batched_gpu["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(
                **batched_gpu,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        generated = out[:, input_len:]

        results: list[dict] = []
        for i, (item, gt_anomaly, gt_dict, _frames) in enumerate(batch_quads):
            raw = processor.batch_decode(
                generated[i : i + 1],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            response = "{" + raw
            obj = extract_json_from_text(response)
            is_json_ok = obj is not None
            pred_anomaly = classification_is_abnormal(obj.get("classification")) if obj else False
            conf = 1.0 if pred_anomaly else 0.0
            gt = 1 if gt_anomaly else 0
            score = conf
            clip_id = item.get("clip_id", f"idx_{len(all_records) + len(results)}")
            pred_t = prediction_text_for_similarity(obj)
            ref_t = str(item.get("reference_description", "") or "").strip()
            if _should_skip_ecva_similarity(item, str(clip_id)):
                ecva_ref_t = ""
                ecva_similarity = None
            else:
                ecva_ref_t = ecva_text_for_similarity(item, gt_dict)
                ecva_similarity = _cosine_similarity_from_text(pred_t, ecva_ref_t)
            rec = {
                "clip_id": clip_id,
                "model_tag": model_tag,
                "frames_dir": str(item.get("frames_dir", "") or ""),
                "ground_truth_json": gt_dict,
                "ground_truth_is_anomaly": bool(gt_anomaly),
                "raw_response": response,
                "parsed_prediction_json": obj,
                "json_ok": is_json_ok,
                "predicted_is_anomaly": pred_anomaly,
                "confidence": conf,
                "y_true": gt,
                "y_score": score,
                "reference_text": ref_t,
                "reference_key_sentences": item.get("reference_key_sentences", []),
                "ecva_reference_text": ecva_ref_t,
                "prediction_text": pred_t,
                "explanation_ecva_similarity": ecva_similarity,
                "ref_text": ref_t,
                "pred_text": pred_t,
            }
            results.append(rec)
        return results

    for start in tqdm(range(0, len(valid_items), batch_size), desc=f"Eval-{model_tag}"):
        chunk = valid_items[start : start + batch_size]
        current_bs = 1 if oom_fallback else min(batch_size, len(chunk))
        batch_quads = chunk[:current_bs]

        try:
            batch_results = run_batch(batch_quads)
            _append_predictions_jsonl(predictions_jsonl, batch_results)
            all_records.extend(batch_results)
            n_since_flush += len(batch_results)
            if n_since_flush >= persist_every_n:
                _write_predictions_snapshot(predictions_json, model_tag, all_records)
                n_since_flush = 0
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current_bs > 1:
                tqdm.write("OOM (batch=%d)，自动降为 batch=1 重试" % current_bs)
                oom_fallback = True
                for single in batch_quads:
                    try:
                        single_results = run_batch([single])
                        _append_predictions_jsonl(predictions_jsonl, single_results)
                        all_records.extend(single_results)
                        n_since_flush += len(single_results)
                        if n_since_flush >= persist_every_n:
                            _write_predictions_snapshot(predictions_json, model_tag, all_records)
                            n_since_flush = 0
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        n_skipped += 1
                        tqdm.write("单条 OOM 跳过: %s" % single[0].get("clip_id", ""))
            else:
                torch.cuda.empty_cache()
                n_skipped += 1
                tqdm.write("单条 OOM 跳过: %s" % batch_quads[0][0].get("clip_id", ""))

    _write_predictions_snapshot(predictions_json, model_tag, all_records)

    elapsed = time.perf_counter() - run_start
    item_by_clip_id = {
        str(item.get("clip_id", "") or ""): item
        for item in test_data
        if str(item.get("clip_id", "") or "")
    }
    _enrich_prediction_records(all_records, item_by_clip_id)
    _write_predictions_snapshot(predictions_json, model_tag, all_records)

    y_true = [int(r["y_true"]) for r in all_records]
    y_score = [float(r["y_score"]) for r in all_records]
    json_ok = sum(1 for r in all_records if r.get("json_ok"))
    ref_texts_all = [str(r.get("reference_text", "")) for r in all_records]
    pred_texts_all = [str(r.get("prediction_text", "")) for r in all_records]
    ecva_sims = [
        float(sim)
        for sim in (r.get("explanation_ecva_similarity") for r in all_records)
        if sim is not None
    ]

    y_true_arr = np.array(y_true, dtype=np.int64)
    y_score_arr = np.array(y_score, dtype=np.float64)
    n = len(y_true_arr)
    json_compliance = json_ok / n if n else 0.0
    ecva_similarity_mean = float(sum(ecva_sims) / len(ecva_sims)) if ecva_sims else 0.0

    from sklearn.metrics import (
        roc_auc_score,
        accuracy_score,
        roc_curve,
        precision_recall_fscore_support,
    )

    try:
        auc = float(roc_auc_score(y_true_arr, y_score_arr))
    except Exception:
        auc = 0.0
    pred_binary = (y_score_arr >= 0.5).astype(np.int64)
    acc = float(accuracy_score(y_true_arr, pred_binary))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_arr,
        pred_binary,
        average="macro",
        zero_division=0,
    )
    prec_bin, rec_bin, f1_bin, _ = precision_recall_fscore_support(
        y_true_arr,
        pred_binary,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    refs_sub: list[str] = []
    preds_sub: list[str] = []
    for r, p in zip(ref_texts_all, pred_texts_all):
        if (r or "").strip():
            refs_sub.append(r.strip())
            preds_sub.append((p or "").strip())
    rouge_l = _mean_rouge_l(refs_sub, preds_sub)
    bert_f1 = _mean_bertscore_f1(refs_sub, preds_sub, lang=bertscore_lang)
    fpr, tpr, thresholds = roc_curve(y_true_arr, y_score_arr)
    idx = int(np.argmin(np.abs(tpr - 0.95)))
    fpr_at_95 = float(fpr[idx]) if idx is not None else 0.0

    tpr_levels = [0.5, 0.7, 0.8, 0.9, 0.95]
    fpr_at_tpr_levels: dict[str, float] = {}
    for t in tpr_levels:
        i = int(np.argmin(np.abs(tpr - t)))
        fpr_at_tpr_levels[f"fpr_at_tpr_{t}"] = float(fpr[i])

    report = {
        "n_samples": int(n),
        "n_skipped": n_skipped,
        "binary_metrics": {
            "accuracy": acc,
            "auc": auc,
            "macro_precision": float(prec_macro),
            "macro_recall": float(rec_macro),
            "macro_f1": float(f1_macro),
            "precision": float(prec_bin),
            "recall": float(rec_bin),
            "f1": float(f1_bin),
        },
        "text_similarity": {
            "rouge_l": rouge_l,
            "bertscore_f1": bert_f1,
            "n_with_reference": len(refs_sub),
            "ecva_explanation_vector_cosine_mean": ecva_similarity_mean,
            "n_with_ecva_text": len(ecva_sims),
        },
        "auc": auc,
        "accuracy": acc,
        "json_compliance_rate": json_compliance,
        "fpr_at_95_tpr": fpr_at_95,
        "fpr_at_tpr_levels": fpr_at_tpr_levels,
        "run_seconds": round(elapsed, 2),
        "clips_per_sec": round(n / elapsed, 4) if elapsed > 0 else 0,
        "run_config": {
            "model_tag": model_tag,
            "base_model_path": base_model_path,
            "adapter_path": adapter_path,
            "use_4bit": use_4bit,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "bertscore_lang": bertscore_lang,
            "predictions_jsonl": str(predictions_jsonl),
            "predictions_json": str(predictions_json),
        },
    }

    base_rep_path = baseline_report_for_merge
    if base_rep_path and base_rep_path.is_file():
        try:
            with base_rep_path.open("r", encoding="utf-8") as f:
                base = json.load(f)
            bm = base.get("binary_metrics", base)
            tx = base.get("text_similarity", {})
            report["baseline_text_similarity"] = {
                "rouge_l": tx.get("rouge_l", 0.0),
                "bertscore_f1": tx.get("bertscore_f1", 0.0),
                "ecva_explanation_vector_cosine_mean": tx.get(
                    "ecva_explanation_vector_cosine_mean", 0.0
                ),
            }
            report["improvement_vs_baseline_report_file"] = {
                "accuracy_delta": round(
                    acc - float(bm.get("accuracy", 0.0)), 6
                ),
                "macro_f1_delta": round(
                    float(f1_macro) - float(bm.get("macro_f1", 0.0)), 6
                ),
                "rouge_l_delta": round(
                    rouge_l - float(tx.get("rouge_l", 0.0)), 6
                ),
                "bertscore_delta": round(
                    bert_f1 - float(tx.get("bertscore_f1", 0.0)), 6
                ),
                "ecva_vector_similarity_delta": round(
                    ecva_similarity_mean
                    - float(tx.get("ecva_explanation_vector_cosine_mean", 0.0)),
                    6,
                ),
            }
        except Exception as e:
            report["baseline_merge_error"] = str(e)

    with output_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[{model_tag}] 评估报告已写入:", output_report)

    out_dir = output_report.parent
    np.savez(
        out_dir / f"{npz_stem}.npz",
        y_true=y_true_arr,
        y_score=y_score_arr,
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
    )
    print(f"[{model_tag}] 已保存: {npz_stem}.npz")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="随机")
        ax.set_xlabel("FPR (假阳性率)")
        ax.set_ylabel("TPR (召回率)")
        ax.set_title("ROC 曲线")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        roc_path = out_dir / f"{roc_stem}.png"
        fig.savefig(roc_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[{model_tag}] 已保存 ROC: {roc_path.name}")
    except ImportError:
        print(f"[{model_tag}] 未安装 matplotlib，跳过 ROC")

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="评估 Qwen3.5-9B（支持双模型一次跑完）")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--model_path", type=str, default="outputs/qlora/final")
    ap.add_argument("--test_json", type=str, default="")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--output_report", type=str, default="outputs/eval_report.json")
    ap.add_argument("--batch_size", type=int, default=10, help="推理 batch 大小，OOM 时自动降为 1")
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--base_model_path", type=str, default="", help="基座模型路径（adapter 时默认从 adapter_config 读取）")
    ap.add_argument("--no_resume", action="store_true", help="忽略预测 jsonl/json 断点，从头评估")
    ap.add_argument(
        "--dual_eval",
        action="store_true",
        help="若 model_path 为 LoRA 目录：先评原始基座，再评微调 adapter，并写汇总 eval_report",
    )
    ap.add_argument(
        "--predictions_flush_every",
        type=int,
        default=10,
        help="每处理多少个 clip 将 predictions 完整快照写入 .json（jsonl 每条即追加）",
    )
    ap.add_argument("--use_4bit", action="store_true", default=None, help="启用 4bit（默认从 config）")
    ap.add_argument("--no_4bit", action="store_true", help="禁用 4bit")
    ap.add_argument(
        "--baseline_report",
        type=str,
        default="",
        help="单模型模式：可选，读入另一份报告 JSON 计算 improvement（旧行为保留）",
    )
    ap.add_argument("--bertscore_lang", type=str, default="en")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / args.config
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    inference_cfg = cfg.get("inference", {})
    clips_cfg = cfg.get("clips", {})
    num_frames = int(clips_cfg.get("clip_len", 16))

    def _argv_has_opt(flag: str) -> bool:
        return any(x == flag or x.startswith(flag + "=") for x in sys.argv[1:])

    test_json = args.test_json or data_cfg.get("test_json", "data/processed/test.json")
    test_path = project_root / test_json if not Path(test_json).is_absolute() else Path(test_json)
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = project_root / model_path
    output_report = Path(args.output_report)
    if not output_report.is_absolute():
        output_report = project_root / output_report
    output_report.parent.mkdir(parents=True, exist_ok=True)

    use_4bit = False if args.no_4bit else (args.use_4bit if args.use_4bit is not None else model_cfg.get("use_4bit", True))
    if _argv_has_opt("--batch_size"):
        batch_size = max(1, args.batch_size)
    else:
        batch_size = max(1, int(inference_cfg.get("eval_batch_size", args.batch_size)))
    max_new_tokens = args.max_new_tokens or inference_cfg.get("max_tokens", 96)
    max_new_tokens = min(int(max_new_tokens), 512)

    with test_path.open("r", encoding="utf-8") as f:
        test_data = json.load(f)
    if args.max_samples > 0:
        test_data = test_data[: args.max_samples]

    adapter_path = None
    base_model_path_cli = args.base_model_path
    adapter_config_file = model_path / "adapter_config.json"
    if adapter_config_file.exists():
        adapter_path = str(model_path)
        if not base_model_path_cli:
            with adapter_config_file.open("r", encoding="utf-8") as f:
                adapter_cfg = json.load(f)
            base_model_path = adapter_cfg.get("base_model_name_or_path", "")
        else:
            base_model_path = base_model_path_cli
        if not base_model_path:
            base_model_path = model_cfg.get("name_or_path", "")
        if not base_model_path:
            raise ValueError("无法确定基座模型路径，请用 --base_model_path 指定")
        print(f"检测到 LoRA adapter 目录: {adapter_path}")
        print(f"基座模型路径: {base_model_path}")
    else:
        base_model_path = str(model_path)
        if base_model_path_cli:
            base_model_path = base_model_path_cli

    out_dir = output_report.parent
    persist_n = max(1, int(args.predictions_flush_every))

    if args.dual_eval:
        if not adapter_path:
            raise ValueError("--dual_eval 需要 --model_path 指向含 adapter_config.json 的目录（如 outputs/qlora/final）")
        summary_path = output_report
        baseline_report_path = out_dir / "baseline_report.json"
        finetuned_report_path = out_dir / "finetuned_report.json"
        baseline_jsonl = out_dir / "baseline_predictions.jsonl"
        baseline_json = out_dir / "baseline_predictions.json"
        ft_jsonl = out_dir / "finetuned_predictions.jsonl"
        ft_json = out_dir / "finetuned_predictions.json"

        if args.no_resume:
            for p in (
                baseline_jsonl,
                baseline_json,
                ft_jsonl,
                ft_json,
                baseline_report_path,
                finetuned_report_path,
            ):
                if p.exists():
                    p.unlink()
            if summary_path.exists():
                summary_path.unlink()
            print("[dual] 已清理 baseline/finetuned 预测与报告（--no_resume）")

        _ = evaluate_one_model(
            project_root=project_root,
            test_data=test_data,
            num_frames=num_frames,
            model_cfg=model_cfg,
            inference_cfg=inference_cfg,
            base_model_path=base_model_path,
            adapter_path=None,
            output_report=baseline_report_path,
            predictions_jsonl=baseline_jsonl,
            predictions_json=baseline_json,
            model_tag="baseline",
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            use_4bit=use_4bit,
            bertscore_lang=args.bertscore_lang,
            no_resume=False,
            persist_every_n=persist_n,
            npz_stem="baseline_eval_data",
            roc_stem="baseline_eval_roc",
            baseline_report_for_merge=None,
        )

        _ = evaluate_one_model(
            project_root=project_root,
            test_data=test_data,
            num_frames=num_frames,
            model_cfg=model_cfg,
            inference_cfg=inference_cfg,
            base_model_path=base_model_path,
            adapter_path=adapter_path,
            output_report=finetuned_report_path,
            predictions_jsonl=ft_jsonl,
            predictions_json=ft_json,
            model_tag="finetuned",
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            use_4bit=use_4bit,
            bertscore_lang=args.bertscore_lang,
            no_resume=False,
            persist_every_n=persist_n,
            npz_stem="finetuned_eval_data",
            roc_stem="finetuned_eval_roc",
            baseline_report_for_merge=None,
        )

        with baseline_report_path.open("r", encoding="utf-8") as f:
            rep_b = json.load(f)
        with finetuned_report_path.open("r", encoding="utf-8") as f:
            rep_f = json.load(f)

        summary: dict[str, Any] = {
            "baseline": rep_b,
            "finetuned": rep_f,
            "improvement": _reports_to_improvement(rep_b, rep_f),
            "paths": {
                "baseline_report": str(baseline_report_path),
                "finetuned_report": str(finetuned_report_path),
                "baseline_predictions_jsonl": str(baseline_jsonl),
                "baseline_predictions_json": str(baseline_json),
                "finetuned_predictions_jsonl": str(ft_jsonl),
                "finetuned_predictions_json": str(ft_json),
            },
        }
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("[dual] 汇总已写入:", summary_path)
        return

    predictions_jsonl = out_dir / f"{output_report.stem}_predictions.jsonl"
    predictions_json = out_dir / f"{output_report.stem}_predictions.json"
    stem = output_report.stem

    bl_merge = Path(args.baseline_report) if args.baseline_report else None
    if bl_merge and not bl_merge.is_absolute():
        bl_merge = project_root / bl_merge

    evaluate_one_model(
        project_root=project_root,
        test_data=test_data,
        num_frames=num_frames,
        model_cfg=model_cfg,
        inference_cfg=inference_cfg,
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        output_report=output_report,
        predictions_jsonl=predictions_jsonl,
        predictions_json=predictions_json,
        model_tag="single",
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        use_4bit=use_4bit,
        bertscore_lang=args.bertscore_lang,
        no_resume=args.no_resume,
        persist_every_n=persist_n,
        npz_stem="eval_data",
        roc_stem="eval_roc",
        baseline_report_for_merge=bl_merge,
    )


if __name__ == "__main__":
    main()
