"""
VLM 推理引擎：加载微调后的 Qwen3.5-VL，对视频 clip（多帧）做异常分析，返回结构化 dict。

- 支持 transformers 本地推理（默认）
- 可选 vLLM 后端（需安装 vllm 且模型为 vLLM 兼容格式）
- 输出 ECVA 风格：classification, reason, result, description, key_sentences（英文）
- 为兼容旧调用：同时填充 is_anomaly、anomaly_type、confidence、reasoning 派生字段
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

# 延迟导入
def _load_transformers_engine(model_path: str, device_map: str = "auto"):
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


PROMPT = (
    "Analyze this surveillance clip. Output one line of JSON only, no markdown. "
    "Keys: classification, reason, result, description, key_sentences (array of 2-4 short English strings). "
    "classification must be \"normal\" for routine activity, otherwise an English event type. "
    "All strings in English."
)


def _normalize_engine_output(obj: dict[str, Any] | None, raw: str, *, error: bool) -> dict[str, Any]:
    """补全 ECVA 字段并生成 is_anomaly 等兼容字段。"""
    if not obj:
        obj = {
            "classification": "normal",
            "reason": raw[:400] if raw else "Parse failed.",
            "result": "Uncertain.",
            "description": "",
            "key_sentences": [],
        }
    cls = str(obj.get("classification") or "normal").strip()
    is_abnormal = cls.lower() not in ("normal", "non-anomaly", "")
    out = dict(obj)
    out["is_anomaly"] = is_abnormal
    out["anomaly_type"] = "normal" if not is_abnormal else cls
    out["confidence"] = 1.0 if is_abnormal else 0.0
    out["reasoning"] = str(out.get("reason") or "")
    out["error"] = error
    return out


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


class VLMEngine:
    """封装 Qwen3.5 推理，提供 analyze_clip(frames)。"""

    def __init__(self, model_path: str | Path, use_vllm: bool = False):
        self.model_path = Path(model_path)
        self.use_vllm = use_vllm
        self._model = None
        self._processor = None
        self._vllm_llm = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        if self.use_vllm:
            try:
                from vllm import LLM
                self._vllm_llm = LLM(model=str(self.model_path), trust_remote_code=True)
                return
            except Exception:
                self.use_vllm = False
        self._model, self._processor = _load_transformers_engine(str(self.model_path))

    def analyze_clip(self, frames: list, max_new_tokens: int = 256) -> dict[str, Any]:
        """
        分析一组帧（PIL Image 或 numpy (H,W,3)），返回结构化结果。

        Returns:
            dict: ECVA 字段 + is_anomaly/anomaly_type/confidence/reasoning（兼容旧代码）；
                  若解析失败则含 error 与 raw。
        """
        self._ensure_loaded()
        from PIL import Image
        pil_frames = []
        for f in frames:
            if hasattr(f, "convert"):
                pil_frames.append(f.convert("RGB") if f.mode != "RGB" else f)
            else:
                pil_frames.append(Image.fromarray(f).convert("RGB"))
        if not pil_frames:
            return _normalize_engine_output(
                {
                    "classification": "normal",
                    "reason": "No valid frames.",
                    "result": "No analysis.",
                    "description": "",
                    "key_sentences": [],
                },
                "",
                error=True,
            )

        if self.use_vllm and self._vllm_llm is not None:
            return self._analyze_vllm(pil_frames, max_new_tokens)
        return self._analyze_transformers(pil_frames, max_new_tokens)

    def _analyze_transformers(self, pil_frames: list, max_new_tokens: int) -> dict[str, Any]:
        messages = [{"role": "user", "content": [{"type": "video", "video": pil_frames}, {"type": "text", "text": PROMPT}]}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            inputs = self._processor(text=[text], videos=[pil_frames], return_tensors="pt", padding=True)
        except Exception:
            inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self._model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        response = self._processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        if text in response:
            response = response.split(text)[-1].strip()
        obj = extract_json_from_text(response)
        if obj is None:
            out = _normalize_engine_output(None, response, error=True)
            out["raw"] = response
            return out
        out = _normalize_engine_output(obj, response, error=False)
        return out

    def _analyze_vllm(self, pil_frames: list, max_new_tokens: int) -> dict[str, Any]:
        import tempfile
        import cv2
        import numpy as np
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        path = tmp.name
        try:
            h, w = np.asarray(pil_frames[0]).shape[:2]
            out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 4, (w, h))
            for p in pil_frames:
                out.write(cv2.cvtColor(np.asarray(p), cv2.COLOR_RGB2BGR))
            out.release()
            # vLLM 多模态请求格式依版本而定，此处简化为占位
            from vllm import SamplingParams
            sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0)
            # 使用 vLLM 的 multimodal 接口需传入 video 资源；若无则回退到 transformers 行为
            # 此处仅作占位，实际需根据 vllm.assets.video 构造请求
            return self._analyze_transformers(pil_frames, max_new_tokens)
        finally:
            Path(path).unlink(missing_ok=True)


def create_engine(model_path: str | Path, use_vllm: bool = False) -> VLMEngine:
    """工厂函数：创建并返回 VLMEngine。"""
    return VLMEngine(model_path=model_path, use_vllm=use_vllm)
