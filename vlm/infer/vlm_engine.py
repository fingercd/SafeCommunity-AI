"""
VLM 推理引擎：加载微调后的 Qwen3.5-VL，对视频 clip（多帧）做异常分析，返回结构化 dict。

- 支持 transformers 本地推理（默认）
- 可选 vLLM 后端（需安装 vllm 且模型为 vLLM 兼容格式）
- 输出 JSON：classification, reason（中文）
- 为兼容旧调用：同时填充 is_anomaly、anomaly_type、confidence、reasoning 派生字段
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from vlm.utils import extract_json_from_text, load_transformers_engine

SYSTEM_PROMPT = (
    "你是监控视频分析专家。分析视频后，只输出一行纯JSON，禁止任何解释、markdown、换行。"
    "JSON字段：classification（正常或异常类型如打架/火灾/盗窃）、reason（30字内中文说明）。"
    "画面正常时classification写正常，异常时写具体异常类型。"
)

USER_PROMPT = (
    '分析这段监控视频，只输出一行JSON，不要任何其他文字：\n'
    '示例1: {"classification":"正常","reason":"画面平静，人员活动正常"}\n'
    '示例2: {"classification":"打架","reason":"两名男子互相殴打推搡"}\n'
    '示例3: {"classification":"火灾","reason":"画面中可见明显火焰和浓烟"}\n'
    '现在分析这段视频，只输出JSON：'
)


def _normalize_engine_output(obj: dict[str, Any] | None, raw: str, *, error: bool) -> dict[str, Any]:
    """补全 ECVA 字段并生成 is_anomaly 等兼容字段。"""
    if not obj:
        # 解析完全失败时的兜底：返回中文提示，不暴露原始乱码给用户
        obj = {
            "classification": "正常",
            "reason": "模型未按格式输出，请检查模型状态或重试。",
            "result": "Uncertain.",
            "description": "",
            "key_sentences": [],
        }
    cls = str(obj.get("classification") or "normal").strip()
    is_abnormal = cls.lower() not in ("normal", "non-anomaly", "", "正常")
    out = dict(obj)
    out["is_anomaly"] = is_abnormal
    out["anomaly_type"] = "normal" if not is_abnormal else cls
    out["confidence"] = 1.0 if is_abnormal else 0.0
    out["reasoning"] = str(out.get("reason") or "")
    out["error"] = error
    return out


def _detect_model_type(model_path: Path) -> str:
    """检测模型是 Base 还是 Instruct。"""
    readme = model_path / "README.md"
    if readme.exists():
        content = readme.read_text(encoding="utf-8", errors="ignore")
        # Base 模型的 README 会有 base_model: 字段
        if "base_model:" in content and "-Base" in content:
            return "base"
        # Instruct 模型通常会在标题或标签中体现
        if "instruct" in content.lower() or "chat" in content.lower():
            return "instruct"
    # 备用检测：看目录名
    name = model_path.name.lower()
    if "instruct" in name or "chat" in name:
        return "instruct"
    if "base" in name:
        return "base"
    return "unknown"


class VLMEngine:
    """封装 Qwen3.5 推理，提供 analyze_clip(frames)。"""

    def __init__(self, model_path: str | Path, use_vllm: bool = False):
        self.model_path = Path(model_path)
        self.use_vllm = use_vllm
        self._model = None
        self._processor = None
        self._vllm_llm = None
        self.model_type = _detect_model_type(self.model_path)
        if self.model_type == "base":
            import logging
            logging.getLogger("vlm.engine").warning(
                "检测到基础模型(Base)：%s。Base模型不支持指令遵循，请换成Instruct版本。"
                "推荐：Qwen3.5-VL-Instruct 或 Qwen2.5-VL-Instruct", self.model_path
            )

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
        self._model, self._processor = load_transformers_engine(str(self.model_path))

    def analyze_clip(self, frames: list, max_new_tokens: int = 128) -> dict[str, Any]:
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

    def _analyze_transformers(self, pil_frames: list, max_new_tokens: int = 128) -> dict[str, Any]:
        import re

        # Qwen3.5 支持 system + user 的 chat format
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "video", "video": pil_frames}, {"type": "text", "text": USER_PROMPT}]},
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            inputs = self._processor(text=[text], videos=[pil_frames], return_tensors="pt", padding=True)
        except Exception:
            inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self._model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # 记录输入长度，用于只解码生成的新 token
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._processor.tokenizer.pad_token_id,
                eos_token_id=self._processor.tokenizer.eos_token_id,
            )

        # 只解码模型新生成的 token（跳过输入 prompt）
        new_tokens = out[:, input_len:]
        response = self._processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # Qwen3.5 可能输出 <think>...</think>，去掉它
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        # 清理模型可能输出的 markdown 代码块标记
        response = response.strip().lstrip("`").rstrip("`")
        if response.lower().startswith("json"):
            response = response[4:].strip()

        obj = extract_json_from_text(response)
        if obj is None:
            # 解析失败时把原始输出透传给前端，方便调试 Prompt
            out = _normalize_engine_output(
                {
                    "classification": "正常",
                    "reason": response.strip() or "模型未按格式输出，请检查模型状态或重试。",
                },
                response,
                error=True,
            )
            out["raw"] = response[:500]
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
