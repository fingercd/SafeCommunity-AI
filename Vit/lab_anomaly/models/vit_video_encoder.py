"""
VideoMAE v2（HuggingFace OpenGVLab）视频编码器：训练时可反传，支持冻结与按层解冻。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class VideoMAEv2EncoderConfig:
    """VideoMAE v2 配置"""
    model_name: str = "OpenGVLab/VideoMAEv2-Base"
    image_size: int = 224
    num_frames: int = 16
    # 与训练里 encoder_use_half 对齐：表示「在 CUDA 上是否配合 torch.amp」，不再把骨干权重 .half()
    use_half: bool = True
    pooling: str = "auto"  # auto / cls / mean / pooler


class VideoMAEv2Encoder(nn.Module):
    """
    VideoMAE v2 封装。输入可为：
    - pixel_values: (B, C, T, H, W) float，已由 processor 处理好；或
    - list[list[np.ndarray]]：每个样本一段视频，长度为 T 的 RGB HWC uint8 帧列表
    输出：(B, D) embedding。
    """

    def __init__(self, cfg: VideoMAEv2EncoderConfig, device: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_half = bool(cfg.use_half and self.device_str.startswith("cuda"))

        try:
            from transformers import AutoConfig, AutoModel, AutoImageProcessor
        except Exception as e:  # pragma: no cover
            raise RuntimeError("缺少 transformers。请安装：pip install transformers") from e

        config = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=True)
        try:
            from transformers import VideoMAEImageProcessor

            self.processor = VideoMAEImageProcessor.from_pretrained(cfg.model_name, trust_remote_code=True)
        except Exception:
            self.processor = AutoImageProcessor.from_pretrained(cfg.model_name, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(
            cfg.model_name,
            config=config,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
        )
        self.backbone.to(self.device_str)
        # 权重始终保持 FP32；半精度由训练端的 torch.amp.autocast 完成，避免与 GradScaler 冲突（FP16 梯度 unscale 报错）。

        hid = getattr(config, "hidden_size", None) or getattr(config, "embed_dim", None) or 768
        self.embedding_dim = int(hid)

    def _get_encoder_layers(self) -> Optional[nn.ModuleList]:
        def _get_by_path(root: nn.Module, path: tuple[str, ...]) -> Optional[nn.Module]:
            cur: Any = root
            for name in path:
                if not hasattr(cur, name):
                    return None
                cur = getattr(cur, name)
            return cur if isinstance(cur, nn.Module) else None

        def _looks_like_blocks(mod: Optional[nn.Module]) -> Optional[nn.ModuleList]:
            if not isinstance(mod, nn.ModuleList) or len(mod) == 0:
                return None
            first = mod[0]
            if any(hasattr(first, name) for name in ("attn", "mlp", "norm1", "norm2")):
                return mod
            return None

        m = self.backbone
        for path in (
            ("videomae", "encoder", "layer"),
            ("encoder", "layer"),
            ("model", "blocks"),
            ("blocks",),
        ):
            layers = _looks_like_blocks(_get_by_path(m, path))
            if layers is not None:
                return layers

        # 兜底：递归找最像 Transformer block 列表的 ModuleList。
        best: Optional[nn.ModuleList] = None
        best_len = -1
        for _, mod in m.named_modules():
            layers = _looks_like_blocks(mod)
            if layers is not None and len(layers) > best_len:
                best = layers
                best_len = len(layers)
        return best

    def freeze_backbone(self) -> None:
        """冻结整个 backbone 参数。"""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int) -> None:
        """
        先全部冻结，再解冻最后 n 个 Transformer block（以及 layernorm 等常见附加层可按需扩展）。
        n<=0 等价于全部冻结。
        """
        self.freeze_backbone()
        layers = self._get_encoder_layers()
        if layers is None or n <= 0:
            return
        n = min(int(n), len(layers))
        for i in range(len(layers) - n, len(layers)):
            for p in layers[i].parameters():
                p.requires_grad = True

    def trainable_param_count(self) -> int:
        """当前 requires_grad=True 的参数数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_trainable_param_count(self) -> int:
        return self.trainable_param_count()

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        if hasattr(self.backbone, "gradient_checkpointing_enable") and enabled:
            self.backbone.gradient_checkpointing_enable()
        elif hasattr(self.backbone, "gradient_checkpointing_disable") and not enabled:
            self.backbone.gradient_checkpointing_disable()

    def _pool(self, outputs: Any) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            t = outputs.float()
            if t.dim() == 1:
                return t.unsqueeze(0)
            if t.dim() == 2:
                return t
            return t.mean(dim=tuple(range(1, t.dim())))

        pooling = (self.cfg.pooling or "auto").lower().strip()
        if pooling in {"auto", "pooler"} and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output.float()
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            lhs = outputs.last_hidden_state
            if lhs.dim() == 3:
                if pooling in {"auto", "cls"}:
                    return lhs[:, 0].float()
                return lhs.mean(dim=1).float()
            return lhs.float().mean(dim=tuple(range(1, lhs.dim())))
        if hasattr(outputs, "__dict__"):
            for v in outputs.__dict__.values():
                if isinstance(v, torch.Tensor):
                    return v.mean(dim=tuple(range(1, v.dim()))).float()
        raise RuntimeError("无法从模型输出中取出 embedding")

    def _tensor_from_rgb_lists(self, clips: Sequence[Sequence[np.ndarray]]) -> torch.Tensor:
        """list[list[HWC uint8]] -> processor -> (B,C,T,H,W) 与骨干期望一致。"""
        proc_list = []
        for clip in clips:
            frames = list(clip)
            if len(frames) != self.cfg.num_frames:
                raise ValueError(
                    f"每段需 {self.cfg.num_frames} 帧，当前 {len(frames)}"
                )
            proc_list.append([np.asarray(f, dtype=np.uint8) for f in frames])

        try:
            inputs = self.processor(proc_list, return_tensors="pt")
        except TypeError:
            inputs = self.processor(videos=proc_list, return_tensors="pt")

        pv = inputs["pixel_values"]
        if pv.dim() == 5 and pv.shape[1] != 3 and pv.shape[2] == 3:
            # (B, T, C, H, W) -> (B, C, T, H, W)
            pv = pv.permute(0, 2, 1, 3, 4)
        return pv

    def forward(self, clips: Any) -> torch.Tensor:
        """
        clips: torch.Tensor (B,C,T,H,W) 或 list[list[np.ndarray]] RGB uint8
        返回 (B, D) float32
        """
        if isinstance(clips, torch.Tensor):
            pixel_values = clips.to(self.device_str)
        else:
            pixel_values = self._tensor_from_rgb_lists(clips).to(self.device_str)

        # 与骨干权重 dtype 一致；CUDA 上半精度由外层 autocast 负责，勿手动 .half() 以免产生 FP16 梯度。
        pixel_values = pixel_values.float()

        outputs = self.backbone(pixel_values=pixel_values)
        return self._pool(outputs)
