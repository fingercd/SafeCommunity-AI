"""
HuggingFace VideoMAE ViT 视频编码器封装。

- RGB 分支：AutoImageProcessor + AutoModel，输出 768 维 embedding
- 可选双流：RGB + 光流（FlowCNNEncoder），concat/add/mlp 融合 → 1536 维或 1024 维
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class HfVideoEncoderConfig:
    """ViT 编码器配置：模型名、图像尺寸、FP16、池化方式、双流/融合等"""
    model_name: str = "MCG-NJU/videomae-base"  # 需要首次联网下载
    image_size: int = 224
    use_half: bool = True
    pooling: str = "auto"  # auto/pooler/cls/mean

    # 双流：RGB + 光流
    use_dual_stream: bool = False
    flow_branch_type: str = "cnn"  # "cnn" | "vit"
    flow_embedding_dim: int = 768
    fusion_method: str = "concat"  # "concat" | "add" | "mlp"


class FlowCNNEncoder(nn.Module):
    """
    光流 3D CNN 分支：输入 (B, T, 2, H, W)，输出 (B, D)。
    轻量级 3D Conv + 全局池化 + Linear，不依赖 torchvision 内部结构。
    """

    def __init__(self, flow_embedding_dim: int = 768, flow_input_size: int = 224):
        super().__init__()
        self.flow_input_size = flow_input_size
        self._out_dim = flow_embedding_dim
        self.conv = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(256, flow_embedding_dim)

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(0)
        if x.shape[1] == 2 and x.shape[2] != 2:
            pass
        else:
            if x.size(2) == 2:
                x = x.permute(0, 2, 1, 3, 4)
        if x.size(-2) != self.flow_input_size or x.size(-1) != self.flow_input_size:
            x = torch.nn.functional.interpolate(
                x, size=(x.size(2), self.flow_input_size, self.flow_input_size), mode="trilinear", align_corners=False
            )
        x = self.conv(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class HfVideoEncoder(nn.Module):
    """
    HuggingFace 视频编码器封装。

    输入：list[list[np.ndarray(H,W,3) uint8 RGB]] 或 torch tensor (B,T,C,H,W) float
    输出：embedding (B, D)
    """

    def __init__(self, cfg: HfVideoEncoderConfig, device: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_half = bool(cfg.use_half and self.device_str.startswith("cuda"))

        try:
            from transformers import AutoImageProcessor, AutoModel
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "缺少 transformers 依赖。请安装：pip install transformers"
            ) from e

        self.processor = AutoImageProcessor.from_pretrained(cfg.model_name)
        self.model = AutoModel.from_pretrained(cfg.model_name)
        self.model.eval()
        self.model.to(self.device_str)
        if self.use_half:
            self.model.half()

        rgb_dim = getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "dim", None)
        self.embedding_dim = rgb_dim

        if getattr(cfg, "use_dual_stream", False):
            self.flow_encoder = FlowCNNEncoder(
                flow_embedding_dim=getattr(cfg, "flow_embedding_dim", 768),
                flow_input_size=getattr(cfg, "image_size", 224),
            )
            self.flow_encoder.to(self.device_str)
            self.flow_encoder.eval()
            flow_dim = self.flow_encoder.out_dim
            if getattr(cfg, "fusion_method", "concat") == "concat":
                self.embedding_dim = (rgb_dim or 768) + flow_dim
            elif getattr(cfg, "fusion_method", "") == "mlp":
                self.fusion_proj = nn.Linear((rgb_dim or 768) + flow_dim, 1024)
                self.fusion_proj.to(self.device_str)
                self.embedding_dim = 1024
            else:
                self.embedding_dim = rgb_dim or 768
        else:
            self.flow_encoder = None
            self.fusion_proj = None

    def _forward_rgb_branch(self, clips: Any) -> torch.Tensor:
        """RGB 分支：processor → model → pooling（pooler/cls/mean）"""
        if isinstance(clips, torch.Tensor):
            pixel_values = clips.to(self.device_str)
        else:
            try:
                inputs = self.processor(clips, return_tensors="pt")
            except TypeError:
                inputs = self.processor(videos=clips, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device_str)
        if self.use_half:
            pixel_values = pixel_values.half()
        outputs = self.model(pixel_values=pixel_values)
        pooling = (self.cfg.pooling or "auto").lower().strip()
        if pooling in {"auto", "pooler"} and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output.float()
        if pooling in {"auto", "cls"} and hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            lhs = outputs.last_hidden_state
            if lhs.dim() == 3:
                return lhs[:, 0].float()
            return lhs.float()
        if pooling in {"auto", "mean"}:
            for v in outputs.__dict__.values():
                if isinstance(v, torch.Tensor):
                    return v.mean(dim=tuple(range(1, v.dim()))).float()
            raise RuntimeError("cannot extract embedding from HF model outputs")
        raise ValueError(f"unknown pooling: {self.cfg.pooling!r}")

    def _forward_flow_branch(self, clips_flow: Any) -> torch.Tensor:
        """光流分支：3D CNN → 768 维"""
        if isinstance(clips_flow, torch.Tensor):
            x = clips_flow.to(self.device_str)
        else:
            tensors = []
            for item in clips_flow:
                t = torch.from_numpy(np.asarray(item, dtype=np.float32)).to(self.device_str)
                if t.dim() == 3:
                    t = t.unsqueeze(0)
                tensors.append(t)
            x = torch.stack(tensors, dim=0)
        if x.size(2) == 2:
            x = x.permute(0, 2, 1, 3, 4)
        return self.flow_encoder(x).float()

    @torch.no_grad()
    def forward(self, clips: Any, clips_flow: Optional[Any] = None) -> torch.Tensor:
        """
        clips: list of RGB clips or tensor (B,T,C,H,W)
        clips_flow: optional list of (T-1,2,H,W) per clip or tensor (B,T-1,2,H,W)
        """
        emb_rgb = self._forward_rgb_branch(clips)
        if not getattr(self.cfg, "use_dual_stream", False) or self.flow_encoder is None or clips_flow is None:
            return emb_rgb
        emb_flow = self._forward_flow_branch(clips_flow)
        fusion = getattr(self.cfg, "fusion_method", "concat")
        if fusion == "concat":
            return torch.cat([emb_rgb, emb_flow], dim=-1)
        if fusion == "add":
            return (emb_rgb + emb_flow)
        if fusion == "mlp" and self.fusion_proj is not None:
            return self.fusion_proj(torch.cat([emb_rgb, emb_flow], dim=-1))
        return torch.cat([emb_rgb, emb_flow], dim=-1)


def freeze_module(m: nn.Module) -> None:
    """冻结模块参数（requires_grad=False）"""
    for p in m.parameters():
        p.requires_grad = False

