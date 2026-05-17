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

        # Patch for transformers 5.x: older remote code models may lack
        # `all_tied_weights_keys` which 5.x expects during _finalize_model_loading.
        try:
            auto_map = getattr(config, "auto_map", None) or {}
            model_cls_path = auto_map.get("AutoModel")
            if model_cls_path:
                from transformers.dynamic_module_utils import get_class_from_dynamic_module
                cls = get_class_from_dynamic_module(model_cls_path, cfg.model_name)
                if not hasattr(cls, "all_tied_weights_keys"):
                    @property
                    def _all_tied_weights_keys(self):
                        tied = getattr(self, "_tied_weights_keys", {})
                        return tied if tied is not None else {}
                    cls.all_tied_weights_keys = _all_tied_weights_keys
        except Exception:
            pass

        try:
            from transformers import VideoMAEImageProcessor

            self.processor = VideoMAEImageProcessor.from_pretrained(cfg.model_name, trust_remote_code=True)
        except Exception:
            self.processor = AutoImageProcessor.from_pretrained(cfg.model_name, trust_remote_code=True)

        # 显式关闭低内存加载 + 不传递 config，避免远程代码模型走上 from_config 路径导致权重加载不完整。
        self.backbone = AutoModel.from_pretrained(
            cfg.model_name,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
        )
        self.backbone.to(self.device_str)

        # 防御性检查：某些 transformers / accelerate 版本下 remote-code 模型仍可能残留 meta tensor。
        meta_params = [n for n, p in self.backbone.named_parameters() if getattr(p, "is_meta", False)]
        meta_buffers = [n for n, b in self.backbone.named_buffers() if getattr(b, "is_meta", False)]
        if meta_params or meta_buffers:
            raise RuntimeError(
                f"VideoMAE v2 backbone 加载后仍存在 meta tensor，无法推理。\n"
                f"meta parameters: {meta_params}\n"
                f"meta buffers: {meta_buffers}\n"
                f"建议：删除 HuggingFace 缓存目录后重试：\n"
                f"  rmdir /s /q %USERPROFILE%\\.cache\\huggingface\\modules\\transformers_modules\\OpenGVLab"
            )

        # 关键修复：远程代码模型中 pos_embed / cls_token 可能是普通 Python 属性（非 Parameter/Buffe
        # 在 init_empty_weights 下会变成 meta tensor 且不会被 load_state_dict 恢复，需递归重建。
        self._fix_meta_tensors(self.backbone)

        # 权重始终保持 FP32；半精度由训练端的 torch.amp.autocast 完成，避免与 GradScaler 冲突（FP16 梯度 unscale 报错）。

        hid = getattr(config, "hidden_size", None) or getattr(config, "embed_dim", None) or 768
        self.embedding_dim = int(hid)

    @staticmethod
    def _get_sinusoid_encoding_table(n_position: int, d_hid: int) -> torch.Tensor:
        """重建 VideoMAE v2 非可学习正弦位置编码（与远程代码一致）。"""

        def _get_position_angle_vec(position: int) -> list[float]:
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [_get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)

    def _fix_meta_tensors(self, module: nn.Module) -> None:
        """递归修复非 Parameter / Buffer 的 meta tensor（如 pos_embed、cls_token）。"""
        for name, obj in list(module.__dict__.items()):
            if not isinstance(obj, torch.Tensor) or not getattr(obj, "is_meta", False):
                continue

            if name == "pos_embed":
                patch_embed = getattr(module, "patch_embed", None)
                embed_dim = getattr(module, "embed_dim", None)
                if patch_embed is not None and embed_dim is not None:
                    num_patches = getattr(patch_embed, "num_patches", None)
                    if num_patches is not None:
                        new_tensor = self._get_sinusoid_encoding_table(num_patches, embed_dim)
                        setattr(module, name, new_tensor.to(self.device_str))
                        continue
            elif name == "cls_token":
                embed_dim = getattr(module, "embed_dim", None)
                if embed_dim is not None:
                    new_tensor = torch.zeros(1, 1, embed_dim)
                    nn.init.trunc_normal_(new_tensor, std=0.02)
                    setattr(module, name, new_tensor.to(self.device_str))
                    continue

            # 兜底：用 zeros 替换并告警
            new_tensor = torch.zeros_like(obj, device=self.device_str)
            setattr(module, name, new_tensor)
            import logging

            logging.getLogger("vit_video_encoder").warning(
                "Fixed meta tensor %s.%s with zeros", module.__class__.__name__, name
            )

        for child in module.children():
            self._fix_meta_tensors(child)

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
        # OpenGVLab/VideoMAEv2-Base (custom remote code) expects (B, C, T, H, W).
        # The HF video processor returns (B, T, C, H, W); permute accordingly.
        if pv.dim() == 5 and pv.shape[1] != 3 and pv.shape[2] == 3:
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
