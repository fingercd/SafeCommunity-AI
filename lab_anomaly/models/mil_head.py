"""
MIL（Multiple Instance Learning）头：将多个 clip embedding 聚合为视频级分类 logits。

支持两种聚合方式：
  - attn：attention pooling，学权重后加权求和
  - topk：按 clip 分类分数选 top-k clip，再平均 logits
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(x: torch.Tensor, mask: Optional[torch.Tensor], dim: int) -> torch.Tensor:
    """带 mask 的 softmax：无效位置置 -inf 后 softmax，再 mask 为 0"""
    """
    x: (..., N, ...)
    mask: same shape as x reduced on dim, or broadcastable to x
      - True 表示有效位置
    """
    if mask is None:
        return F.softmax(x, dim=dim)
    # 把无效位置置为 -inf
    x = x.masked_fill(~mask, float("-inf"))
    # softmax 后无效位置变成 0
    w = F.softmax(x, dim=dim)
    w = w.masked_fill(~mask, 0.0)
    return w


@dataclass(frozen=True)
class MILHeadConfig:
    """MIL 头配置：embedding 维、类别数、pooling 类型、topk、attention hidden 等"""
    embedding_dim: int
    num_classes: int
    pooling: str = "attn"  # attn | topk
    topk: int = 2
    attn_hidden: int = 256
    dropout: float = 0.0
    use_layernorm: bool = True


class AttentionPooling(nn.Module):
    """
    对 clips 做 attention pooling：
      a_i = softmax( w^T tanh(W x_i) )
      pooled = sum_i a_i * x_i
    """

    def __init__(self, embedding_dim: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, N, D)
        mask: (B, N) bool, True=valid
        returns: pooled (B, D), attn (B, N)
        """
        h = torch.tanh(self.fc1(x))  # (B,N,H)
        logits = self.fc2(h).squeeze(-1)  # (B,N)
        attn = masked_softmax(logits, mask=mask, dim=1)
        pooled = torch.sum(attn.unsqueeze(-1) * x, dim=1)
        return pooled, attn


class MILClassifier(nn.Module):
    """
    输入：clip embeddings (B, N, D)
    输出：视频级 logits (B, C)
    """

    def __init__(self, cfg: MILHeadConfig):
        super().__init__()
        self.cfg = cfg
        self.pooling = cfg.pooling.lower().strip()
        if self.pooling not in {"attn", "topk"}:
            raise ValueError(f"unknown pooling={cfg.pooling!r}, expected 'attn' or 'topk'")

        self.norm = nn.LayerNorm(cfg.embedding_dim) if cfg.use_layernorm else nn.Identity()
        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

        if self.pooling == "attn":
            self.pool = AttentionPooling(cfg.embedding_dim, hidden=cfg.attn_hidden)
            self.classifier = nn.Linear(cfg.embedding_dim, cfg.num_classes)
        else:
            # top-k：先做 clip-level logits，再选择 top-k clip 聚合
            self.clip_classifier = nn.Linear(cfg.embedding_dim, cfg.num_classes)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        x: (B,N,D)
        mask: (B,N) bool, True=valid
        y: (B,) long, 仅 top-k pooling 训练时用于“按 GT 类别选 top-k”
        """
        x = self.drop(self.norm(x))

        details: dict = {}
        if self.pooling == "attn":
            pooled, attn = self.pool(x, mask=mask)
            logits = self.classifier(self.drop(pooled))
            if return_details:
                details["attn"] = attn
                details["pooled"] = pooled
            return logits, details

        # top-k pooling
        clip_logits = self.clip_classifier(x)  # (B,N,C)
        if return_details:
            details["clip_logits"] = clip_logits

        b, n, c = clip_logits.shape
        if mask is None:
            mask = torch.ones((b, n), device=clip_logits.device, dtype=torch.bool)

        pooled_logits = []
        for i in range(b):
            valid_idx = torch.nonzero(mask[i], as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                pooled_logits.append(torch.zeros((c,), device=clip_logits.device, dtype=clip_logits.dtype))
                continue

            if y is not None:
                yi = int(y[i].item())
                score = clip_logits[i, valid_idx, yi]
            else:
                # 推理：用“最强证据”选择 top-k
                score = clip_logits[i, valid_idx, :].max(dim=-1).values

            k = min(int(self.cfg.topk), int(valid_idx.numel()))
            topk_rel = torch.topk(score, k=k, largest=True).indices
            topk_idx = valid_idx[topk_rel]
            pooled_logits.append(clip_logits[i, topk_idx, :].mean(dim=0))

        logits = torch.stack(pooled_logits, dim=0)  # (B,C)
        return logits, details

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits, _ = self.forward(x, mask=mask, y=None, return_details=False)
        return F.softmax(logits, dim=-1)

