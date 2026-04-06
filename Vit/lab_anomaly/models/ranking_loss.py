"""
MIL 排序损失（借鉴 CVPR 2018: Real-world Anomaly Detection in Surveillance Videos）。

核心思想：
  - 正 bag（异常视频）中得分最高的 clip 应该比负 bag（正常视频）中得分最高的 clip 分数更高
  - 稀疏约束：异常只出现在少数 clip，大部分 clip 分数应该为 0
  - 时间平滑约束：相邻 clip 的分数应该连续变化，不应跳动

损失公式：
  L = max(0, 1 - max(scores_pos) + max(scores_neg))
    + lambda_sparse * sum(scores_pos)
    + lambda_smooth * sum((scores_pos[i] - scores_pos[i+1])^2)
"""
from __future__ import annotations

from typing import Optional

import torch


def masked_max(scores: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    对每行求带 mask 的最大值。

    Args:
        scores: (B, N)  每个 clip 的异常分数
        mask:   (B, N) bool, True=有效位置。None 时视为全部有效。

    Returns:
        (B,) 每行的最大分数
    """
    if mask is not None:
        fill_val = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask, fill_val)
    return scores.max(dim=1).values


def mil_ranking_loss(
    scores_pos: torch.Tensor,
    scores_neg: torch.Tensor,
    mask_pos: Optional[torch.Tensor] = None,
    mask_neg: Optional[torch.Tensor] = None,
    lambda_sparse: float = 8e-5,
    lambda_smooth: float = 8e-5,
) -> torch.Tensor:
    """
    MIL 排序损失 + 稀疏约束 + 时间平滑约束。

    Args:
        scores_pos: (B_pos, N) 正 bag（异常视频）各 clip 的异常分数（Sigmoid 输出 0~1）
        scores_neg: (B_neg, N) 负 bag（正常视频）各 clip 的异常分数
        mask_pos:   (B_pos, N) bool, True=有效 clip。None=全有效。
        mask_neg:   (B_neg, N) bool。None=全有效。
        lambda_sparse: 稀疏约束权重（论文建议 8e-5）
        lambda_smooth: 平滑约束权重（论文建议 8e-5）

    Returns:
        标量 loss
    """
    b_pos = scores_pos.size(0)
    b_neg = scores_neg.size(0)
    device = scores_pos.device

    # 边界：正或负 bag 为空时返回 0（不参与反向，避免无梯度的常量张量）
    if b_pos == 0 or b_neg == 0:
        return torch.tensor(0.0, device=device)

    # --- 1) 排序项 ---
    max_pos = masked_max(scores_pos, mask_pos)  # (B_pos,)
    max_neg = masked_max(scores_neg, mask_neg)  # (B_neg,)

    # 配对方式：对齐 min(B_pos, B_neg) 组；多出来的 max 分数循环配对
    n_pairs = max(b_pos, b_neg)
    if b_pos < n_pairs:
        idx_pos = torch.arange(n_pairs, device=device) % b_pos
        max_pos = max_pos[idx_pos]
    if b_neg < n_pairs:
        idx_neg = torch.arange(n_pairs, device=device) % b_neg
        max_neg = max_neg[idx_neg]

    # hinge loss: max(0, 1 - max_pos + max_neg)
    rank_loss = torch.clamp(1.0 - max_pos + max_neg, min=0.0).mean()

    # --- 2) 稀疏项：正 bag 分数之和尽量小 ---
    if mask_pos is not None:
        sparse_loss = (scores_pos * mask_pos.float()).sum(dim=1).mean()
    else:
        sparse_loss = scores_pos.sum(dim=1).mean()

    # --- 3) 平滑项：相邻 clip 分数差的平方和（仅对有效位置，避免 padding 边界干扰）---
    if scores_pos.size(1) > 1:
        diff = scores_pos[:, 1:] - scores_pos[:, :-1]  # (B_pos, N-1)
        if mask_pos is not None:
            valid_pair = mask_pos[:, 1:] & mask_pos[:, :-1]  # 仅当相邻两 clip 都有效时才计入
            n_valid = valid_pair.float().sum(dim=1).clamp(min=1)
            smooth_per_bag = (diff ** 2 * valid_pair.float()).sum(dim=1) / n_valid
            smooth_loss = smooth_per_bag.mean()
        else:
            smooth_loss = (diff ** 2).sum(dim=1).mean()
    else:
        smooth_loss = torch.tensor(0.0, device=device)

    total = rank_loss + lambda_sparse * sparse_loss + lambda_smooth * smooth_loss
    return total
