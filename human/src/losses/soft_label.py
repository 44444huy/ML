"""Soft-label KL-divergence loss.

Given model logits and a target probability distribution, minimise
KL(target || softmax(logits)). This is the standard knowledge-distillation
formulation with temperature T=1.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def soft_kl_loss(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """KL-divergence loss for soft labels.

    Args:
        logits       : (B, C) raw model outputs
        target_probs : (B, C) target probability distribution (rows sum to ~1)
    Returns:
        scalar loss averaged over the batch
    """
    # Stabilise: clamp target to avoid log(0) when computing KL in the other
    # direction internally. PyTorch KLDivLoss expects log-probs on input side
    # and probs on target side.
    log_pred = F.log_softmax(logits, dim=-1)
    return F.kl_div(log_pred, target_probs, reduction="batchmean")
