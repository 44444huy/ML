"""Binary MLP for dog eye-color classification.

Same architecture used by both M1 (BCE + class_weight) and M2 (focal
loss). Output is a single logit (sigmoid applied externally).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class MLPBinary(nn.Module):
    """Small MLP for binary classification on tabular SNP dosage data."""

    def __init__(self, in_dim: int, hidden: int = 128, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        assert n_layers >= 1
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logit, shape (B,)."""
        return self.head(self.backbone(x)).squeeze(-1)
