"""Simple MLP for single-trait classification.

Used by two training objectives:
    M1: hard-label cross-entropy on argmax(HIrisPlex)
    M2: soft-label KL-divergence on raw HIrisPlex p_values

Same architecture, only the loss differs. Keeping architecture identical
across M1 and M2 is important for a clean ablation.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """A small 2-hidden-layer MLP.

    Args:
        in_dim      : number of SNP features (41)
        n_classes   : number of output classes for the trait
        hidden      : hidden layer width
        n_layers    : number of hidden layers (>=1)
        dropout     : dropout after each hidden activation
    """

    def __init__(self,
                 in_dim: int,
                 n_classes: int,
                 hidden: int = 128,
                 n_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        assert n_layers >= 1
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (not softmax)."""
        return self.head(self.backbone(x))
