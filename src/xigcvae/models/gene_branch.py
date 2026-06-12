"""
Gene-expression branch model.

Input:
    DEG-ranked gene/probe expression vector

Output:
    GFV = gene-expression feature vector
    logits = AD/CN prediction logits
"""

import torch
import torch.nn as nn


class GeneExpressionBranch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        gfv_dim: int = 128,
        dropout: float = 0.60,
        num_classes: int = 2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.gfv_dim = gfv_dim

        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, gfv_dim),
            nn.LayerNorm(gfv_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(gfv_dim, num_classes)

    def forward(self, x: torch.Tensor):
        gfv = self.encoder(x)
        logits = self.classifier(gfv)
        return logits, gfv
