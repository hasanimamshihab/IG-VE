from __future__ import annotations

import torch
import torch.nn as nn


class AGFVNNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: tuple[int, ...] = (64,),
        dropout: float = 0.30,
    ):
        super().__init__()

        dims = [input_dim] + list(hidden_dims)
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-1], 2))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
