"""
3D MRI branch model.

Input:
    3D structural MRI volume, shape approximately 1 x 97 x 115 x 97

Output:
    MFV = MRI feature vector
    logits = AD/CN prediction logits
"""

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.10):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.MaxPool3d(kernel_size=2),
        )

    def forward(self, x):
        return self.block(x)


class MRI3DBranch(nn.Module):
    def __init__(
        self,
        mfv_dim: int = 128,
        dropout: float = 0.30,
        num_classes: int = 2,
    ):
        super().__init__()

        self.mfv_dim = mfv_dim

        self.features = nn.Sequential(
            ConvBlock3D(1, 8, dropout=0.05),
            ConvBlock3D(8, 16, dropout=0.05),
            ConvBlock3D(16, 32, dropout=0.10),
            ConvBlock3D(32, 64, dropout=0.10),
            nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, mfv_dim),
            nn.LayerNorm(mfv_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(mfv_dim, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        mfv = self.encoder(x)
        logits = self.classifier(mfv)
        return logits, mfv
