from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dims, dropout=0.20):
        super().__init__()

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AlignedGFVModel(nn.Module):
    """
    MRI-aligned molecular representation model.

    GFV encoder:
        measured GFV128 -> aligned molecular feature AGFV128

    MFV encoder:
        MRI-derived MFV128 -> aligned molecular feature AGFV128

    Shared classifier:
        AGFV128 -> AD/CN

    GFV decoder:
        AGFV128 -> approximate original GFV128

    Main inference for MRI-only subjects:
        MFV128 -> AGFV128

    Diagnosis is never used as input to either encoder.
    """

    def __init__(
        self,
        gfv_dim=128,
        mfv_dim=128,
        agfv_dim=128,
        hidden_dim=256,
        dropout=0.20,
    ):
        super().__init__()

        self.gfv_dim = gfv_dim
        self.mfv_dim = mfv_dim
        self.agfv_dim = agfv_dim

        self.gfv_encoder = MLP(
            [gfv_dim, hidden_dim, hidden_dim, agfv_dim],
            dropout=dropout,
        )

        self.mfv_encoder = MLP(
            [mfv_dim, hidden_dim, hidden_dim, agfv_dim],
            dropout=dropout,
        )

        self.classifier = MLP(
            [agfv_dim, hidden_dim // 2, 2],
            dropout=dropout,
        )

        self.gfv_decoder = MLP(
            [agfv_dim, hidden_dim, hidden_dim, gfv_dim],
            dropout=dropout,
        )

    def encode_gfv(self, gfv):
        return self.gfv_encoder(gfv)

    def encode_mfv(self, mfv):
        return self.mfv_encoder(mfv)

    def decode_to_gfv(self, agfv):
        return self.gfv_decoder(agfv)

    def forward(self, gfv, mfv):
        z_g = self.encode_gfv(gfv)
        z_m = self.encode_mfv(mfv)

        logits_g = self.classifier(z_g)
        logits_m = self.classifier(z_m)

        recon_g = self.decode_to_gfv(z_g)
        pred_g_from_m = self.decode_to_gfv(z_m)

        return {
            "z_g": z_g,
            "z_m": z_m,
            "logits_g": logits_g,
            "logits_m": logits_m,
            "recon_g": recon_g,
            "pred_g_from_m": pred_g_from_m,
        }
