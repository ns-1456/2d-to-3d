"""
MLP head: maps global image features to a fixed number of Gaussians.

Input:  features [B, D]  (pooled ViM token sequence)
Output: raw_params [B, N, P] where P = num_param_per_gaussian (14 for this project)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GaussianMLPDecoder(nn.Module):
    NUM_PARAM_PER_GAUSSIAN = 14  # xyz(3) + scale(3) + quat(4) + opacity(1) + rgb(3)

    def __init__(self, dim: int, num_gaussians: int, hidden_mult: int = 4) -> None:
        super().__init__()
        self.num_gaussians = num_gaussians
        out_dim = num_gaussians * self.NUM_PARAM_PER_GAUSSIAN
        hidden = dim * hidden_mult
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D] pooled features (float32 or bfloat16 under autocast).
        returns raw_params: [B, N, P] — unconstrained network outputs; SplatModel maps to valid ranges.
        """
        b = x.shape[0]
        flat = self.net(x)
        return flat.view(b, self.num_gaussians, self.NUM_PARAM_PER_GAUSSIAN)
