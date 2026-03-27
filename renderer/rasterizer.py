"""
Differentiable Gaussian rasterizer — Phase 1 **shell**.

Target API for a future fused Triton/CUDA implementation:
  forward(gaussians, camera, image_hw) -> [B, 3, H, W]

The placeholder forward expands a simple function of opacity-weighted mean RGB so
`train.py` can run backward through decoder/encoder before the real kernel exists.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class DifferentiableGaussianRasterizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        gaussians: Dict[str, torch.Tensor],
        camera: Optional[Dict[str, torch.Tensor]] = None,
        image_hw: Tuple[int, int] = (256, 256),
    ) -> torch.Tensor:
        """
        gaussians: dict with keys xyz [B,N,3], scale, quaternion, opacity [B,N,1], rgb [B,N,3]
        camera: optional dict with pose/intrinsics (reserved for real projection).
        image_hw: (H, W) output resolution.

        **Placeholder:** opacity-weighted mean RGB, broadcast to full image — NOT physically correct.
        Preserves autograd to rgb/opacity for scaffolding only.
        """
        _ = camera
        rgb = gaussians["rgb"]
        opa = gaussians["opacity"]
        # rgb: [B, N, 3], opa: [B, N, 1]
        w = opa.clamp(1e-4, 1.0)
        num = (rgb * w).sum(dim=1)
        den = w.sum(dim=1).clamp_min(1e-6)
        mean = num / den  # [B, 3]
        h, w_px = image_hw
        b = mean.shape[0]
        out = mean[:, :, None, None].expand(b, 3, h, w_px)
        return out

    def forward_fused(
        self,
        gaussians: Dict[str, torch.Tensor],
        camera: Dict[str, Any],
        image_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Reserved entry point for Triton/CUDA fused path (project, sort, alpha blend).
        Raises until implemented.
        """
        raise NotImplementedError(
            "Fused Triton/CUDA rasterizer not implemented; use forward() placeholder."
        )
