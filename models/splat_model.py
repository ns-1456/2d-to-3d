"""
Full splat model: ViM encoder -> MLP decoder -> constrained Gaussian parameters.

Typical conditioning x is a **3×H×W sketch** (line drawing); training uses paired RGB targets.

Tensor shape contract (batch B, num_gaussians N, encoder dim D):

1) Input image x (sketch or RGB):
   x.shape == [B, 3, H, W]

2) Encoder forward:
   pooled = encoder(x)
   pooled.shape == [B, D]

3) Decoder forward:
   raw = decoder(pooled)
   raw.shape == [B, N, 14]

4) Parameter split (per Gaussian, diffuse RGB only — no spherical harmonics):
   indices 0:3   -> xyz (unbounded raw -> tanh * scale for bounded scene prior)
   indices 3:6   -> log_scale (softplus -> positive scales)
      We store network output as"log_scale"; exp after softplus yields ellipsoid radii.
   indices 6:10  -> quaternion (qx,qy,qz,qw); F.normalize for valid rotation
   indices 10    -> logit_opacity -> sigmoid
   indices 11:14 -> rgb pre-activation -> sigmoid for [0,1]

5) Output dict of tensors, each [B, N, ?]:
   "xyz": [B, N, 3]
   "scale": [B, N, 3] strictly positive
   "quaternion": [B, N, 4] unit L2 norm
   "opacity": [B, N, 1] in (0, 1)
   "rgb": [B, N, 3] in (0, 1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder_mlp import GaussianMLPDecoder
from .encoder_vim import VimEncoder


@dataclass
class GaussianParameters:
    """Typed view of batched Gaussian parameters (all floats under autocast)."""

    xyz: torch.Tensor  # [B, N, 3]
    scale: torch.Tensor  # [B, N, 3]
    quaternion: torch.Tensor  # [B, N, 4]
    opacity: torch.Tensor  # [B, N, 1]
    rgb: torch.Tensor  # [B, N, 3]

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "xyz": self.xyz,
            "scale": self.scale,
            "quaternion": self.quaternion,
            "opacity": self.opacity,
            "rgb": self.rgb,
        }


def split_raw_gaussians(raw: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    raw: [B, N, 14] decoder outputs (any dtype under autocast).
    Returns tuple of tensors matching GaussianParameters fields' last dims.
    """
    xyz = raw[..., 0:3]
    log_scale = raw[..., 3:6]
    quat = raw[..., 6:10]
    logit_o = raw[..., 10:11]
    rgb = raw[..., 11:14]
    return xyz, log_scale, quat, logit_o, rgb


class SplatModel(nn.Module):
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 32,
        num_gaussians: int = 10_000,
        encoder_dim: int = 256,
        encoder_depth: int = 6,
        scene_scale: float = 2.0,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_gaussians = num_gaussians
        self.encoder_dim = encoder_dim
        self.scene_scale = scene_scale

        self.encoder = VimEncoder(
            image_size=image_size,
            patch_size=patch_size,
            dim=encoder_dim,
            depth=encoder_depth,
            in_ch=3,
        )
        self.decoder = GaussianMLPDecoder(dim=encoder_dim, num_gaussians=num_gaussians)

    def raw_to_parameters(self, raw: torch.Tensor) -> GaussianParameters:
        """
        raw: [B, N, 14]
        Returns GaussianParameters with valid ranges for renderer / loss.
        """
        xyz_r, log_scale, quat, logit_o, rgb_r = split_raw_gaussians(raw)
        # Bounded positions in [-scene_scale, scene_scale]^3 (soft prior for vehicles near origin)
        xyz = torch.tanh(xyz_r) * self.scene_scale
        # Positive scales; softplus + small epsilon for stability
        scale = F.softplus(log_scale) + 1e-4
        quat = F.normalize(quat, dim=-1)
        opacity = torch.sigmoid(logit_o)
        rgb = torch.sigmoid(rgb_r)
        return GaussianParameters(
            xyz=xyz,
            scale=scale,
            quaternion=quat,
            opacity=opacity,
            rgb=rgb,
        )

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        image: [B, 3, H, W] — should match self.image_size for patch math.
        Returns dict of [B, N, *] tensors (see GaussianParameters).
        """
        pooled = self.encoder(image)
        raw = self.decoder(pooled)
        gp = self.raw_to_parameters(raw)
        return gp.as_dict()
