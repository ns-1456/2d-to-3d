"""
Vision Mamba–style encoder over image patches.

Sequence length L = (H/patch) * (W/patch). With H=W=256 and patch=32 → L=64 tokens.

If `mamba-ssm` is installed, stacks real Mamba SSM blocks. Otherwise uses a lightweight
GLU/Linear mixer as a drop-in stub so the repo runs without compiling extensions locally.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba

    _HAS_MAMBA = True
except ImportError:
    Mamba = None  # type: ignore[misc, assignment]
    _HAS_MAMBA = False


class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W] — H,W divisible by patch_size.
        returns tokens: [B, L, D], L = (H/ps)*(W/ps).
        """
        z = self.proj(x)  # [B, D, H', W']
        b, d, h, w = z.shape
        z = z.flatten(2).transpose(1, 2)  # [B, L, D]
        return z


class SequenceMixerBlock(nn.Module):
    """Fallback block when mamba_ssm is unavailable: LN + gating on sequence dimension."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, dim * 2)
        self.down = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        h = self.norm(x)
        a, b = self.up(h).chunk(2, dim=-1)
        h = a * torch.sigmoid(b)
        h = self.down(h)
        return x + h


class VimEncoder(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        dim: int,
        depth: int,
        in_ch: int = 3,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embed = PatchEmbed(in_ch, dim, patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        blocks = []
        for _ in range(depth):
            if _HAS_MAMBA:
                blocks.append(
                    nn.Sequential(
                        nn.LayerNorm(dim),
                        Mamba(d_model=dim, d_state=16, d_conv=4, expand=2),
                    )
                )
            else:
                blocks.append(SequenceMixerBlock(dim))
        self.blocks = nn.ModuleList(blocks)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W] — expect H,W == image_size; use torch.cuda.amp.autocast(dtype=bfloat16) outside.

        Returns pooled: [B, D] global average over patch tokens after stack.
        Intermediate after patch embedding + pos: [B, L, D], L = num_patches.
        """
        t = self.patch_embed(x)
        t = t + self.pos_embed
        for blk in self.blocks:
            if _HAS_MAMBA:
                # Mamba submodule: residual inside our Sequential is (LN -> Mamba), add outer residual
                residual = t
                t = blk(t) + residual
            else:
                t = blk(t)
        t = self.out_norm(t)
        # [B, L, D] -> [B, D]
        return t.mean(dim=1)
