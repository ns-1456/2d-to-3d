"""L1 and D-SSIM style losses for photometric supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """pred, target: [B, C, H, W]."""
    return F.l1_loss(pred, target)


def gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d @ window_1d.transpose(0, 1)
    return window_2d.unsqueeze(0).unsqueeze(0)


def ssim_map(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """
    x, y: [B, C, H, W], values in [0, data_range].
    Returns SSIM map [B, C, H- window_size + 1, ...] — we reduce to scalar in d_ssim_loss.
    """
    device, dtype = x.device, x.dtype
    C = x.shape[1]
    window = gaussian_window(window_size, sigma, device, dtype).expand(C, 1, window_size, window_size)
    pad = window_size // 2
    mu_x = F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), window, groups=C)
    mu_y = F.conv2d(F.pad(y, (pad, pad, pad, pad), mode="reflect"), window, groups=C)

    mu_xx = mu_x * mu_x
    mu_yy = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(F.pad(x * x, (pad, pad, pad, pad), mode="reflect"), window, groups=C) - mu_xx
    sigma_y2 = F.conv2d(F.pad(y * y, (pad, pad, pad, pad), mode="reflect"), window, groups=C) - mu_yy
    sigma_xy = F.conv2d(F.pad(x * y, (pad, pad, pad, pad), mode="reflect"), window, groups=C) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_xx + mu_yy + c1) * (sigma_x2.clamp_min(0) + sigma_y2.clamp_min(0) + c2)
    ssim = num / den.clamp_min(1e-8)
    return ssim


def d_ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Returns 1 - mean(SSIM) for stability as additive loss."""
    m = ssim_map(pred, target)
    return 1.0 - m.mean()


def photometric_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    ssim_lambda: float = 0.2,
) -> torch.Tensor:
    """
    L = (1 - λ) L1 + λ * (1 - SSIM).
    pred, target: [B, 3, H, W] in [0, 1].
    """
    return (1.0 - ssim_lambda) * l1_loss(pred, target) + ssim_lambda * d_ssim_loss(pred, target)
