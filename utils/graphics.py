"""Camera and rotation helpers. Tensor shapes annotated for batched use (B = batch)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def normalize_quaternion(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    q: [..., 4] XYZW convention (vector part first, scalar last) or scalar-first;
    We use (qx, qy, qz, qw) per row.
    Returns unit quaternion with same shape.
    """
    return F.normalize(q, dim=-1, eps=eps)


def quaternion_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    q: [..., 4] as (qx, qy, qz, qw), normalized.
    Returns R: [..., 3, 3] rotation matrices.
    """
    q = normalize_quaternion(q)
    x, y, z, w = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    m00 = 1 - 2 * (yy + zz)
    m01 = 2 * (xy - wz)
    m02 = 2 * (xz + wy)
    m10 = 2 * (xy + wz)
    m11 = 1 - 2 * (xx + zz)
    m12 = 2 * (yz - wx)
    m20 = 2 * (xz - wy)
    m21 = 2 * (yz + wx)
    m22 = 1 - 2 * (xx + yy)

    row0 = torch.stack((m00, m01, m02), dim=-1)
    row1 = torch.stack((m10, m11, m12), dim=-1)
    row2 = torch.stack((m20, m21, m22), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def build_intrinsics(fx: float, fy: float, cx: float, cy: float, device, dtype) -> torch.Tensor:
    """Returns 3x3 intrinsics K on device."""
    z = torch.zeros((), device=device, dtype=dtype)
    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    return K


def perspective_project(
    points_cam: torch.Tensor,
    K: torch.Tensor,
) -> torch.Tensor:
    """
    points_cam: [B, N, 3] camera-space XYZ (Z forward, right-handed typical OpenGL uses -Z; caller aligns).
    K: [3, 3] or [B, 3, 3]
    Returns uv: [B, N, 2] pixel coordinates (before clipping).
    """
    if K.dim() == 2:
        K = K.unsqueeze(0).expand(points_cam.shape[0], -1, -1)
    z = points_cam[..., 2:3].clamp_min(1e-6)
    xy = points_cam[..., 0:2]
    xyn = xy / z
    ones = torch.ones_like(z)
    homog = torch.cat((xyn, ones), dim=-1)
    # uv_h = K @ xyn_h
    uv_h = torch.einsum("bij,bnj->bni", K, homog)
    uv = uv_h[..., :2]
    return uv
