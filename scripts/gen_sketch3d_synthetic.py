#!/usr/bin/env python3
"""
Build a **narrow-domain synthetic dataset**: random rigid primitives (box / sphere / cylinder)
→ orthographic depth buffer → **sketch** (edge map) + **shaded RGB** (pseudo photo).

Trains the existing pipeline as: **human-like drawing (sketch) in**, **shaded render out**
(photometric loss). Domain is small synthetic solids so a modest network can specialize.

Usage (Colab, from repo root):
  python scripts/gen_sketch3d_synthetic.py --out_dir data/sketch3d_train --num_samples 800 --size 128
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Random rotation (3,3) via QR."""
    a = rng.standard_normal((3, 3))
    q, _ = np.linalg.qr(a)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.astype(np.float32)


def _sample_box_surface(rng: np.random.Generator, grid: int = 10) -> np.ndarray:
    """Uniform samples on axis-aligned unit-ish box surface, scaled per-axis."""
    s = rng.uniform(0.35, 1.0, size=3).astype(np.float32)
    pts = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            u = rng.uniform(-1, 1, size=(grid * grid, 2))
            patch = np.zeros((grid * grid, 3), dtype=np.float32)
            patch[:, axis] = sign
            a, b = (axis + 1) % 3, (axis + 2) % 3
            patch[:, a] = u[:, 0] * s[a]
            patch[:, b] = u[:, 1] * s[b]
            patch[:, axis] *= s[axis]
            pts.append(patch)
    return np.concatenate(pts, axis=0)


def _sample_sphere(rng: np.random.Generator, n: int = 600) -> np.ndarray:
    r = rng.uniform(0.35, 0.95)
    # Uniform on sphere: normalise Gaussian 3-vectors
    v = rng.standard_normal((n, 3)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
    return (v * r).astype(np.float32)


def _sample_cylinder(rng: np.random.Generator, rings: int = 16, per_ring: int = 24) -> np.ndarray:
    r = rng.uniform(0.25, 0.85)
    h = rng.uniform(0.4, 1.2)
    pts = []
    th = np.linspace(0, 2 * math.pi, per_ring, endpoint=False)
    for z in np.linspace(-h, h, rings):
        ring = np.stack([r * np.cos(th), r * np.sin(th), np.full(per_ring, z)], axis=1)
        pts.append(ring.astype(np.float32))
    # caps
    for z, sgn in ((-h, -1), (h, 1)):
        rr = np.linspace(0, r, 6)
        tt = np.linspace(0, 2 * math.pi, 24, endpoint=False)
        for rv in rr[1:]:
            cap = np.stack([rv * np.cos(tt), rv * np.sin(tt), np.full_like(tt, z)], axis=1)
            pts.append(cap.astype(np.float32))
    return np.concatenate(pts, axis=0)


def _random_shape_points(rng: np.random.Generator) -> np.ndarray:
    k = rng.integers(0, 3)
    if k == 0:
        p = _sample_box_surface(rng)
    elif k == 1:
        p = _sample_sphere(rng)
    else:
        p = _sample_cylinder(rng)
    R = _random_rotation_matrix(rng)
    p = (R @ p.T).T
    t = rng.uniform(-0.15, 0.15, size=3).astype(np.float32)
    return p + t


def _dilate_binary(sk: np.ndarray) -> np.ndarray:
    h, w = sk.shape
    pad = np.pad(sk, 1, mode="constant", constant_values=0.0)
    acc = pad[1 : h + 1, 1 : w + 1].copy()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            acc = np.maximum(acc, pad[1 + di : h + 1 + di, 1 + dj : w + 1 + dj])
    return acc


def _z_buffer(u: np.ndarray, v: np.ndarray, z: np.ndarray, H: int, W: int) -> np.ndarray:
    buf = np.full((H, W), np.nan, dtype=np.float32)
    order = np.argsort(z)  # draw nearer points last (overwrite)
    for i in order:
        vi, ui = int(v[i]), int(u[i])
        buf[vi, ui] = z[i]
    far = np.nanmax(buf)
    buf = np.where(np.isnan(buf), far, buf)
    return buf


def _depth_to_sketch_and_rgb(depth: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    d = depth.copy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    gy, gx = np.gradient(d.astype(np.float64))
    mag = np.sqrt(gx * gx + gy * gy)
    thr = np.percentile(mag, 82 + rng.integers(0, 10))
    sketch = (mag > thr).astype(np.float32)
    for _ in range(int(rng.integers(0, 2))):
        sketch = _dilate_binary(sketch)
    sk_rgb = (np.stack([sketch] * 3, axis=-1) * 255).astype(np.uint8)

    dzdx = np.gradient(d, axis=1)
    dzdy = np.gradient(d, axis=0)
    nx, ny = -dzdx, -dzdy
    nz = np.ones_like(d)
    nn = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    nx, ny, nz = nx / nn, ny / nn, nz / nn
    lx, ly, lz = 0.25, 0.35, 0.92
    shade = np.clip(nx * lx + ny * ly + nz * lz, 0, 1)
    tint = np.array([0.92, 0.94, 1.0], dtype=np.float32)
    rgb = (shade[..., None] * tint * 255).astype(np.uint8)
    return sk_rgb, rgb


def render_one(rng: np.random.Generator, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    pts = _random_shape_points(rng)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    u = np.clip(((x + 1) * 0.5 * (W - 1)).astype(np.int32), 0, W - 1)
    v = np.clip(((1 - (y + 1) * 0.5) * (H - 1)).astype(np.int32), 0, H - 1)
    zi = z.astype(np.float32)
    depth = _z_buffer(u, v, zi, H, W)
    return _depth_to_sketch_and_rgb(depth, rng)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="data/sketch3d_train")
    p.add_argument("--num_samples", type=int, default=800)
    p.add_argument("--size", type=int, default=128, help="H=W for raster (128 is fast on Colab)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    root = Path(args.out_dir)
    sk_dir = root / "sketch"
    rgb_dir = root / "rgb"
    sk_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    manifest_path = root / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for i in range(args.num_samples):
            sk, rgb = render_one(rng, args.size, args.size)
            sid = f"{i:06d}"
            sk_p = sk_dir / f"{sid}.png"
            rgb_p = rgb_dir / f"{sid}.png"
            Image.fromarray(sk).save(sk_p)
            Image.fromarray(rgb).save(rgb_p)
            rec = {
                "model_id": sid,
                "input": f"sketch/{sid}.png",
                "target": f"rgb/{sid}.png",
            }
            mf.write(json.dumps(rec) + "\n")
            if (i + 1) % 100 == 0:
                print(f"wrote {i + 1}/{args.num_samples}")

    print(f"Done: {manifest_path} with {args.num_samples} pairs under {root}")


if __name__ == "__main__":
    main()
