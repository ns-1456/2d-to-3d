#!/usr/bin/env python3
"""
Training entry point — mixed precision (bfloat16 preferred), checkpoint/resume for Colab.

Checkpoints are written to `checkpoint_dir` (persist on Google Drive in Colab).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from data.dataset import make_dataloader
from models.splat_model import SplatModel
from renderer.rasterizer import DifferentiableGaussianRasterizer
from utils import colab_setup
from utils.loss import photometric_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


amp_dtype_map = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def build_scaler(device_type: str, amp_dtype: torch.dtype):
    # GradScaler mainly for float16; bfloat16 on CUDA often trains fine without scaling.
    enabled = device_type == "cuda" and amp_dtype == torch.float16
    return torch.amp.GradScaler("cuda", enabled=enabled)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    cfg: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else {},
        "epoch": epoch,
        "global_step": global_step,
        "config": cfg,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "np_rng": np.random.get_state(),
        "py_random": random.getstate(),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> Tuple[int, int]:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler.is_enabled() and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    if ckpt.get("torch_rng") is not None:
        torch.set_rng_state(ckpt["torch_rng"])
    if torch.cuda.is_available() and ckpt.get("cuda_rng"):
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    if ckpt.get("np_rng") is not None:
        np.random.set_state(ckpt["np_rng"])
    if ckpt.get("py_random") is not None:
        random.setstate(ckpt["py_random"])
    return int(ckpt.get("epoch", 0)), int(ckpt.get("global_step", 0))


def find_latest_checkpoint_dir(checkpoint_dir: Path) -> Optional[Path]:
    if not checkpoint_dir.is_dir():
        return None
    best: Optional[Tuple[int, Path]] = None
    pat = re.compile(r"step_(\d+)\.pt$")
    for p in checkpoint_dir.glob("*.pt"):
        m = pat.search(p.name)
        if m:
            step = int(m.group(1))
            if best is None or step > best[0]:
                best = (step, p)
    return best[1] if best else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="2D-to-3DGS vehicle training")
    p.add_argument("--config", type=str, default="configs/colab_config.yaml")
    p.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="Path to .pt checkpoint or directory; directory loads latest step_*.pt",
    )
    p.add_argument(
        "--stage_data",
        action="store_true",
        help="Unzip dataset_archive from config into data_root (run once per Colab session)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = load_config(str(cfg_path))
    set_seed(int(cfg.get("seed", 42)))

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    if args.stage_data:
        arch = cfg.get("dataset_archive", "") or ""
        if not arch:
            raise ValueError("--stage_data requires dataset_archive set in config")
        colab_setup.unzip_to_local(arch, cfg["data_root"], overwrite=False)
        print(f"Staged dataset to {cfg['data_root']}")

    loader = make_dataloader(
        data_root=cfg["data_root"],
        batch_size=int(cfg["batch_size"]),
        image_size=int(cfg["image_size"]),
        num_workers=int(cfg["num_workers"]),
        shuffle=True,
    )

    amp_dtype = amp_dtype_map[str(cfg.get("amp_dtype", "bfloat16")).lower()]
    model = SplatModel(
        image_size=int(cfg["image_size"]),
        patch_size=int(cfg["patch_size"]),
        num_gaussians=int(cfg["num_gaussians"]),
        encoder_dim=int(cfg["encoder_dim"]),
        encoder_depth=int(cfg["encoder_depth"]),
    ).to(device)

    rasterizer = DifferentiableGaussianRasterizer().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scaler = build_scaler(device.type, amp_dtype)

    epoch_start = 0
    global_step = 0
    resume_path: Optional[Path] = None
    if args.resume_from:
        rp = Path(args.resume_from).expanduser()
        if rp.is_dir():
            resume_path = find_latest_checkpoint_dir(rp)
            if resume_path is None:
                print(f"No checkpoints found in {rp}; starting fresh.")
        else:
            resume_path = rp
        if resume_path and resume_path.is_file():
            print(f"Resuming from {resume_path}")
            epoch_start, global_step = load_checkpoint(resume_path, model, optimizer, scaler)

    ckpt_dir = Path(cfg["checkpoint_dir"])
    ssim_lambda = float(cfg.get("ssim_lambda", 0.2))
    log_every = int(cfg.get("log_every_n_steps", 10))
    ckpt_every = int(cfg.get("checkpoint_every_n_steps", 100))
    epochs = int(cfg["epochs"])

    model.train()
    for epoch in range(epoch_start, epochs):
        for batch in loader:
            global_step += 1
            inp, tgt, pose = batch
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            _ = pose.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                # inp: [B,3,H,W]
                gaussians = model(inp)
                pred = rasterizer(
                    gaussians,
                    camera=None,
                    image_hw=(inp.shape[-2], inp.shape[-1]),
                )
                loss = photometric_loss(pred, tgt, ssim_lambda=ssim_lambda)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if global_step % log_every == 0:
                print(
                    json.dumps(
                        {"epoch": epoch, "step": global_step, "loss": float(loss.detach().cpu())}
                    )
                )

            if global_step % ckpt_every == 0:
                ckpt_path = ckpt_dir / f"step_{global_step}.pt"
                save_checkpoint(
                    ckpt_path, model, optimizer, scaler, epoch, global_step, cfg
                )
                print(f"Saved checkpoint {ckpt_path}")

    # Final checkpoint
    save_checkpoint(
        ckpt_dir / f"step_{global_step}.pt",
        model,
        optimizer,
        scaler,
        epochs - 1,
        global_step,
        cfg,
    )
    print("Training scaffold complete.")


if __name__ == "__main__":
    main()
