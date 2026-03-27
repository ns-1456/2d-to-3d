"""
ShapeNet-style vehicle dataloader: **real images only** from manifest.jsonl.

Layout:
  data_root/
    manifest.jsonl   # one JSON object per line
    ...              # image paths relative to data_root

Each line must include keys "input" and "target" (relative paths to RGB images).
Optional "pose": relative path to a 4x4 float32 .npy (world/camera); if missing, identity is used.

Loads only from `data_root` (use fast local disk on Colab, e.g. /content/data/...).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def _load_image(path: Path, image_size: int) -> torch.Tensor:
    """Load image as [3, H, W] float in [0,1]. Fails if file missing or corrupt."""
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as im:
        im = im.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(im) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).float()


class ShapeNetVehicleDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        image_size: int = 256,
        split: str = "train",
    ) -> None:
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.split = split
        manifest = self.data_root / "manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(
                f"Real-data training requires {manifest.resolve()}. "
                'Each line: {{"model_id":"...","input":"rel/input.png","target":"rel/target.png","pose":"optional.npy"}} '
                "Paths are relative to data_root. For a tiny local demo run: python scripts/bootstrap_demo_images.py"
            )
        self._entries = self._load_manifest(manifest)
        if not self._entries:
            raise ValueError(f"{manifest} contains no samples (empty or invalid lines).")

    def _load_manifest(self, manifest: Path) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        entry = self._entries[idx]
        inp_path = self.data_root / entry["input"]
        tgt_path = self.data_root / entry["target"]
        inp = _load_image(inp_path, self.image_size)
        tgt = _load_image(tgt_path, self.image_size)
        pose_rel = entry.get("pose", "")
        if pose_rel:
            pose_path = self.data_root / pose_rel
            if pose_path.is_file():
                pose_np = np.load(pose_path)
                pose = torch.from_numpy(np.asarray(pose_np, dtype=np.float32))
                if pose.shape != (4, 4):
                    raise ValueError(f"Pose must be 4x4, got {tuple(pose.shape)} for {pose_path}")
            else:
                raise FileNotFoundError(f"Pose file not found: {pose_path}")
        else:
            pose = torch.eye(4, dtype=torch.float32)
        return inp, tgt, pose


def make_dataloader(
    data_root: str,
    batch_size: int,
    image_size: int,
    num_workers: int = 2,
    shuffle: bool = True,
) -> DataLoader:
    ds = ShapeNetVehicleDataset(
        data_root=data_root,
        image_size=image_size,
    )
    if len(ds) < batch_size:
        raise ValueError(
            f"Dataset has {len(ds)} samples but batch_size={batch_size}. "
            "Lower batch_size or add more manifest lines."
        )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
