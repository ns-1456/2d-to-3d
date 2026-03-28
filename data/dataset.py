"""
Paired image dataset from **manifest.jsonl** (sketch → 3D training scaffold).

Layout:
  data_root/
    manifest.jsonl   # each line: {"model_id":"...", "input":"rel/sketch.png", "target":"rel/rgb.png", "pose":"optional.npy"}

`input`  = conditioning (e.g. line drawing / synthetic sketch).  
`target` = supervision (e.g. shaded RGB render of the same underlying shape).

Paths are relative to data_root. Use fast local disk on Colab (e.g. under /content/...).
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


class PairedImageDataset(Dataset):
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
                f"Training requires {manifest.resolve()}. "
                'Each line: {"model_id":"...","input":"sketch/xxx.png","target":"rgb/xxx.png","pose":"optional.npy"} '
                "Colab: enable gen_sketch_if_missing in configs/colab_config.yaml or run "
                "python scripts/gen_sketch3d_synthetic.py --out_dir <data_root> ..."
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


# Backwards compatibility
ShapeNetVehicleDataset = PairedImageDataset


def make_dataloader(
    data_root: str,
    batch_size: int,
    image_size: int,
    num_workers: int = 2,
    shuffle: bool = True,
) -> DataLoader:
    ds = PairedImageDataset(
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
