"""
ShapeNet vehicle subset dataloader (skeleton).

Expected on-disk layout when `synthetic=False`:
  data_root/
    manifest.jsonl   # each line: {"model_id": "...", "input": "rel/path.png", "target": "rel/target.png", "pose": "rel/pose.npy"}

Alternatively, a flat folder with paired files can be added later.

Loads only from `data_root` on **local Colab NVMe** (`/content/data/...`) — never from Drive in the hot path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _load_image(path: Path, image_size: int) -> torch.Tensor:
    """Load image as [3, H, W] float in [0,1]. Uses PIL if available."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
            arr = np.asarray(im) / 255.0
    except Exception:
        # Fallback: random tensor for broken paths during bring-up
        arr = np.random.rand(image_size, image_size, 3).astype(np.float32)
    t = torch.from_numpy(arr).permute(2, 0, 1).float()
    return t


class ShapeNetVehicleDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        image_size: int = 256,
        synthetic: bool = False,
        synthetic_len: int = 256,
        split: str = "train",
    ) -> None:
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.synthetic = synthetic
        self.synthetic_len = synthetic_len
        self.split = split
        self._entries: List[Dict[str, Any]] = []
        if not self.synthetic:
            self._entries = self._discover_manifest()
            if not self._entries:
                # Auto-fallback for empty root (smoke tests / first Colab unzip)
                self.synthetic = True

    def _discover_manifest(self) -> List[Dict[str, Any]]:
        manifest = self.data_root / "manifest.jsonl"
        entries: List[Dict[str, Any]] = []
        if manifest.is_file():
            with open(manifest, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        return entries

    def __len__(self) -> int:
        if self.synthetic:
            return self.synthetic_len
        return len(self._entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          input_image:      [3, H, W]
          target_novel_view: [3, H, W]
          camera_pose:       [4, 4] float32 (world-to-camera or c2w — document in your preprocessing; here 4x4 identity + small nudge for test)
        """
        if self.synthetic:
            g = torch.Generator()
            g.manual_seed(idx * 9973 + 42)
            inp = torch.rand(3, self.image_size, self.image_size, generator=g)
            tgt = torch.rand(3, self.image_size, self.image_size, generator=g)
            pose = torch.eye(4, dtype=torch.float32)
            pose[:3, 3] = torch.randn(3) * 0.01
            return inp, tgt, pose

        entry = self._entries[idx]
        inp_path = self.data_root / entry["input"]
        tgt_path = self.data_root / entry["target"]
        pose_path = self.data_root / entry.get("pose", "")
        inp = _load_image(inp_path, self.image_size)
        tgt = _load_image(tgt_path, self.image_size)
        if pose_path and Path(pose_path).is_file():
            pose_np = np.load(pose_path)
            pose = torch.from_numpy(np.asarray(pose_np, dtype=np.float32))
            if pose.shape == (4, 4):
                pass
            else:
                pose = torch.eye(4, dtype=torch.float32)
        else:
            pose = torch.eye(4, dtype=torch.float32)
        return inp, tgt, pose


def make_dataloader(
    data_root: str,
    batch_size: int,
    image_size: int,
    num_workers: int = 2,
    synthetic: bool = False,
    shuffle: bool = True,
) -> DataLoader:
    ds = ShapeNetVehicleDataset(
        data_root=data_root,
        image_size=image_size,
        synthetic=synthetic,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
