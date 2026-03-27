#!/usr/bin/env python3
"""Create data/demo/ with manifest.jsonl + tiny PNGs for local smoke tests (real files on disk)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "demo"
    root.mkdir(parents=True, exist_ok=True)
    img_dir = root / "images"
    img_dir.mkdir(exist_ok=True)

    def save_rgb(name: str, rgb: tuple[int, int, int]) -> None:
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[:, :] = rgb
        Image.fromarray(arr).save(img_dir / name)

    save_rgb("s0_in.png", (40, 60, 200))
    save_rgb("s0_tgt.png", (30, 180, 50))
    save_rgb("s1_in.png", (200, 50, 50))
    save_rgb("s1_tgt.png", (50, 200, 200))

    entries = [
        {"model_id": "demo0", "input": "images/s0_in.png", "target": "images/s0_tgt.png"},
        {"model_id": "demo1", "input": "images/s1_in.png", "target": "images/s1_tgt.png"},
    ]
    manifest = root / "manifest.jsonl"
    with open(manifest, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    print("Wrote", manifest, "and images under", img_dir)


if __name__ == "__main__":
    main()
