#!/usr/bin/env python3
"""
Download vehicle-related 3D models (GLB) from Objaverse 1.0 via the official `objaverse` package.

License (you must comply): dataset overview ODC-By v1.0; each object has its own CC-style
license in metadata — see https://huggingface.co/datasets/allenai/objaverse

This repo's training dataloader expects rendered RGB + manifest.jsonl, not raw GLB.
Use this script to **collect meshes**; add a rendering pipeline (Blender / PyTorch3D / etc.)
to produce `input`/`target` PNGs and a manifest (see data/dataset.py).

Modes:
  - lvis: merge LVIS categories like car_(automobile), pickup_truck, bus_(vehicle), … (~hundreds of GLBs)
  - tags: scan Objaverse metadata for vehicle-like tags (scale with --scan_uids; slower, more models)

Example:
  python scripts/download_objaverse_vehicles.py --out_dir data/objaverse_vehicles --max_objects 200 --mode lvis

Colab cache under /content (optional):
  python scripts/download_objaverse_vehicles.py --objaverse_cache /content/data/.objaverse_cache ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

# Patch cache root *before* any objaverse downloads (default ~/.objaverse).
import objaverse as ov  # noqa: E402

LVIS_VEHICLE_KEYS = [
    "car_(automobile)",
    "bus_(vehicle)",
    "minivan",
    "pickup_truck",
    "motorcycle",
    "motor_vehicle",
    "school_bus",
    "tow_truck",
    "garbage_truck",
    "camper_(vehicle)",
    "golfcart",
    "race_car",
]

TAG_HINTS = (
    "car",
    "truck",
    "vehicle",
    "automobile",
    "bus",
    "van",
    "motorcycle",
    "sedan",
    "suv",
    "pickup",
    "coupe",
    "wagon",
    "jeep",
    "motorbike",
    "limousine",
    "taxi",
    "van",
)


def _vehicle_from_tags(meta: Dict[str, Any]) -> bool:
    tags = meta.get("tags") or []
    parts: List[str] = []
    for t in tags:
        if isinstance(t, dict):
            parts.append(str(t.get("name", "")).lower())
            parts.append(str(t.get("slug", "")).lower())
    blob = " ".join(parts)
    return any(h in blob for h in TAG_HINTS)


def _uids_from_lvis() -> List[str]:
    la = ov.load_lvis_annotations()
    uids: Set[str] = set()
    for k in LVIS_VEHICLE_KEYS:
        for u in la.get(k, []):
            uids.add(u)
    return sorted(uids)


def _uids_from_tag_scan(all_uids: List[str], scan_limit: int, max_objects: int) -> List[str]:
    found: List[str] = []
    batch = 512
    limit = min(len(all_uids), scan_limit)
    for start in range(0, limit, batch):
        chunk = all_uids[start : start + batch]
        ann = ov.load_annotations(chunk)
        for uid, meta in ann.items():
            if _vehicle_from_tags(meta):
                found.append(uid)
            if len(found) >= max_objects:
                return found[:max_objects]
    return found[:max_objects]


def _write_index(out_dir: Path, uid_to_path: Dict[str, str]) -> None:
    manifest = []
    for uid, glb in sorted(uid_to_path.items()):
        manifest.append({"uid": uid, "glb": glb, "license": "see Objaverse metadata for this uid"})
    (out_dir / "glb_index.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Download Objaverse vehicle GLBs (licensed use required).")
    p.add_argument("--out_dir", type=str, required=True, help="Directory for glb_index.json + symlinks")
    p.add_argument("--max_objects", type=int, default=400, help="Cap on number of GLBs to download")
    p.add_argument("--mode", choices=("lvis", "tags", "both"), default="lvis")
    p.add_argument(
        "--scan_uids",
        type=int,
        default=50_000,
        help="For mode=tags/both: how many Objaverse uids (in order) to scan for vehicle tags",
    )
    p.add_argument(
        "--objaverse_cache",
        type=str,
        default="",
        help="If set, store Hugging Face cache metadata/glbs here instead of ~/.objaverse",
    )
    p.add_argument("--download_processes", type=int, default=4)
    args = p.parse_args()

    if args.objaverse_cache:
        cache = Path(args.objaverse_cache).resolve()
        cache.mkdir(parents=True, exist_ok=True)
        ov.BASE_PATH = str(cache)
        ov._VERSIONED_PATH = os.path.join(ov.BASE_PATH, "hf-objaverse-v1")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_uids = ov.load_uids()
    picked: List[str] = []
    if args.mode == "lvis":
        picked = _uids_from_lvis()
    elif args.mode == "tags":
        picked = _uids_from_tag_scan(all_uids, args.scan_uids, args.max_objects)
    else:
        lvis_list = _uids_from_lvis()
        tag_list = _uids_from_tag_scan(all_uids, args.scan_uids, args.max_objects)
        picked = []
        seen: Set[str] = set()
        for u in lvis_list + tag_list:
            if u in seen:
                continue
            seen.add(u)
            picked.append(u)
            if len(picked) >= args.max_objects:
                break

    picked = picked[: args.max_objects]
    if not picked:
        print("No UIDs selected.", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading {len(picked)} objects (mode={args.mode}) …")
    paths = ov.load_objects(picked, download_processes=max(1, args.download_processes))

    # Symlink (or copy) into out_dir/glbs/{uid}.glb for a stable layout
    glb_dir = out_dir / "glbs"
    glb_dir.mkdir(exist_ok=True)
    index: Dict[str, str] = {}
    for uid, src in paths.items():
        src_p = Path(src)
        if not src_p.is_file():
            continue
        dst = glb_dir / f"{uid}.glb"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(src_p.resolve())
        except OSError:
            import shutil

            shutil.copy2(src_p, dst)
        index[uid] = str(dst)

    _write_index(out_dir, index)
    print(f"Wrote {len(index)} models under {glb_dir} and {out_dir / 'glb_index.json'}")
    print("Next: rasterize each GLB to multi-view PNGs + build manifest.jsonl (see data/dataset.py).")


if __name__ == "__main__":
    main()
