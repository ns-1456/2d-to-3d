"""
Stage dataset archives from slow storage (Drive) to fast Colab local NVMe.

Usage (typically from `train.py --stage_data` or the notebook):
  unzip_shapenet_to_local(zip_path: Path on Drive or URL-ready path, dest: /content/data/shapenet_vehicle)
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


def unzip_to_local(
    archive_path: str,
    dest_dir: str,
    overwrite: bool = False,
) -> Path:
    """
    Extract `archive_path` (.zip) into `dest_dir` (creates parents).

    Returns the resolved destination path. Prefer calling this once per session
    before training; dataloaders should only read under `dest_dir`.
    """
    src = Path(archive_path)
    dst = Path(dest_dir)
    dst.mkdir(parents=True, exist_ok=True)
    if not src.is_file():
        raise FileNotFoundError(f"Archive not found: {src}")

    if overwrite and dst.exists():
        shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(src, "r") as zf:
        zf.extractall(dst)
    return dst.resolve()


def copy_tree_to_local(src: str, dest: str) -> Path:
    """Optional: shutil.copytree from Drive folder to /content/data (for non-zip dumps)."""
    s, d = Path(src), Path(dest)
    if not s.is_dir():
        raise NotADirectoryError(str(s))
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(s, d)
    return d.resolve()
