"""
Fetch a dataset `.zip` inside Colab before unzipping to fast local disk.

Supports:
  - `dataset_download_url`: direct HTTPS link to a zip
  - `dataset_gdrive_id`: Google Drive *file* id (from a share link)
  - `dataset_archive`: path to an already-mounted zip (e.g. on `/content/drive/...`)

ShapeNet and many research datasets require you to obtain files under their license;
this module only moves bytes — you supply the URL or Drive id for **your** copy.
"""

from __future__ import annotations

import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional


def _download_http(url: str, dest: Path, chunk_size: int = 1 << 20) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp, open(dest, "wb") as out:
        while True:
            block = resp.read(chunk_size)
            if not block:
                break
            out.write(block)
    return dest


def _download_gdrive(file_id: str, dest: Path) -> Path:
    import gdown

    dest.parent.mkdir(parents=True, exist_ok=True)
    # Large file confirm handled by gdown when needed
    gdown.download(id=file_id, output=str(dest), quiet=False)
    if not dest.is_file() or dest.stat().st_size < 10:
        raise RuntimeError(
            f"gdown failed or file empty: {dest}. Check sharing ('Anyone with link') or file id."
        )
    return dest


def copy_drive_zip_to_cache(src_drive_zip: str, cache_dir: str) -> str:
    """Copy a zip from `/content/drive/...` into local cache (one-time before unzip)."""
    s, d = Path(src_drive_zip), Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    dest = d / s.name
    shutil.copy2(s, dest)
    return str(dest)


def resolve_local_zip_path(cfg: Dict[str, Any]) -> Optional[str]:
    """
    Returns a path string to a local .zip, or None if nothing resolved.

    Priority:
      1. dataset_gdrive_id -> download to download_cache_dir
      2. dataset_download_url -> download to download_cache_dir
      3. dataset_drive_zip -> copy from mounted Drive into cache (one-time fast extract source)
      4. dataset_archive -> return if path exists on disk
    """
    cache = Path(cfg.get("download_cache_dir") or "/content/data/_download_cache")
    cache.mkdir(parents=True, exist_ok=True)

    gid = str(cfg.get("dataset_gdrive_id") or "").strip()
    url = str(cfg.get("dataset_download_url") or "").strip()
    drive_zip = str(cfg.get("dataset_drive_zip") or "").strip()
    archive = str(cfg.get("dataset_archive") or "").strip()

    if gid:
        target = cache / "dataset_gdrive.zip"
        _download_gdrive(gid, target)
        return str(target)

    if url:
        target = cache / "dataset_http.zip"
        try:
            _download_http(url, target)
        except urllib.error.URLError as e:
            raise RuntimeError(f"HTTP download failed for {url!r}: {e}") from e
        if not target.is_file() or target.stat().st_size < 10:
            raise RuntimeError(f"Downloaded file missing or too small: {target}")
        return str(target)

    if drive_zip:
        p = Path(drive_zip)
        if p.is_file():
            return copy_drive_zip_to_cache(drive_zip, str(cache))

    if archive:
        p = Path(archive)
        if p.is_file():
            return str(p.resolve())

    return None
