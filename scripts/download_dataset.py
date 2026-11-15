#!/usr/bin/env python3
"""Download and extract the Ultralytics Construction-PPE dataset (CC BY 4.0).

Usage:
    python scripts/download_dataset.py            # download + extract
    python scripts/download_dataset.py --zip file.zip   # extract an already-downloaded zip
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from computer_vision.config import load_config, resolve_path  # noqa: E402


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}\n  -> {dest}")

    def _progress(blocks: int, block_size: int, total: int) -> None:
        if total > 0:
            done = min(blocks * block_size, total)
            pct = 100.0 * done / total
            sys.stdout.write(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)")
            sys.stdout.flush()

    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp, reporthook=_progress)
    tmp.rename(dest)
    print()


def extract(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    # The archive contains a single top-level folder "construction-ppe"
    candidates = [p for p in out_dir.iterdir() if p.is_dir() and (p / "images").exists()]
    if not candidates:
        raise RuntimeError("Extracted archive does not contain an images/ folder")
    return candidates[0]


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=cfg["dataset"]["url"])
    ap.add_argument("--zip", help="use an existing zip instead of downloading")
    ap.add_argument("--keep-zip", action="store_true", help="do not delete the zip after extraction")
    args = ap.parse_args()

    raw_dir = resolve_path(cfg["dataset"]["raw_dir"])
    if args.zip:
        zip_path = Path(args.zip)
    else:
        zip_path = raw_dir / "construction-ppe.zip"
        if not zip_path.exists():
            download(args.url, zip_path)
    root = extract(zip_path, raw_dir)
    if not args.zip and not args.keep_zip:
        zip_path.unlink(missing_ok=True)
    n_imgs = sum(1 for _ in root.glob("images/*/*"))
    print(f"Done: {n_imgs} images in {root}")
    print("Next step: python scripts/prepare_dataset.py")


if __name__ == "__main__":
    main()
