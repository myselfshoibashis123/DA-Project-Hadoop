#!/usr/bin/env python3
"""Download COCO-pretrained Ultralytics YOLO weights into models/.

The COCO model is used (a) for person detection and (b) as the starting point for
fine-tuning the PPE model. The PPE model itself is produced by
scripts/train_ppe_model.py (or shipped as models/ppe_yolo11n_best.pt in the ZIP).

Usage:
    python scripts/download_weights.py                 # yolo11n.pt
    python scripts/download_weights.py --names yolo11n.pt yolo11s.pt
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from computer_vision.config import load_config, resolve_path  # noqa: E402


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", nargs="+", default=[Path(cfg["models"]["person_weights"]).name])
    ap.add_argument("--base-url", default=cfg["models"]["weights_base_url"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    models_dir = resolve_path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        dest = models_dir / name
        if dest.exists() and not args.force:
            print(f"{dest} already exists ({dest.stat().st_size / 1e6:.1f} MB)")
            continue
        url = f"{args.base_url.rstrip('/')}/{name}"
        print(f"Downloading {url}")
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
        print(f"  saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")

    ppe = resolve_path(cfg["models"]["ppe_weights"])
    if not ppe.exists():
        print(f"\nNOTE: PPE weights {ppe} not found. Train them with:\n"
              "  python scripts/download_dataset.py && python scripts/prepare_dataset.py\n"
              "  python scripts/train_ppe_model.py")


if __name__ == "__main__":
    main()
