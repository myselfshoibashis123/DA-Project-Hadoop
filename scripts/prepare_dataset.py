#!/usr/bin/env python3
"""Prepare the Construction-PPE dataset for YOLO training and evaluation.

Steps
  1. Resize every image so its longest side is <= ``dataset.max_side`` (YOLO labels are
     normalised, so they stay valid without changes). This makes CPU training feasible.
  2. Re-split: the ORIGINAL train split is divided into train/val (seeded), and the
     ORIGINAL val split becomes an untouched TEST split used only for final metrics.
  3. Validate every label line (class id range, 5 values, coordinates in [0, 1]).
  4. Write a YOLO dataset YAML and a class-statistics CSV.
  5. Optionally copy a few TEST images to ``data/sample/images`` for demos.

Usage:
    python scripts/prepare_dataset.py [--src data/datasets/raw/construction-ppe] [--samples 8]
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from computer_vision.config import load_config, resolve_path  # noqa: E402

CLASS_NAMES = [
    "helmet", "gloves", "vest", "boots", "goggles", "Person",
    "no_helmet", "no_goggle", "no_gloves", "no_boots", "forklift", "no_vest",
]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resize_image(src: Path, dst: Path, max_side: int) -> None:
    img = cv2.imread(str(src))
    if img is None:
        raise RuntimeError(f"Cannot read image {src}")
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst.with_suffix(".jpg")), img, [cv2.IMWRITE_JPEG_QUALITY, 92])


def clean_labels(src: Path, dst: Path, n_classes: int, counter: Counter) -> int:
    """Copy a label file keeping only valid lines. Returns number of dropped lines."""
    dropped = 0
    lines = []
    if src.exists():
        for raw in src.read_text().splitlines():
            parts = raw.split()
            if not parts:
                continue
            try:
                cls = int(parts[0])
                coords = [float(v) for v in parts[1:5]]
            except ValueError:
                dropped += 1
                continue
            if len(parts) != 5 or not 0 <= cls < n_classes or any(not 0.0 <= c <= 1.0 for c in coords):
                dropped += 1
                continue
            counter[cls] += 1
            lines.append(" ".join([str(cls)] + [f"{c:.6f}" for c in coords]))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + ("\n" if lines else ""))
    return dropped


def main() -> None:
    cfg = load_config()
    dcfg = cfg["dataset"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(resolve_path(dcfg["raw_dir"]) / "construction-ppe"))
    ap.add_argument("--out", default=str(resolve_path(dcfg["prepared_dir"])))
    ap.add_argument("--max-side", type=int, default=dcfg["max_side"])
    ap.add_argument("--samples", type=int, default=8, help="test images to copy to data/sample/images")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not (src / "images" / "train").exists():
        sys.exit(f"Source dataset not found at {src}. Run scripts/download_dataset.py first.")
    if out.exists():
        shutil.rmtree(out)

    rng = random.Random(dcfg["seed"])
    train_imgs = sorted(p for p in (src / "images" / "train").iterdir() if p.suffix.lower() in IMG_EXT)
    test_imgs = sorted(p for p in (src / "images" / "val").iterdir() if p.suffix.lower() in IMG_EXT)
    rng.shuffle(train_imgs)
    n_val = int(round(len(train_imgs) * dcfg["val_fraction"]))
    splits = {"train": train_imgs[n_val:], "val": train_imgs[:n_val], "test": test_imgs}

    stats_rows = []
    total_dropped = 0
    for split, images in splits.items():
        counter: Counter = Counter()
        src_split = "val" if split == "test" else "train"
        for i, img in enumerate(images, 1):
            resize_image(img, out / "images" / split / img.name, args.max_side)
            total_dropped += clean_labels(
                src / "labels" / src_split / (img.stem + ".txt"),
                out / "labels" / split / (img.stem + ".txt"),
                len(CLASS_NAMES), counter,
            )
            if i % 100 == 0 or i == len(images):
                print(f"  {split}: {i}/{len(images)} images", flush=True)
        for cid, name in enumerate(CLASS_NAMES):
            stats_rows.append({"split": split, "class_id": cid, "class": name, "instances": counter[cid]})
        print(f"{split}: {len(images)} images, {sum(counter.values())} boxes")

    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    yaml_path = resolve_path(dcfg["yaml_path"])
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    stats_path = resolve_path("outputs/metrics/dataset_class_distribution.csv")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["split", "class_id", "class", "instances"])
        w.writeheader()
        w.writerows(stats_rows)

    if args.samples > 0:
        sample_dir = resolve_path("data/sample/images")
        sample_dir.mkdir(parents=True, exist_ok=True)
        picks = sorted((out / "images" / "test").iterdir())
        rng.shuffle(picks)
        for p in picks[: args.samples]:
            shutil.copy2(p, sample_dir / p.name)
        print(f"Copied {min(args.samples, len(picks))} test images to {sample_dir}")

    print(f"Dropped invalid label lines: {total_dropped}")
    print(f"Dataset YAML: {yaml_path}\nClass stats:  {stats_path}")


if __name__ == "__main__":
    main()
