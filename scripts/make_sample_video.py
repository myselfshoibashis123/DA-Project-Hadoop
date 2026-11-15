#!/usr/bin/env python3
"""Build a short demo video from still images (slow pan/zoom over each image).

IMPORTANT: the result is a SYNTHETIC slideshow assembled from dataset photos so the
video path of the pipeline can be demonstrated without distributing third-party CCTV
footage. It is not real surveillance footage. To use real footage, pass any .mp4 to
scripts/run_pipeline.py instead.

Usage:
    python scripts/make_sample_video.py --images data/sample/images \
        --out data/sample/videos/site_walkthrough.mp4 --seconds-per-image 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from computer_vision.config import resolve_path  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png"}


def letterbox(img: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = int(iw * s), int(ih * s)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default="data/sample/images")
    ap.add_argument("--out", default="data/sample/videos/site_walkthrough.mp4")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--seconds-per-image", type=float, default=2.0)
    ap.add_argument("--zoom", type=float, default=0.12, help="max zoom-in per image (0 = static)")
    args = ap.parse_args()

    imgs = sorted(p for p in resolve_path(args.images).iterdir() if p.suffix.lower() in IMG_EXT)
    if not imgs:
        sys.exit(f"No images in {args.images}")
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    n = max(1, int(args.fps * args.seconds_per_image))
    for p in imgs:
        base = letterbox(cv2.imread(str(p)), args.width, args.height)
        for k in range(n):
            z = 1.0 + args.zoom * k / max(1, n - 1)
            cw, ch = int(args.width / z), int(args.height / z)
            x0, y0 = (args.width - cw) // 2, (args.height - ch) // 2
            frame = cv2.resize(base[y0:y0 + ch, x0:x0 + cw], (args.width, args.height))
            writer.write(frame)
    writer.release()
    print(f"Wrote {out} ({len(imgs)} images x {n} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
