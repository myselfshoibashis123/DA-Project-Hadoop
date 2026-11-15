#!/usr/bin/env python3
"""Run the safety-analytics computer-vision pipeline on images, a folder, or a video.

Examples:
    python scripts/run_pipeline.py --input data/sample/images --camera cam_A
    python scripts/run_pipeline.py --input data/sample/videos/site_walkthrough.mp4 --camera cam_B
    python scripts/run_pipeline.py --input some.jpg --out outputs/single --fresh
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from computer_vision.config import load_config, merged, resolve_path  # noqa: E402
from computer_vision.pipeline import IMAGE_EXT, VIDEO_EXT, SafetyPipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="image file, folder of images, or video file")
    ap.add_argument("--camera", default="cam_01", help="camera / location identifier")
    ap.add_argument("--out", default="outputs/pipeline", help="output directory")
    ap.add_argument("--config", default=None)
    ap.add_argument("--start-time", default=None,
                    help="ISO time of the first frame, e.g. 2026-11-12T08:00:00 (default: now)")
    ap.add_argument("--interval", type=float, default=60.0,
                    help="seconds between consecutive still images (assigned timestamps)")
    ap.add_argument("--stride", type=int, default=None, help="analyse every Nth video frame")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--person-source", choices=["ppe_model", "coco", "both"], default=None)
    ap.add_argument("--fresh", action="store_true", help="delete the output directory first")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.person_source:
        cfg = merged(cfg, {"detection": {"person_source": args.person_source}})
    out_dir = resolve_path(args.out)
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    start = datetime.fromisoformat(args.start_time) if args.start_time else None

    inp = resolve_path(args.input)
    pipe = SafetyPipeline(cfg)
    if inp.is_dir():
        imgs = [p for p in inp.iterdir() if p.suffix.lower() in IMAGE_EXT]
        vids = [p for p in inp.iterdir() if p.suffix.lower() in VIDEO_EXT]
        if not imgs and not vids:
            sys.exit(f"No images or videos found in {inp}")
        summaries = []
        if imgs:
            print(f"Analysing {len(imgs)} images from {inp} as {args.camera}")
            summaries.append(pipe.run_images(imgs, out_dir, args.camera, start, args.interval))
        for v in sorted(vids):
            print(f"Analysing video {v.name} as {args.camera}")
            summaries.append(pipe.run_video(v, out_dir, args.camera, start,
                                            args.stride or cfg["video"]["frame_stride"],
                                            args.max_frames if args.max_frames is not None
                                            else cfg["video"]["max_frames"]))
    elif inp.suffix.lower() in IMAGE_EXT:
        summaries = [pipe.run_images([inp], out_dir, args.camera, start, args.interval)]
    elif inp.suffix.lower() in VIDEO_EXT:
        print(f"Analysing video {inp.name} as {args.camera}")
        summaries = [pipe.run_video(inp, out_dir, args.camera, start,
                                    args.stride or cfg["video"]["frame_stride"],
                                    args.max_frames if args.max_frames is not None
                                    else cfg["video"]["max_frames"])]
    else:
        sys.exit(f"Unsupported input: {inp}")

    for s in summaries:
        print(json.dumps(s, indent=2))
    print(f"\nOutputs written to {out_dir}")


if __name__ == "__main__":
    main()
