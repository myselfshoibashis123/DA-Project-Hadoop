#!/usr/bin/env python3
"""Fine-tune a COCO-pretrained YOLO model on the prepared Construction-PPE dataset.

Usage examples:
    # quick CPU run (what the bundled weights were produced with - see docs/RESULTS.md)
    python scripts/train_ppe_model.py --epochs 10 --imgsz 416 --batch 8

    # recommended on a GPU
    python scripts/train_ppe_model.py --epochs 100 --imgsz 640 --batch 16 --device 0

The best checkpoint is copied to ``models.ppe_weights`` (config.yaml) and the training
arguments are stored next to it as JSON, so results can be traced back to a run.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from computer_vision.config import load_config, resolve_path  # noqa: E402


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(resolve_path(cfg["dataset"]["yaml_path"])))
    ap.add_argument("--base", default=str(resolve_path(cfg["models"]["person_weights"])),
                    help="pretrained starting weights (COCO yolo11n by default)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default=cfg["models"]["device"])
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of train set to use")
    ap.add_argument("--name", default="ppe_yolo11n")
    ap.add_argument("--out", default=str(resolve_path(cfg["models"]["ppe_weights"])))
    args = ap.parse_args()

    from ultralytics import YOLO

    if not Path(args.data).exists():
        sys.exit(f"{args.data} not found - run scripts/prepare_dataset.py first")
    base = args.base
    if not Path(base).exists():
        print(f"{base} not found locally; Ultralytics will try to download {Path(base).name}")
        base = Path(base).name

    model = YOLO(base)
    t0 = time.time()
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        workers=args.workers, device=args.device, patience=args.patience,
        fraction=args.fraction, project=str(resolve_path("outputs/training")),
        name=args.name, exist_ok=True, plots=True, seed=0, deterministic=True,
        verbose=True,
    )
    elapsed = time.time() - t0
    run_dir = resolve_path("outputs/training") / args.name
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        sys.exit(f"Training finished but {best} is missing")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    info = {k: v for k, v in vars(args).items()}
    info.update({"training_seconds": round(elapsed, 1), "run_dir": str(run_dir)})
    out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(f"Best weights copied to {out} ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
