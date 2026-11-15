#!/usr/bin/env python3
"""Evaluate the PPE model on the held-out TEST split (never used for training/selection).

Produces in outputs/metrics/:
  test_metrics.json          overall precision, recall, F1, mAP@0.5, mAP@0.5:0.95
  per_class_metrics.csv      the same metrics per class
  violation_image_level.csv  image-level detection of each violation category
                             (does the image contain >= 1 'no_helmet', etc.?)
  plots/                     confusion matrix and PR curves produced by Ultralytics

Usage:
    python scripts/evaluate_model.py [--weights models/ppe_yolo11n_best.pt] [--imgsz 416]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from computer_vision.config import load_config, resolve_path  # noqa: E402


def f1(p: float, r: float) -> float:
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def image_level_violation_eval(model, data_cfg: dict, cfg: dict, imgsz: int, conf: float) -> list:
    """Per violation class: TP/FP/FN at image level (presence of >= 1 instance)."""
    root = Path(data_cfg["path"])
    names = {int(k): v for k, v in data_cfg["names"].items()}
    neg_classes = [v["negative"] for v in cfg["classes"]["ppe_items"].values()]
    neg_ids = {cid: n for cid, n in names.items() if n in neg_classes}
    counts = {n: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for n in neg_ids.values()}
    images = sorted((root / data_cfg["test"]).iterdir())
    for img in images:
        lbl = root / "labels" / "test" / (img.stem + ".txt")
        gt = set()
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                if line.strip():
                    c = int(line.split()[0])
                    if c in neg_ids:
                        gt.add(neg_ids[c])
        res = model.predict(str(img), imgsz=imgsz, conf=conf, verbose=False, device=cfg["models"]["device"])[0]
        pred = {names[int(c)] for c in res.boxes.cls.cpu().numpy()} & set(neg_ids.values())
        for n in neg_ids.values():
            key = ("tp" if n in pred else "fn") if n in gt else ("fp" if n in pred else "tn")
            counts[n][key] += 1
    rows = []
    for n, c in counts.items():
        p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        rows.append({"violation_class": n, **c, "images_with_violation": c["tp"] + c["fn"],
                     "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1(p, r), 4)})
    return rows


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(resolve_path(cfg["models"]["ppe_weights"])))
    ap.add_argument("--data", default=str(resolve_path(cfg["dataset"]["yaml_path"])))
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--conf", type=float, default=cfg["detection"]["ppe_conf"],
                    help="confidence threshold for the image-level violation evaluation")
    args = ap.parse_args()

    from ultralytics import YOLO

    if not Path(args.weights).exists():
        sys.exit(f"Weights {args.weights} not found")
    if not Path(args.data).exists():
        sys.exit(f"{args.data} not found - run scripts/download_dataset.py and prepare_dataset.py")
    data_cfg = yaml.safe_load(Path(args.data).read_text())
    out_dir = resolve_path("outputs/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    val_dir = resolve_path("outputs/evaluation")
    metrics = model.val(data=args.data, split="test", imgsz=args.imgsz, batch=8,
                        device=cfg["models"]["device"], project=str(val_dir), name="test",
                        exist_ok=True, plots=True, verbose=False)
    box = metrics.box
    names = model.names
    overall = {
        "weights": Path(args.weights).name, "split": "test", "imgsz": args.imgsz,
        "images": len(list((Path(data_cfg["path"]) / data_cfg["test"]).iterdir())),
        "precision": round(float(box.mp), 4), "recall": round(float(box.mr), 4),
        "f1": round(f1(float(box.mp), float(box.mr)), 4),
        "mAP50": round(float(box.map50), 4), "mAP50_95": round(float(box.map), 4),
    }
    rows = []
    for i, cid in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        rows.append({"class_id": int(cid), "class": names[int(cid)], "precision": round(float(p), 4),
                     "recall": round(float(r), 4), "f1": round(f1(float(p), float(r)), 4),
                     "mAP50": round(float(ap50), 4), "mAP50_95": round(float(ap), 4)})
    with open(out_dir / "per_class_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("Running image-level violation evaluation ...")
    vrows = image_level_violation_eval(model, data_cfg, cfg, args.imgsz, args.conf)
    with open(out_dir / "violation_image_level.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(vrows[0].keys()))
        w.writeheader()
        w.writerows(vrows)
    overall["image_level_violation_conf"] = args.conf
    (out_dir / "test_metrics.json").write_text(json.dumps(overall, indent=2))

    plots = out_dir / "plots"
    plots.mkdir(exist_ok=True)
    for f in (val_dir / "test").glob("*.png"):
        shutil.copy2(f, plots / f.name)

    print(json.dumps(overall, indent=2))
    for r in rows:
        print(f"  {r['class']:<10} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
              f"mAP50={r['mAP50']:.3f} mAP50-95={r['mAP50_95']:.3f}")
    print("Image-level violation detection:")
    for r in vrows:
        print(f"  {r['violation_class']:<10} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
              f"(images with violation: {r['images_with_violation']})")


if __name__ == "__main__":
    main()
