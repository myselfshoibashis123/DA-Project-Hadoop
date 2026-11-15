# Results

All numbers below were produced by the scripts in this repository on the date of the build
(13 Nov 2026). Nothing is estimated. Re-running the scripts regenerates every file.

## 1. Training

| Item | Value |
|---|---|
| Model | YOLO11n, initialised from COCO weights |
| Data | prepared Construction-PPE: 680 train / 120 val images, 416 px |
| Hardware | CPU only (one shared core), batch 8 |
| Epochs | 5 (stage 1, run interrupted by the build environment) + 12 (stage 2, resumed from stage-1 `last.pt`) = **17** |
| Best validation mAP@0.5 / mAP@0.5:0.95 | **0.680 / 0.470** (stage 2) |
| Stage-2 training time | 25.0 min |

Curves: `outputs/metrics/training/stage1_results.csv`, `stage2_results.csv`, `stage2_results.png`.
Full provenance: `models/ppe_yolo11n_best.json`.

## 2. Detection metrics – held-out TEST split (200 images, never used for training)

Produced by `python scripts/evaluate_model.py --imgsz 416` → `outputs/metrics/test_metrics.json`.

| Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| 0.825 | 0.572 | 0.676 | **0.644** | 0.431 |

Per class (`outputs/metrics/per_class_metrics.csv`):

| Class | P | R | F1 | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| helmet | 0.913 | 0.622 | 0.740 | 0.681 | 0.518 |
| gloves | 0.855 | 0.640 | 0.732 | 0.708 | 0.470 |
| vest | 0.926 | 0.762 | 0.836 | 0.828 | 0.617 |
| boots | 0.768 | 0.580 | 0.661 | 0.631 | 0.454 |
| goggles | 0.809 | 0.748 | 0.777 | 0.845 | 0.398 |
| Person | 0.913 | 0.811 | 0.859 | 0.878 | 0.688 |
| no_helmet | 0.782 | 0.402 | 0.531 | 0.485 | 0.288 |
| no_goggle | 0.710 | 0.060 | 0.111 | 0.082 | 0.023 |
| no_gloves | 0.796 | 0.424 | 0.553 | 0.516 | 0.298 |
| no_boots | 0.832 | 0.527 | 0.645 | 0.669 | 0.463 |
| forklift | 0.688 | 0.781 | 0.732 | 0.780 | 0.548 |
| no_vest | 0.904 | 0.508 | 0.651 | 0.622 | 0.412 |

(P/R are reported by Ultralytics at the confidence that maximises mean F1.)

## 3. Violation detection at image level – TEST split

Question asked per image and per violation type: *does the image contain at least one
such violation?* Ground truth = at least one labelled `no_*` box; prediction = at least one
`no_*` detection with confidence ≥ 0.30 (`outputs/metrics/violation_image_level.csv`).

| Violation | Images with violation | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| no_helmet | 91 | 64 | 3 | 27 | 0.955 | 0.703 | 0.810 |
| no_vest | 84 | 62 | 2 | 22 | 0.969 | 0.738 | 0.838 |
| no_gloves | 94 | 60 | 5 | 34 | 0.923 | 0.638 | 0.755 |
| no_boots | 71 | 45 | 7 | 26 | 0.865 | 0.634 | 0.732 |
| no_goggle | 56 | 6 | 2 | 50 | 0.750 | 0.107 | 0.188 |

**Interpretation.** When the system raises a helmet, vest or gloves alert it is right more
than 90 % of the time, but it misses roughly 26–36 % of violating images. Missing goggles
are effectively **not detectable** with this model (recall 0.11): `no_goggle` is one of the
rarest training classes (~230 train boxes) and the objects are tiny at 416 px. Longer GPU
training at 640 px is the first thing to try (see README §16).

## 4. End-to-end run on the bundled samples

`bash scripts/run_end_to_end.sh` (local-simulation mode) on 8 still images (camera
`cam_A`, assigned timestamps 15 min apart from 08:00) and the 16-second synthetic demo
video (camera `cam_B`, every 5th frame → 32 frames).

> These are **illustrative** outputs on 8 images. The video re-shows the same 8 photos, so
> its counts largely repeat cam_A's and must not be read as plant statistics.

MapReduce aggregates (`outputs/analytics/analytics_summary.json`, mode **local-simulation**):

| Metric | Value |
|---|---|
| Frames analysed | 40 |
| Worker observations | 65 (cam_A 11, cam_B 54) |
| Compliant observations | 9 |
| Total violations | 82 |
| By category | forklift proximity 27 · missing gloves 18 · missing helmet 15 · missing vest 10 · missing boots 7 · restricted zone 5 |
| By risk level (observations) | High 9 · Medium 36 · Low 12 · None 9 |
| Average risk score | 2.894 |

Consistency check: every MapReduce total equals the sum of the per-run pipeline summaries
(`outputs/run_cam_*/reports/summary.json`), and the categories match exactly.

Raw reducer output: `outputs/analytics/mapreduce_part-00000.txt`.

## 5. Qualitative observations from the samples
* A worker missing helmet, vest, gloves and boots scores 3+2+1+1 = 7 → **High**.
* A worker standing directly in front of a forklift is flagged for proximity (Medium) even
  though all PPE is worn.
* In the warehouse image, small distant workers are detected and correctly placed near
  forklifts, but their PPE is mostly `unknown` – too small to classify at 416 px.
* An early version double-counted two adjacent workers because of a low-confidence box
  spanning both; the coverage-based person merge (README §8.1) fixed this.
