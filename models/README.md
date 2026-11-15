# Model weights

| File | What it is | How to get it |
|---|---|---|
| `ppe_yolo11n_best.pt` | YOLO11n fine-tuned on Construction-PPE (12 classes). **Included in the ZIP** (~5.5 MB). Trained on CPU with a short budget – see `docs/RESULTS.md` for its real test metrics. | Re-train: `python scripts/train_ppe_model.py` |
| `ppe_yolo11n_best.json` | Training arguments of the run that produced the weights | written by the training script |
| `yolo11n.pt` | Ultralytics YOLO11n pretrained on COCO, used for person detection and as the fine-tuning start point. **Not included.** | `python scripts/download_weights.py` |

The COCO weights are downloaded from
`https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt`.

To use person detection from the PPE model only (no COCO weights needed), set
`detection.person_source: ppe_model` in `config/config.yaml` or pass
`--person-source ppe_model` to `scripts/run_pipeline.py`.

Ultralytics models and code are licensed AGPL-3.0; check that licence before any
non-academic use.
