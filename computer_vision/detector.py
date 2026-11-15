"""YOLO-based detection of PPE items, persons and forklifts.

Two models can be combined:
  * a PPE model fine-tuned on Construction-PPE (12 classes, incl. explicit violation
    classes such as ``no_helmet``) - see scripts/train_ppe_model.py
  * a COCO-pretrained YOLO model used as an additional person detector.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import resolve_path


@dataclass
class Detection:
    """A single detected object. ``bbox`` is (x1, y1, x2, y2) in pixels."""

    label: str
    confidence: float
    bbox: tuple
    source: str = "ppe_model"

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["bbox"] = [round(float(v), 1) for v in self.bbox]
        d["confidence"] = round(float(self.confidence), 4)
        return d


def iou(a: tuple, b: tuple) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def _covered_fraction(box: tuple, other: tuple) -> float:
    """Fraction of ``box``'s area that lies inside ``other``."""
    ix1, iy1 = max(box[0], other[0]), max(box[1], other[1])
    ix2, iy2 = min(box[2], other[2]), min(box[3], other[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / max(1e-9, (box[2] - box[0]) * (box[3] - box[1]))


def merge_person_detections(dets: List[Detection], iou_thr: float,
                            coverage_thr: float = 0.7) -> List[Detection]:
    """Greedy merge of person boxes coming from different models.

    A box is dropped when it overlaps an already-kept (higher-confidence) box with
    IoU >= ``iou_thr`` (classic NMS), or when the kept boxes together cover at least
    ``coverage_thr`` of its area - this removes a low-confidence box spanning two
    neighbouring workers while keeping a partly occluded worker standing behind someone.
    """
    kept: List[Detection] = []
    for d in sorted(dets, key=lambda x: x.confidence, reverse=True):
        if any(iou(d.bbox, k.bbox) >= iou_thr for k in kept):
            continue
        if coverage_thr and sum(_covered_fraction(d.bbox, k.bbox) for k in kept) >= coverage_thr:
            continue
        kept.append(d)
    return kept


class SafetyDetector:
    """Runs the PPE model (and optionally a COCO person model) on an image."""

    def __init__(self, cfg: Dict[str, Any]):
        from ultralytics import YOLO  # imported lazily so pure-logic modules work without torch

        self.cfg = cfg
        mcfg, dcfg = cfg["models"], cfg["detection"]
        self.person_class = cfg["classes"]["person"]
        self.person_source = dcfg.get("person_source", "both")
        self.device = mcfg.get("device", "cpu")
        self.imgsz = int(mcfg.get("imgsz", 640))

        ppe_path = resolve_path(mcfg["ppe_weights"])
        if not ppe_path.exists():
            raise FileNotFoundError(
                f"PPE weights not found at {ppe_path}. Train them with "
                "scripts/train_ppe_model.py (see README, section 'Model weights')."
            )
        self.ppe_model = YOLO(str(ppe_path))
        self.ppe_names: Dict[int, str] = dict(self.ppe_model.names)

        self.person_model = None
        if self.person_source in ("coco", "both"):
            person_path = resolve_path(mcfg["person_weights"])
            if not person_path.exists():
                raise FileNotFoundError(
                    f"Person weights not found at {person_path}. Run scripts/download_weights.py"
                )
            self.person_model = YOLO(str(person_path))

    @staticmethod
    def _to_detections(result, names: Dict[int, str], source: str,
                       keep: Optional[set] = None) -> List[Detection]:
        out: List[Detection] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for b, c, k in zip(xyxy, conf, cls):
            label = names[int(k)]
            if keep is not None and label not in keep:
                continue
            out.append(Detection(label=label, confidence=float(c),
                                 bbox=tuple(float(v) for v in b), source=source))
        return out

    def detect(self, image: np.ndarray) -> Dict[str, List[Detection]]:
        """Return {'persons': [...], 'objects': [...]} for a BGR image."""
        dcfg = self.cfg["detection"]
        ppe_res = self.ppe_model.predict(image, conf=dcfg["ppe_conf"], iou=dcfg["iou"],
                                         imgsz=self.imgsz, device=self.device, verbose=False)[0]
        ppe_dets = self._to_detections(ppe_res, self.ppe_names, "ppe_model")

        persons = [d for d in ppe_dets if d.label == self.person_class]
        objects = [d for d in ppe_dets if d.label != self.person_class]
        if self.person_source == "coco":
            persons = []
        if self.person_model is not None:
            coco_res = self.person_model.predict(image, conf=dcfg["person_conf"], iou=dcfg["iou"],
                                                 imgsz=self.imgsz, device=self.device,
                                                 classes=[0], verbose=False)[0]
            coco_persons = self._to_detections(coco_res, dict(self.person_model.names), "coco_model",
                                               keep={"person"})
            for d in coco_persons:
                d.label = self.person_class
            persons = merge_person_detections(persons + coco_persons, dcfg["person_merge_iou"],
                                              dcfg.get("person_merge_coverage", 0.7))
        return {"persons": persons, "objects": objects}


def weights_available(cfg: Dict[str, Any]) -> bool:
    """True when the PPE weights (and person weights if needed) exist on disk."""
    m = cfg["models"]
    need = [m["ppe_weights"]]
    if cfg["detection"].get("person_source", "both") in ("coco", "both"):
        need.append(m["person_weights"])
    return all(Path(resolve_path(p)).exists() for p in need)
