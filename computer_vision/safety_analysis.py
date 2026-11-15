"""Turns raw detections into per-worker safety assessments.

For every detected person we determine, for each PPE item, whether it is
  * "worn"    - a positive PPE box (e.g. ``helmet``) is attached to the person,
  * "missing" - a negative box (e.g. ``no_helmet``) is attached to the person, or
  * "unknown" - neither was detected (item occluded / out of frame / missed).
Only REQUIRED items that are "missing" become violations (see config ppe_policy).
Restricted-zone entry and forklift proximity are added as further conditions, and the
combined conditions are scored by :mod:`computer_vision.risk`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .detector import Detection
from .risk import assess
from .zones import Zone, foot_point, zones_for_box


def overlap_ratio(inner: tuple, outer: tuple) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer``."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1e-9, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


def associate(persons: List[Detection], objects: List[Detection],
              min_overlap: float) -> Tuple[Dict[int, List[Detection]], List[Detection]]:
    """Attach each object to the person box that contains most of it.

    Returns (mapping person-index -> objects, unassigned objects).
    """
    mapping: Dict[int, List[Detection]] = {i: [] for i in range(len(persons))}
    unassigned: List[Detection] = []
    for obj in objects:
        best_i, best_r = None, 0.0
        for i, p in enumerate(persons):
            r = overlap_ratio(obj.bbox, p.bbox)
            # tie-break towards the smaller person box (the closer, more specific match)
            if r > best_r + 1e-6 or (best_i is not None and abs(r - best_r) <= 1e-6
                                     and p.area < persons[best_i].area):
                best_i, best_r = i, r
        if best_i is not None and best_r >= min_overlap:
            mapping[best_i].append(obj)
        else:
            unassigned.append(obj)
    return mapping, unassigned


def ppe_status(attached: List[Detection], cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return {item: {"status": worn|missing|unknown, "confidence": float|None}}."""
    status: Dict[str, Dict[str, Any]] = {}
    unknown_as_missing = set(cfg["ppe_policy"].get("treat_unknown_as_missing", []) or [])
    for item, names in cfg["classes"]["ppe_items"].items():
        pos = [d.confidence for d in attached if d.label == names["positive"]]
        neg = [d.confidence for d in attached if d.label == names["negative"]]
        if pos or neg:
            best_pos, best_neg = max(pos, default=0.0), max(neg, default=0.0)
            if best_neg > best_pos:
                status[item] = {"status": "missing", "confidence": round(best_neg, 4), "inferred": False}
            else:
                status[item] = {"status": "worn", "confidence": round(best_pos, 4), "inferred": False}
        elif item in unknown_as_missing:
            status[item] = {"status": "missing", "confidence": None, "inferred": True}
        else:
            status[item] = {"status": "unknown", "confidence": None, "inferred": False}
    return status


def box_gap(point: Tuple[float, float], box: tuple) -> float:
    """Euclidean distance from a point to a rectangle (0 when inside)."""
    x, y = point
    dx = max(box[0] - x, 0.0, x - box[2])
    dy = max(box[1] - y, 0.0, y - box[3])
    return (dx * dx + dy * dy) ** 0.5


def near_forklift(person: Detection, forklifts: List[Detection], ratio: float) -> Optional[float]:
    """Return the normalised gap to the closest forklift if within ``ratio``, else None.

    A person is treated as the forklift's operator (and not flagged) when their box lies
    mostly inside the forklift box AND their feet are clearly above the forklift's ground
    line, i.e. they are seated in the cab. A pedestrian standing in front of a forklift
    also overlaps its box in the image, but their feet are at or below that ground line.
    """
    h = max(1.0, person.bbox[3] - person.bbox[1])
    fp = foot_point(person.bbox)
    best: Optional[float] = None
    for f in forklifts:
        f_h = max(1.0, f.bbox[3] - f.bbox[1])
        seated = fp[1] < f.bbox[3] - 0.15 * f_h
        if overlap_ratio(person.bbox, f.bbox) > 0.6 and seated:
            continue
        gap = box_gap(fp, f.bbox) / h
        if gap <= ratio and (best is None or gap < best):
            best = gap
    return best


def analyse_frame(persons: List[Detection], objects: List[Detection], cfg: Dict[str, Any],
                  zones: List[Zone], camera_id: str, width: int, height: int) -> List[Dict[str, Any]]:
    """Produce one assessment dict per worker (plus unassigned violation boxes)."""
    forklift_label = cfg["classes"]["forklift"]
    forklifts = [o for o in objects if o.label == forklift_label]
    ppe_objects = [o for o in objects if o.label != forklift_label]
    required = list(cfg["ppe_policy"]["required"])
    negatives = {v["negative"]: k for k, v in cfg["classes"]["ppe_items"].items()}

    mapping, unassigned = associate(persons, ppe_objects, cfg["detection"]["association_min_overlap"])
    results: List[Dict[str, Any]] = []

    for i, person in enumerate(persons):
        status = ppe_status(mapping[i], cfg)
        violations = [f"missing_{item}" for item in required if status[item]["status"] == "missing"]
        in_zones = zones_for_box(person.bbox, zones, camera_id, width, height)
        gap = near_forklift(person, forklifts, cfg["risk"]["forklift_proximity_ratio"])
        conditions = list(violations)
        if in_zones:
            conditions.append("restricted_zone")
        if gap is not None:
            conditions.append("forklift_proximity")
        score, level = assess(conditions, cfg)
        results.append({
            "worker_id": f"w{i}",
            "assigned": True,
            "bbox": [round(v, 1) for v in person.bbox],
            "confidence": round(person.confidence, 4),
            "person_source": person.source,
            "ppe_status": status,
            "violations": conditions,
            "zones": in_zones,
            "forklift_gap": None if gap is None else round(gap, 3),
            "risk_score": score,
            "risk_level": level,
        })

    # Negative PPE boxes that could not be attached to any detected person still indicate
    # a violation (e.g. a "no_helmet" head whose body was not detected).
    for j, obj in enumerate(o for o in unassigned if o.label in negatives):
        item = negatives[obj.label]
        if item not in required:
            continue
        conditions = [f"missing_{item}"]
        obj_zones = zones_for_box(obj.bbox, zones, camera_id, width, height)
        if obj_zones:
            conditions.append("restricted_zone")
        score, level = assess(conditions, cfg)
        results.append({
            "worker_id": f"u{j}",
            "assigned": False,
            "bbox": [round(v, 1) for v in obj.bbox],
            "confidence": round(obj.confidence, 4),
            "person_source": "unassigned_ppe_box",
            "ppe_status": {item: {"status": "missing", "confidence": round(obj.confidence, 4),
                                  "inferred": False}},
            "violations": conditions,
            "zones": obj_zones,
            "forklift_gap": None,
            "risk_score": score,
            "risk_level": level,
        })
    return results
