"""Drawing of detections, zones and risk levels onto frames."""
from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

from .detector import Detection
from .zones import Zone

RISK_COLORS = {  # BGR
    "None": (60, 170, 60),
    "Low": (0, 210, 230),
    "Medium": (0, 140, 255),
    "High": (40, 40, 220),
}
POSITIVE_COLOR = (80, 200, 80)
NEGATIVE_COLOR = (60, 60, 230)
FORKLIFT_COLOR = (220, 140, 30)


def _label(img: np.ndarray, text: str, x: int, y: int, color: tuple, scale: float) -> None:
    thick = max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    y = max(th + base + 2, y)
    cv2.rectangle(img, (x, y - th - base - 2), (x + tw + 4, y), color, -1)
    cv2.putText(img, text, (x + 2, y - base), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                thick, cv2.LINE_AA)


def draw_frame(image: np.ndarray, workers: List[Dict[str, Any]], objects: List[Detection],
               zones: List[Zone], camera_id: str, cfg: Dict[str, Any], header: str = "") -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    scale = max(0.4, min(w, h) / 1100.0)
    thick = max(1, int(round(scale * 3)))

    # restricted zones (translucent fill + outline)
    overlay = out.copy()
    for z in zones:
        if not z.applies_to(camera_id):
            continue
        poly = z.pixel_polygon(w, h).astype(np.int32)
        cv2.fillPoly(overlay, [poly], (40, 40, 200))
        cv2.polylines(out, [poly], True, (40, 40, 200), thick, cv2.LINE_AA)
    out = cv2.addWeighted(overlay, 0.18, out, 0.82, 0)
    for z in zones:
        if z.applies_to(camera_id):
            p = z.pixel_polygon(w, h).astype(int)
            _label(out, f"ZONE: {z.name}", int(p[:, 0].min()) + 3, int(p[:, 1].min()) + 22,
                   (40, 40, 200), scale * 0.8)

    negatives = {v["negative"] for v in cfg["classes"]["ppe_items"].values()}
    for o in objects:
        x1, y1, x2, y2 = map(int, o.bbox)
        if o.label == cfg["classes"]["forklift"]:
            color = FORKLIFT_COLOR
        else:
            color = NEGATIVE_COLOR if o.label in negatives else POSITIVE_COLOR
        cv2.rectangle(out, (x1, y1), (x2, y2), color, max(1, thick - 1))
        _label(out, f"{o.label} {o.confidence:.2f}", x1, y1, color, scale * 0.6)

    for wk in workers:
        x1, y1, x2, y2 = map(int, wk["bbox"])
        color = RISK_COLORS.get(wk["risk_level"], (200, 200, 200))
        if wk["assigned"]:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thick + 1)
        text = f"{wk['worker_id']} {wk['risk_level']}({wk['risk_score']})"
        _label(out, text, x1, y2 + int(22 * scale) + 4, color, scale * 0.7)

    if header:
        _label(out, header, 4, int(30 * scale) + 6, (30, 30, 30), scale * 0.75)
    return out
