"""Restricted / hazardous zone handling with normalised polygons."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class Zone:
    name: str
    polygon: List[Tuple[float, float]]  # normalised (0..1) coordinates
    cameras: List[str]

    def applies_to(self, camera_id: str) -> bool:
        return "*" in self.cameras or camera_id in self.cameras

    def pixel_polygon(self, width: int, height: int) -> np.ndarray:
        return np.array([[x * width, y * height] for x, y in self.polygon], dtype=np.float32)


def load_zones(cfg: Dict[str, Any]) -> List[Zone]:
    zones = []
    for z in cfg.get("zones", []) or []:
        poly = [(float(x), float(y)) for x, y in z["polygon"]]
        if len(poly) < 3:
            raise ValueError(f"Zone {z.get('name')} needs at least 3 points")
        zones.append(Zone(name=str(z["name"]), polygon=poly, cameras=list(z.get("cameras", ["*"]))))
    return zones


def point_in_polygon(point: Tuple[float, float], polygon: Sequence[Tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test (points on an edge count as inside)."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # on-edge check
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) < 1e-9 and min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9 \
                and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9:
            return True
        if (y1 > y) != (y2 > y):
            x_int = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_int:
                inside = not inside
    return inside


def foot_point(bbox: tuple) -> Tuple[float, float]:
    """Bottom-centre of a box - approximates where the worker stands."""
    x1, _, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def zones_for_box(bbox: tuple, zones: List[Zone], camera_id: str, width: int, height: int) -> List[str]:
    fx, fy = foot_point(bbox)
    norm = (fx / width, fy / height)
    return [z.name for z in zones if z.applies_to(camera_id) and point_in_polygon(norm, z.polygon)]
