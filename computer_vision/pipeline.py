"""End-to-end computer-vision safety pipeline for images and videos.

Output layout (``out_dir``):
    annotated/                 annotated images / key frames (jpg) and annotated video (mp4)
    detections/observations.jsonl   one JSON line per worker observation (MapReduce input)
    detections/frames.jsonl         one JSON line per analysed frame (MapReduce input)
    reports/violations.csv          flat table of observations that have >= 1 violation
    reports/summary.json            totals for this run
"""
from __future__ import annotations

import csv
import json
import uuid
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import numpy as np

from .detector import SafetyDetector
from .safety_analysis import analyse_frame
from .visualize import draw_frame
from .zones import load_zones

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def _iso(ts: datetime) -> str:
    return ts.replace(microsecond=0).isoformat()


class SafetyPipeline:
    def __init__(self, cfg: Dict[str, Any], detector: Optional[SafetyDetector] = None):
        self.cfg = cfg
        self.detector = detector or SafetyDetector(cfg)
        self.zones = load_zones(cfg)
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]

    # ------------------------------------------------------------------ core
    def process_frame(self, image: np.ndarray, *, source: str, camera_id: str, frame_index: int,
                      timestamp: datetime, timestamp_source: str) -> Dict[str, Any]:
        h, w = image.shape[:2]
        dets = self.detector.detect(image)
        workers = analyse_frame(dets["persons"], dets["objects"], self.cfg, self.zones, camera_id, w, h)
        n_viol = sum(len(wk["violations"]) for wk in workers)
        worst = max((wk["risk_level"] for wk in workers),
                    key=["None", "Low", "Medium", "High"].index, default="None")
        header = (f"{camera_id} | {_iso(timestamp)} | workers: "
                  f"{sum(1 for wk in workers if wk['assigned'])} | violations: {n_viol} | max risk: {worst}")
        annotated = draw_frame(image, workers, dets["objects"], self.zones, camera_id, self.cfg, header)

        base = {"run_id": self.run_id, "camera_id": camera_id, "source": source,
                "frame_index": frame_index, "timestamp": _iso(timestamp),
                "timestamp_source": timestamp_source, "image_width": w, "image_height": h}
        observations = []
        for wk in workers:
            rec = dict(base)
            rec.update({"record_type": "observation", **wk,
                        "ppe_status": {k: v["status"] for k, v in wk["ppe_status"].items()}})
            observations.append(rec)
        frame_rec = dict(base)
        frame_rec.update({
            "record_type": "frame",
            "workers": sum(1 for wk in workers if wk["assigned"]),
            "violations": n_viol,
            "max_risk": worst,
            "detections": [d.to_dict() for d in dets["persons"] + dets["objects"]],
        })
        return {"frame": frame_rec, "observations": observations, "annotated": annotated}

    # ------------------------------------------------------------------ io
    @staticmethod
    def _prepare_out(out_dir: Path) -> Dict[str, Path]:
        dirs = {k: out_dir / k for k in ("annotated", "detections", "reports")}
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        return dirs

    def _finalise(self, out_dir: Path, frames: List[Dict], observations: List[Dict], extra: Dict) -> Dict:
        dirs = self._prepare_out(out_dir)
        with open(dirs["detections"] / "frames.jsonl", "a", encoding="utf-8") as fh:
            for r in frames:
                fh.write(json.dumps(r) + "\n")
        with open(dirs["detections"] / "observations.jsonl", "a", encoding="utf-8") as fh:
            for r in observations:
                fh.write(json.dumps(r) + "\n")

        csv_path = dirs["reports"] / "violations.csv"
        new_file = not csv_path.exists()
        items = list(self.cfg["classes"]["ppe_items"].keys())
        fields = ["run_id", "camera_id", "source", "frame_index", "timestamp", "worker_id", "assigned",
                  "risk_score", "risk_level", "violations", "zones", "forklift_gap"] + \
                 [f"ppe_{i}" for i in items]
        with open(csv_path, "a", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=fields)
            if new_file:
                wr.writeheader()
            for r in observations:
                if not r["violations"]:
                    continue
                row = {k: r.get(k) for k in fields if not k.startswith("ppe_")}
                row["violations"] = ";".join(r["violations"])
                row["zones"] = ";".join(r["zones"])
                for i in items:
                    row[f"ppe_{i}"] = r["ppe_status"].get(i, "")
                wr.writerow(row)

        summary = summarise(frames, observations)
        summary.update(extra)
        summary_path = dirs["reports"] / "summary.json"
        previous = json.loads(summary_path.read_text()) if summary_path.exists() else {"runs": []}
        previous.setdefault("runs", []).append(summary)
        summary_path.write_text(json.dumps(previous, indent=2))
        return summary

    # ------------------------------------------------------------------ public API
    def run_images(self, paths: Iterable[Path], out_dir: Path, camera_id: str = "cam_01",
                   start_time: Optional[datetime] = None, interval_s: float = 60.0) -> Dict:
        """Analyse still images. Each image is given a timestamp ``start_time + i*interval``
        (recorded with ``timestamp_source='assigned'``) because still images carry no
        reliable capture time."""
        dirs = self._prepare_out(out_dir)
        start = start_time or datetime.now()
        frames, observations = [], []
        paths = sorted(Path(p) for p in paths)
        for i, p in enumerate(paths):
            img = cv2.imread(str(p))
            if img is None:
                print(f"  ! skipped unreadable image {p}")
                continue
            res = self.process_frame(img, source=p.name, camera_id=camera_id, frame_index=i,
                                     timestamp=start + timedelta(seconds=i * interval_s),
                                     timestamp_source="assigned")
            cv2.imwrite(str(dirs["annotated"] / f"{p.stem}_annotated.jpg"), res["annotated"])
            frames.append(res["frame"])
            observations.extend(res["observations"])
            print(f"  {p.name}: workers={res['frame']['workers']} violations={res['frame']['violations']} "
                  f"max_risk={res['frame']['max_risk']}")
        return self._finalise(out_dir, frames, observations,
                              {"input_type": "images", "camera_id": camera_id, "n_inputs": len(paths)})

    def run_video(self, path: Path, out_dir: Path, camera_id: str = "cam_01",
                  start_time: Optional[datetime] = None, frame_stride: int = 5, max_frames: int = 0,
                  save_video: bool = True, keyframes: int = 6) -> Dict:
        """Analyse every ``frame_stride``-th frame of a video. Frame timestamps are
        ``start_time + frame_number / fps`` (``timestamp_source='video_offset'``)."""
        dirs = self._prepare_out(out_dir)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        start = start_time or datetime.now()
        writer = None
        frames, observations = [], []
        saved_keyframes = 0
        idx = analysed = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if idx % max(1, frame_stride) == 0:
                res = self.process_frame(img, source=Path(path).name, camera_id=camera_id, frame_index=idx,
                                         timestamp=start + timedelta(seconds=idx / fps),
                                         timestamp_source="video_offset")
                frames.append(res["frame"])
                observations.extend(res["observations"])
                if save_video:
                    if writer is None:
                        h, w = res["annotated"].shape[:2]
                        writer = cv2.VideoWriter(str(dirs["annotated"] / f"{Path(path).stem}_annotated.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                                 max(1.0, fps / max(1, frame_stride)), (w, h))
                    writer.write(res["annotated"])
                if res["frame"]["violations"] and saved_keyframes < keyframes:
                    cv2.imwrite(str(dirs["annotated"] / f"{Path(path).stem}_frame{idx:05d}.jpg"),
                                res["annotated"])
                    saved_keyframes += 1
                analysed += 1
                if analysed % 10 == 0:
                    print(f"  analysed {analysed} frames (frame #{idx})", flush=True)
                if max_frames and analysed >= max_frames:
                    break
            idx += 1
        cap.release()
        if writer is not None:
            writer.release()
        return self._finalise(out_dir, frames, observations,
                              {"input_type": "video", "camera_id": camera_id, "source": Path(path).name,
                               "fps": fps, "frame_stride": frame_stride, "frames_analysed": analysed})


def summarise(frames: List[Dict], observations: List[Dict]) -> Dict[str, Any]:
    cat = Counter(v for o in observations for v in o["violations"])
    risk = Counter(o["risk_level"] for o in observations)
    return {
        "frames_analysed": len(frames),
        "worker_observations": sum(1 for o in observations if o["assigned"]),
        "unassigned_violation_boxes": sum(1 for o in observations if not o["assigned"]),
        "observations_with_violations": sum(1 for o in observations if o["violations"]),
        "total_violations": sum(cat.values()),
        "violations_by_category": dict(cat.most_common()),
        "observations_by_risk_level": {k: risk.get(k, 0) for k in ["None", "Low", "Medium", "High"]},
    }
