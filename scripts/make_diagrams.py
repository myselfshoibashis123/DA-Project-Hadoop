#!/usr/bin/env python3
"""Generate docs/architecture.png and docs/workflow.png with matplotlib."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from computer_vision.config import resolve_path  # noqa: E402

COL = {"input": "#dbeafe", "cv": "#dcfce7", "hadoop": "#fef3c7", "app": "#fce7f3", "edge": "#334155"}


def box(ax, x, y, w, h, text, color, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=color, ec=COL["edge"], lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, label=""):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 lw=1.3, color=COL["edge"]))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, ha="center", fontsize=7.5, color="#475569")


def elbow(ax, points, label="", label_at=0):
    """Polyline whose last segment ends in an arrow head."""
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color=COL["edge"], lw=1.3)
    ax.plot(xs[-2:-1] + xs[-1:], ys[-2:-1] + ys[-1:], color=COL["edge"], lw=0)
    arrow(ax, *points[-2], *points[-1])
    if label:
        (x1, y1), (x2, y2) = points[label_at], points[label_at + 1]
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.08, label, ha="center", fontsize=7.5, color="#475569")


def group(ax, x, y, w, h, title, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.2",
                                fc=color, ec="#94a3b8", lw=1, ls="--", alpha=0.45))
    ax.text(x + 0.12, y + h - 0.28, title, fontsize=10, fontweight="bold", color="#1e293b")


def architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 8.2))
    ax.set_xlim(0, 15)
    ax.set_ylim(-0.1, 8.4)
    ax.axis("off")
    ax.text(7.5, 8.15, "System architecture - Safety & Risk Analytics for Manufacturing Plants",
            ha="center", fontsize=13, fontweight="bold")

    group(ax, 0.2, 0.4, 2.8, 7.0, "1. Data sources", COL["input"])
    box(ax, 0.45, 5.4, 2.3, 1.1, "Plant cameras\n(images / video)", "white")
    box(ax, 0.45, 3.7, 2.3, 1.1, "Construction-PPE\ndataset (CC BY 4.0)\n1,000 labelled images", "white", 8.5)
    box(ax, 0.45, 2.0, 2.3, 1.1, "config.yaml\nPPE policy, zones,\nrisk weights", "white", 8.5)

    group(ax, 3.4, 0.4, 4.6, 7.0, "2. Computer-vision layer (Python)", COL["cv"])
    box(ax, 3.65, 5.9, 4.1, 0.85, "OpenCV frame extraction (stride N)", "white")
    box(ax, 3.65, 4.75, 1.95, 0.9, "YOLO11n PPE model\n(fine-tuned, 12 cls)", "white", 8.5)
    box(ax, 5.8, 4.75, 1.95, 0.9, "YOLO11n COCO\nperson model", "white", 8.5)
    box(ax, 3.65, 3.6, 4.1, 0.85, "PPE-to-worker association\n(worn / missing / unknown)", "white", 8.5)
    box(ax, 3.65, 2.5, 4.1, 0.8, "Restricted-zone + forklift-proximity rules", "white", 8.5)
    box(ax, 3.65, 1.4, 4.1, 0.8, "Risk scoring -> None / Low / Medium / High", "white", 8.5)
    box(ax, 3.65, 0.6, 4.1, 0.6, "JSONL records + annotated frames", "white", 8.5)

    group(ax, 8.4, 0.4, 3.3, 7.0, "3. Apache Hadoop", COL["hadoop"])
    box(ax, 8.65, 5.3, 2.8, 1.5, "HDFS\n/raw  /processed\n/detections  /analytics", "white", 8.5)
    box(ax, 8.65, 2.7, 2.8, 2.0, "MapReduce (Streaming)\nmapper.py -> combiner\n-> shuffle/sort\n-> reducer.py", "white", 8.5)
    box(ax, 8.65, 0.8, 2.8, 1.3, "Aggregates: workers,\nviolations by category /\nrisk / hour / camera", "white", 8)

    group(ax, 12.1, 0.4, 2.7, 7.0, "4. Presentation", COL["app"])
    box(ax, 12.3, 4.2, 2.3, 2.2, "Streamlit\ndashboard\nKPIs, charts,\nrecords, gallery,\nlive analysis", "white", 8.5)
    box(ax, 12.3, 1.4, 2.3, 1.9, "Reports\nviolations.csv\nsummary.json\nmetrics", "white", 8.5)

    arrow(ax, 2.75, 5.95, 3.65, 6.3)
    arrow(ax, 2.75, 4.25, 3.65, 5.1, "train")
    arrow(ax, 2.75, 2.55, 3.65, 2.9)
    arrow(ax, 5.7, 5.9, 4.6, 5.65)
    arrow(ax, 5.7, 5.9, 6.8, 5.65)
    arrow(ax, 4.6, 4.75, 5.2, 4.45)
    arrow(ax, 6.8, 4.75, 6.2, 4.45)
    arrow(ax, 5.7, 3.6, 5.7, 3.3)
    arrow(ax, 5.7, 2.5, 5.7, 2.2)
    arrow(ax, 5.7, 1.4, 5.7, 1.2)
    elbow(ax, [(1.6, 6.5), (1.6, 7.6), (10.05, 7.6), (10.05, 6.8)], "hdfs dfs -put  (raw images / videos)", 1)
    arrow(ax, 7.75, 0.9, 8.65, 5.5, "upload results")
    arrow(ax, 10.05, 5.3, 10.05, 4.7)
    arrow(ax, 10.05, 2.7, 10.05, 2.1)
    arrow(ax, 11.45, 1.45, 12.3, 4.6, "getmerge")
    elbow(ax, [(5.7, 0.6), (5.7, 0.2), (13.45, 0.2), (13.45, 1.4)], "per-run reports", 1)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def workflow(path: Path) -> None:
    steps = [
        ("1", "Download & prepare\ndataset", "download_dataset.py\nprepare_dataset.py", COL["input"]),
        ("2", "Fine-tune / load\nYOLO models", "train_ppe_model.py\nevaluate_model.py", COL["cv"]),
        ("3", "Upload raw data\nto HDFS", "hdfs_upload.sh raw", COL["hadoop"]),
        ("4", "Run CV pipeline\n(detect, associate,\nzones, risk)", "run_pipeline.py", COL["cv"]),
        ("5", "Store detections\nin HDFS", "hdfs_upload.sh results", COL["hadoop"]),
        ("6", "MapReduce\naggregation", "run_mapreduce.sh\n(or local simulation)", COL["hadoop"]),
        ("7", "Fetch aggregates\n-> CSV / JSON", "parse_results.py", COL["hadoop"]),
        ("8", "Dashboard", "streamlit run\napp/dashboard.py", COL["app"]),
    ]
    fig, ax = plt.subplots(figsize=(16, 4.4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    ax.text(8, 4.1, "End-to-end workflow", ha="center", fontsize=13, fontweight="bold")
    w, gap = 1.75, 0.2
    for i, (n, title, cmd, color) in enumerate(steps):
        x = 0.15 + i * (w + gap)
        box(ax, x, 1.75, w, 1.85, f"{n}. {title}", color, 8.8, bold=True)
        ax.text(x + w / 2, 1.35, cmd, ha="center", va="top", fontsize=7.5, family="monospace", color="#334155")
        if i < len(steps) - 1:
            arrow(ax, x + w, 2.67, x + w + gap, 2.67)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    docs = resolve_path("docs")
    docs.mkdir(exist_ok=True)
    architecture(docs / "architecture.png")
    workflow(docs / "workflow.png")
    print(f"Wrote {docs / 'architecture.png'} and {docs / 'workflow.png'}")


if __name__ == "__main__":
    main()
