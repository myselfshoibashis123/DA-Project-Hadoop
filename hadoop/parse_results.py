#!/usr/bin/env python3
"""Convert raw MapReduce output into dashboard-ready files.

Input : reducer output lines ``<metric>|<dimension>\\t<value>``
Output (in --out-dir, default outputs/analytics):
  aggregates.csv          metric,dimension,value
  analytics_summary.json  headline totals + provenance (execution_mode, source, time)

``execution_mode`` records HOW the numbers were produced so the dashboard can label them:
  hadoop-native / hadoop-docker  -> a real Hadoop Streaming job on YARN
  local-simulation              -> the same mapper/reducer run through a local
                                   map -> sort -> reduce pipe (no Hadoop involved)
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_lines(lines):
    rows = []
    for line in lines:
        line = line.rstrip("\n")
        if not line or "\t" not in line:
            continue
        key, value = line.rsplit("\t", 1)
        metric, _, dimension = key.partition("|")
        rows.append({"metric": metric, "dimension": dimension, "value": int(value)})
    return rows


def build_summary(rows):
    def total(metric):
        return sum(r["value"] for r in rows if r["metric"] == metric)

    def by(metric):
        return {r["dimension"]: r["value"] for r in rows if r["metric"] == metric}

    workers = total("workers")
    obs_all = sum(by("risk_level").values())
    return {
        "frames_analysed": total("frames"),
        "frames_with_violations": total("frames_with_violations"),
        "total_workers_detected": workers,
        "compliant_workers": total("compliant_workers"),
        "observations_with_violations": total("observations_with_violations"),
        "total_violations": total("violations"),
        "average_risk_score": round(total("risk_score_sum") / obs_all, 3) if obs_all else 0.0,
        "violations_by_category": dict(sorted(by("violations_by_category").items(), key=lambda kv: -kv[1])),
        "violations_by_risk_level": by("risk_level"),
        "violations_by_camera": by("violations_by_camera"),
        "violations_by_hour": dict(sorted(by("violations_by_hour").items())),
        "workers_by_camera": by("workers_by_camera"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "analytics"))
    ap.add_argument("--mode", required=True, help="hadoop-native | hadoop-docker | local-simulation")
    ap.add_argument("--source", default="")
    ap.add_argument("--seconds", type=float, default=None)
    args = ap.parse_args()

    rows = parse_lines(Path(args.input).read_text().splitlines())
    if not rows:
        raise SystemExit(f"No MapReduce results found in {args.input}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "aggregates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["metric", "dimension", "value"])
        w.writeheader()
        w.writerows(rows)
    summary = build_summary(rows)
    summary["_provenance"] = {
        "execution_mode": args.mode,
        "is_real_hadoop": args.mode.startswith("hadoop"),
        "source": args.source,
        "job_seconds": args.seconds,
        "generated_at": datetime.now().replace(microsecond=0).isoformat(),
        "raw_output_file": str(Path(args.input).name),
    }
    (out / "analytics_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{args.mode}] {len(rows)} aggregate rows -> {out / 'aggregates.csv'}")
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
