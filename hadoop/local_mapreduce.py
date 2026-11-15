#!/usr/bin/env python3
"""LOCAL SIMULATION of the Hadoop Streaming job - for machines without Hadoop.

It runs exactly the same mapper.py and reducer.py as the cluster job, wired together
the way Hadoop Streaming does it:

    cat inputs | mapper.py | (combiner) | sort | reducer.py  > part-00000

Results are labelled ``execution_mode = local-simulation`` so they are never confused
with a real Hadoop run. Scale-out, HDFS replication and YARN scheduling are of course
not exercised by this mode.

Usage:
    python hadoop/local_mapreduce.py                                   # all outputs/*/detections/*.jsonl
    python hadoop/local_mapreduce.py --inputs outputs/pipeline/detections/*.jsonl
"""
from __future__ import annotations

import argparse
import glob
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent


def run(cmd, data: bytes) -> bytes:
    proc = subprocess.run(cmd, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return proc.stdout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="*", default=None)
    ap.add_argument("--splits", type=int, default=2, help="simulated number of map tasks")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "analytics"))
    args = ap.parse_args()

    files = args.inputs or sorted(glob.glob(str(PROJECT_ROOT / "outputs" / "*" / "detections" / "*.jsonl")))
    files = [f for f in files if Path(f).is_file()]
    if not files:
        sys.exit("No *.jsonl detection files found. Run scripts/run_pipeline.py first.")
    lines = []
    for f in files:
        lines.extend(Path(f).read_bytes().splitlines(keepends=True))
    print(f"[local-simulation] {len(files)} input files, {len(lines)} records")

    t0 = time.time()
    py = sys.executable
    # map phase: split input like Hadoop input splits, run mapper + combiner per split
    n = max(1, args.splits)
    chunk = (len(lines) + n - 1) // n or 1
    map_out = []
    for i in range(0, len(lines), chunk):
        mapped = run([py, str(HERE / "mapper.py")], b"".join(lines[i:i + chunk]))
        combined = run([py, str(HERE / "reducer.py")], b"".join(sorted(mapped.splitlines(keepends=True))))
        map_out.extend(combined.splitlines(keepends=True))
    # shuffle & sort phase (byte order, like Hadoop's Text key comparator)
    shuffled = b"".join(sorted(map_out, key=lambda l: l.split(b"\t", 1)[0]))
    # reduce phase
    reduced = run([py, str(HERE / "reducer.py")], shuffled)
    elapsed = time.time() - t0

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    part = out / "mapreduce_part-00000.txt"
    part.write_bytes(reduced)
    print(f"[local-simulation] map->sort->reduce done in {elapsed:.2f}s -> {part}")
    subprocess.run([py, str(HERE / "parse_results.py"), "--input", str(part), "--out-dir", str(out),
                    "--mode", "local-simulation",
                    "--source", "local files: " + ", ".join(str(Path(f).relative_to(PROJECT_ROOT))
                                                            if Path(f).is_relative_to(PROJECT_ROOT) else f
                                                            for f in files),
                    "--seconds", f"{elapsed:.2f}"], check=True)


if __name__ == "__main__":
    main()
