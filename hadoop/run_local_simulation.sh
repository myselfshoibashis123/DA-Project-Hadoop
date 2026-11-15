#!/usr/bin/env bash
# LOCAL SIMULATION (no Hadoop): runs hadoop/mapper.py and hadoop/reducer.py through a
# map -> sort -> reduce pipe, the same contract Hadoop Streaming uses.
# Results are labelled "local-simulation" in outputs/analytics/analytics_summary.json.
#
# Usage: bash hadoop/run_local_simulation.sh [JSONL files ...]
#
# Equivalent one-liner, useful for debugging the mapper/reducer by hand:
#   cat outputs/*/detections/*.jsonl | python3 hadoop/mapper.py | sort | python3 hadoop/reducer.py
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
if [ "$#" -gt 0 ]; then
  python3 hadoop/local_mapreduce.py --inputs "$@"
else
  python3 hadoop/local_mapreduce.py
fi
