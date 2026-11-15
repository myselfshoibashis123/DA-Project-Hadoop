#!/usr/bin/env bash
# One-command demo of the whole workflow on the bundled sample data.
#
#   bash scripts/run_end_to_end.sh            # Hadoop if available, else local simulation
#   USE_HADOOP=0 bash scripts/run_end_to_end.sh   # force local simulation
#
# Steps: (1) raw data -> HDFS  (2) CV pipeline on images + video  (3) results -> HDFS
#        (4) MapReduce aggregation  (5) print summary; then start the dashboard yourself.
set -euo pipefail
cd "$(dirname "$0")/.."

hadoop_ok() {
  [ "${USE_HADOOP:-1}" = "1" ] || return 1
  command -v hdfs >/dev/null 2>&1 && return 0
  command -v docker >/dev/null 2>&1 && docker ps --format '{{.Names}}' 2>/dev/null | grep -qx msa-hadoop
}

[ -f models/yolo11n.pt ] || python3 scripts/download_weights.py
[ -f data/sample/videos/site_walkthrough.mp4 ] || python3 scripts/make_sample_video.py

if hadoop_ok; then MODE=hadoop; else MODE=local; fi
echo "== Analytics mode: $MODE =="

[ "$MODE" = hadoop ] && bash hadoop/hdfs_upload.sh raw

echo "== CV pipeline: images (camera cam_A) =="
python3 scripts/run_pipeline.py --input data/sample/images --camera cam_A \
  --out outputs/run_cam_A --start-time 2026-11-12T08:00:00 --interval 900 --fresh
echo "== CV pipeline: video (camera cam_B) =="
python3 scripts/run_pipeline.py --input data/sample/videos/site_walkthrough.mp4 --camera cam_B \
  --out outputs/run_cam_B --start-time 2026-11-12T10:30:00 --fresh

if [ "$MODE" = hadoop ]; then
  bash hadoop/hdfs_upload.sh results outputs/run_cam_A
  bash hadoop/hdfs_upload.sh results outputs/run_cam_B
  bash hadoop/run_mapreduce.sh
else
  bash hadoop/run_local_simulation.sh
fi

echo
echo "Done. Start the dashboard with:  streamlit run app/dashboard.py"
