#!/usr/bin/env bash
# Upload raw inputs and pipeline outputs to HDFS.
#
# Usage:
#   bash hadoop/hdfs_upload.sh raw  [IMAGES_DIR] [VIDEOS_DIR]   # step 1: raw data -> HDFS
#   bash hadoop/hdfs_upload.sh results OUTPUT_DIR [RUN_NAME]    # step 3: CV results -> HDFS
#
# Defaults: IMAGES_DIR=data/sample/images  VIDEOS_DIR=data/sample/videos
#           RUN_NAME = basename of OUTPUT_DIR
#
# HDFS layout:
#   /safety_analytics/raw/images/                       raw images
#   /safety_analytics/raw/videos/                       raw videos
#   /safety_analytics/processed/frames/<RUN_NAME>/      annotated frames / videos
#   /safety_analytics/detections/<RUN_NAME>/*.jsonl     detection + violation records
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

what="${1:-}"
case "$what" in
  raw)
    images="${2:-$PROJECT_DIR/data/sample/images}"
    videos="${3:-$PROJECT_DIR/data/sample/videos}"
    run_hadoop hdfs dfs -mkdir -p "$HDFS_RAW_IMAGES" "$HDFS_RAW_VIDEOS"
    if [ -d "$images" ] && [ -n "$(ls -A "$images" 2>/dev/null)" ]; then
      log "Uploading images from $images -> $HDFS_RAW_IMAGES"
      for f in "$images"/*; do run_hadoop hdfs dfs -put -f "$(exec_path "$f")" "$HDFS_RAW_IMAGES/"; done
    fi
    if [ -d "$videos" ] && [ -n "$(ls -A "$videos" 2>/dev/null)" ]; then
      log "Uploading videos from $videos -> $HDFS_RAW_VIDEOS"
      for f in "$videos"/*; do run_hadoop hdfs dfs -put -f "$(exec_path "$f")" "$HDFS_RAW_VIDEOS/"; done
    fi
    run_hadoop hdfs dfs -ls -h "$HDFS_RAW_IMAGES" "$HDFS_RAW_VIDEOS"
    ;;
  results)
    out="${2:?usage: hdfs_upload.sh results OUTPUT_DIR [RUN_NAME]}"
    out="$(cd "$out" && pwd)"
    run="${3:-$(basename "$out")}"
    [ -f "$out/detections/observations.jsonl" ] || { echo "No detections in $out" >&2; exit 1; }
    run_hadoop hdfs dfs -mkdir -p "$HDFS_DETECTIONS/$run" "$HDFS_FRAMES/$run"
    log "Uploading detection records -> $HDFS_DETECTIONS/$run"
    for f in "$out"/detections/*.jsonl; do
      run_hadoop hdfs dfs -put -f "$(exec_path "$f")" "$HDFS_DETECTIONS/$run/"
    done
    log "Uploading annotated frames -> $HDFS_FRAMES/$run"
    for f in "$out"/annotated/*; do
      [ -f "$f" ] && run_hadoop hdfs dfs -put -f "$(exec_path "$f")" "$HDFS_FRAMES/$run/"
    done
    run_hadoop hdfs dfs -ls -h "$HDFS_DETECTIONS/$run"
    ;;
  *)
    sed -n '2,15p' "$0"; exit 1 ;;
esac
log "Upload finished."
