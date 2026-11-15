#!/usr/bin/env bash
# Shared settings for the Hadoop scripts. Sourced, not executed.
#
# Execution modes (chosen automatically):
#   native : `hdfs` and `hadoop` are on PATH (a real Hadoop installation on this machine)
#   docker : the single-node Hadoop container from docker-compose.yml is running
#            (commands are run with `docker exec`; the project is mounted at /project)
# Set FORCE_DOCKER=1 to prefer the container even when `hdfs` is on PATH.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HDFS_BASE="${HDFS_BASE:-/safety_analytics}"
HDFS_RAW_IMAGES="$HDFS_BASE/raw/images"
HDFS_RAW_VIDEOS="$HDFS_BASE/raw/videos"
HDFS_FRAMES="$HDFS_BASE/processed/frames"
HDFS_DETECTIONS="$HDFS_BASE/detections"
HDFS_MR_OUTPUT="$HDFS_BASE/analytics/mapreduce_output"
HADOOP_CONTAINER="${HADOOP_CONTAINER:-msa-hadoop}"
CONTAINER_PROJECT="${CONTAINER_PROJECT:-/project}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if command -v hdfs >/dev/null 2>&1 && [ -z "${FORCE_DOCKER:-}" ]; then
  HADOOP_MODE="native"
elif command -v docker >/dev/null 2>&1 && \
     docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$HADOOP_CONTAINER"; then
  HADOOP_MODE="docker"
else
  echo "ERROR: No Hadoop available." >&2
  echo "  * Start the bundled single-node cluster:  docker compose up -d hadoop" >&2
  echo "  * or install Hadoop so that 'hdfs' is on PATH," >&2
  echo "  * or use the local simulation:            bash hadoop/run_local_simulation.sh" >&2
  exit 1
fi

# Run a Hadoop CLI command (hdfs / hadoop / bash) in the selected environment.
run_hadoop() {
  if [ "$HADOOP_MODE" = "docker" ]; then
    docker exec -i "$HADOOP_CONTAINER" "$@"
  else
    "$@"
  fi
}

# Translate a path on this machine into the path the Hadoop CLI can see.
exec_path() {
  local abs
  abs="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
  if [ "$HADOOP_MODE" = "docker" ]; then
    case "$abs" in
      "$PROJECT_DIR"/*) echo "$CONTAINER_PROJECT/${abs#"$PROJECT_DIR"/}" ;;
      *) echo "ERROR: $abs is outside the project folder mounted into the container" >&2; return 1 ;;
    esac
  else
    echo "$abs"
  fi
}

log() { echo "[$(date +%H:%M:%S)] [$HADOOP_MODE] $*"; }
