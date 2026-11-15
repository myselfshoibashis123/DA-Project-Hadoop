#!/usr/bin/env bash
# Run the safety-analytics MapReduce job on Hadoop (YARN) with Hadoop Streaming,
# then fetch the aggregated result from HDFS and convert it for the dashboard.
#
# Usage:
#   bash hadoop/run_mapreduce.sh [HDFS_INPUT_GLOB]
# Default input: /safety_analytics/detections/*/*.jsonl  (all uploaded runs)
#
# Result files (local):
#   outputs/analytics/mapreduce_part-00000.txt   raw reducer output fetched from HDFS
#   outputs/analytics/aggregates.csv             metric, dimension, value
#   outputs/analytics/analytics_summary.json     totals + execution_mode = "hadoop-<mode>"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

INPUT="${1:-$HDFS_DETECTIONS/*/*.jsonl}"
OUTPUT="$HDFS_MR_OUTPUT"
LOCAL_OUT="$PROJECT_DIR/outputs/analytics"
mkdir -p "$LOCAL_OUT"

STREAMING_JAR="$(run_hadoop bash -c 'ls "$HADOOP_HOME"/share/hadoop/tools/lib/hadoop-streaming-*.jar | head -n1')"
[ -n "$STREAMING_JAR" ] || { echo "hadoop-streaming jar not found" >&2; exit 1; }
log "Streaming jar: $STREAMING_JAR"
log "Input:  $INPUT"
log "Output: $OUTPUT"

run_hadoop hdfs dfs -rm -r -f -skipTrash "$OUTPUT" >/dev/null 2>&1 || true

MAPPER="$(exec_path "$PROJECT_DIR/hadoop/mapper.py")"
REDUCER="$(exec_path "$PROJECT_DIR/hadoop/reducer.py")"

START=$(date +%s)
run_hadoop hadoop jar "$STREAMING_JAR" \
  -D mapreduce.job.name="manufacturing-safety-analytics" \
  -D mapreduce.job.reduces=1 \
  -files "$MAPPER,$REDUCER" \
  -mapper "python3 mapper.py" \
  -combiner "python3 reducer.py" \
  -reducer "python3 reducer.py" \
  -input "$INPUT" \
  -output "$OUTPUT"
END=$(date +%s)
log "MapReduce job finished in $((END - START)) s"

run_hadoop hdfs dfs -ls "$OUTPUT"
run_hadoop hdfs dfs -cat "$OUTPUT/part-*" > "$LOCAL_OUT/mapreduce_part-00000.txt"
log "Fetched $(wc -l < "$LOCAL_OUT/mapreduce_part-00000.txt") result lines -> $LOCAL_OUT/mapreduce_part-00000.txt"

"$PYTHON_BIN" "$PROJECT_DIR/hadoop/parse_results.py" \
  --input "$LOCAL_OUT/mapreduce_part-00000.txt" \
  --mode "hadoop-$HADOOP_MODE" \
  --source "hdfs://$OUTPUT (input: $INPUT)" \
  --seconds "$((END - START))"
