#!/usr/bin/env python3
"""Hadoop Streaming MAPPER for safety-violation analytics.

Input : JSON lines produced by the CV pipeline (detections/frames.jsonl and
        detections/observations.jsonl), read from HDFS by Hadoop Streaming.
Output: ``<metric>|<dimension>\\t<count>`` - one line per counted fact.

Metrics emitted
  frames|TOTAL                         analysed frames
  frames_with_violations|TOTAL
  workers|TOTAL                        worker observations (person detections)
  workers_by_camera|<camera>
  compliant_workers|TOTAL              observations with no violation
  observations_with_violations|TOTAL
  violations|TOTAL                     individual violation conditions
  violations_by_category|<category>    missing_helmet, restricted_zone, ...
  violations_by_camera|<camera>
  violations_by_hour|<YYYY-MM-DDTHH:00>
  violations_by_date|<YYYY-MM-DD>
  risk_level|<None|Low|Medium|High>    per observation
  risk_by_hour|<YYYY-MM-DDTHH:00>#<level>
  risk_score_sum|TOTAL                 sum of risk scores (for averages)
  ppe_status|<item>#<worn|missing|unknown>

Malformed lines are skipped and counted through a Hadoop counter.
"""
import json
import sys


def emit(metric, dimension, value=1):
    sys.stdout.write("%s|%s\t%d\n" % (metric, dimension, value))


def counter(name):
    # Hadoop Streaming counter protocol (shown in the job's counters)
    sys.stderr.write("reporter:counter:SafetyAnalytics,%s,1\n" % name)


def hour_of(ts):
    # ISO timestamp "YYYY-MM-DDTHH:MM:SS" -> "YYYY-MM-DDTHH:00"
    return ts[:13] + ":00" if ts and len(ts) >= 13 else "unknown"


def process(rec):
    rtype = rec.get("record_type")
    camera = rec.get("camera_id", "unknown")
    ts = rec.get("timestamp", "")
    if rtype == "frame":
        emit("frames", "TOTAL")
        if int(rec.get("violations", 0)) > 0:
            emit("frames_with_violations", "TOTAL")
        counter("frame_records")
        return
    if rtype != "observation":
        counter("unknown_record_type")
        return

    violations = rec.get("violations") or []
    level = rec.get("risk_level", "None")
    hour = hour_of(ts)
    if rec.get("assigned", True):
        emit("workers", "TOTAL")
        emit("workers_by_camera", camera)
        if not violations:
            emit("compliant_workers", "TOTAL")
        for item, status in sorted((rec.get("ppe_status") or {}).items()):
            emit("ppe_status", "%s#%s" % (item, status))
    if violations:
        emit("observations_with_violations", "TOTAL")
        emit("violations", "TOTAL", len(violations))
        emit("violations_by_camera", camera, len(violations))
        emit("violations_by_hour", hour, len(violations))
        emit("violations_by_date", ts[:10] if len(ts) >= 10 else "unknown", len(violations))
        for v in violations:
            emit("violations_by_category", v)
    emit("risk_level", level)
    emit("risk_by_hour", "%s#%s" % (hour, level))
    emit("risk_score_sum", "TOTAL", int(rec.get("risk_score", 0)))
    counter("observation_records")


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            process(json.loads(line))
        except (ValueError, TypeError, AttributeError):
            counter("malformed_lines")


if __name__ == "__main__":
    main()
