import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hadoop"))
from parse_results import build_summary, parse_lines  # noqa: E402

RECORDS = [
    {"record_type": "frame", "camera_id": "cam_A", "timestamp": "2026-11-12T08:05:00", "violations": 2},
    {"record_type": "frame", "camera_id": "cam_A", "timestamp": "2026-11-12T09:10:00", "violations": 0},
    {"record_type": "observation", "camera_id": "cam_A", "timestamp": "2026-11-12T08:05:00", "assigned": True,
     "violations": ["missing_helmet", "restricted_zone"], "risk_score": 7, "risk_level": "High",
     "ppe_status": {"helmet": "missing", "vest": "worn"}},
    {"record_type": "observation", "camera_id": "cam_A", "timestamp": "2026-11-12T09:10:00", "assigned": True,
     "violations": [], "risk_score": 0, "risk_level": "None", "ppe_status": {"helmet": "worn", "vest": "worn"}},
    {"record_type": "observation", "camera_id": "cam_B", "timestamp": "2026-11-12T08:30:00", "assigned": False,
     "violations": ["missing_gloves"], "risk_score": 1, "risk_level": "Low", "ppe_status": {"gloves": "missing"}},
]


def run_job(lines):
    data = "".join(lines).encode()
    mapped = subprocess.run([sys.executable, str(ROOT / "hadoop" / "mapper.py")], input=data,
                            capture_output=True, check=True).stdout
    shuffled = b"".join(sorted(mapped.splitlines(keepends=True)))
    return subprocess.run([sys.executable, str(ROOT / "hadoop" / "reducer.py")], input=shuffled,
                          capture_output=True, check=True).stdout.decode()


def test_mapper_reducer_counts():
    lines = [json.dumps(r) + "\n" for r in RECORDS] + ["not json\n", "\n"]
    rows = {(r["metric"], r["dimension"]): r["value"] for r in parse_lines(run_job(lines).splitlines())}
    assert rows[("frames", "TOTAL")] == 2
    assert rows[("frames_with_violations", "TOTAL")] == 1
    assert rows[("workers", "TOTAL")] == 2                      # unassigned box is not a worker
    assert rows[("compliant_workers", "TOTAL")] == 1
    assert rows[("violations", "TOTAL")] == 3
    assert rows[("violations_by_category", "missing_helmet")] == 1
    assert rows[("violations_by_category", "missing_gloves")] == 1
    assert rows[("violations_by_hour", "2026-11-12T08:00")] == 3
    assert rows[("risk_level", "High")] == 1 and rows[("risk_level", "Low")] == 1
    assert rows[("ppe_status", "helmet#missing")] == 1
    assert rows[("risk_score_sum", "TOTAL")] == 8


def test_reducer_works_as_combiner():
    """Running the reducer on partial (combined) outputs must give the same totals."""
    lines = [json.dumps(r) + "\n" for r in RECORDS]
    direct = run_job(lines)
    part1, part2 = run_job(lines[:3]), run_job(lines[3:])
    shuffled = "".join(sorted((part1 + part2).splitlines(keepends=True)))
    combined = subprocess.run([sys.executable, str(ROOT / "hadoop" / "reducer.py")], input=shuffled.encode(),
                              capture_output=True, check=True).stdout.decode()
    assert sorted(direct.splitlines()) == sorted(combined.splitlines())


def test_summary_builder():
    lines = [json.dumps(r) + "\n" for r in RECORDS]
    s = build_summary(parse_lines(run_job(lines).splitlines()))
    assert s["total_workers_detected"] == 2 and s["total_violations"] == 3
    assert s["average_risk_score"] == round(8 / 3, 3)
