"""Data access for the dashboard (independent of Streamlit so it can be unit-tested)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
RISK_ORDER = ["None", "Low", "Medium", "High"]


def load_observations(outputs: Path = OUTPUTS) -> pd.DataFrame:
    """All worker observations from every pipeline run found under ``outputs/*/detections``."""
    rows: List[Dict[str, Any]] = []
    for f in sorted(outputs.glob("*/detections/observations.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                rec["output_folder"] = f.parent.parent.name
                rows.append(rec)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["n_violations"] = df["violations"].apply(len)
    df["violations_str"] = df["violations"].apply(lambda v: ", ".join(v))
    df["zones_str"] = df["zones"].apply(lambda v: ", ".join(v) if isinstance(v, list) else "")
    df["risk_level"] = pd.Categorical(df["risk_level"], categories=RISK_ORDER, ordered=True)
    return df


def load_frames(outputs: Path = OUTPUTS) -> pd.DataFrame:
    rows = []
    for f in sorted(outputs.glob("*/detections/frames.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                rec.pop("detections", None)
                rec["output_folder"] = f.parent.parent.name
                rows.append(rec)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def load_mapreduce(outputs: Path = OUTPUTS) -> Optional[Dict[str, Any]]:
    """Return {'summary': dict, 'aggregates': DataFrame} from outputs/analytics or None."""
    summ = outputs / "analytics" / "analytics_summary.json"
    agg = outputs / "analytics" / "aggregates.csv"
    if not (summ.exists() and agg.exists()):
        return None
    return {"summary": json.loads(summ.read_text()), "aggregates": pd.read_csv(agg, keep_default_na=False)}


def aggregate(agg: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = agg[agg["metric"] == metric][["dimension", "value"]].copy()
    return sub.sort_values("value", ascending=False).reset_index(drop=True)


def risk_by_hour(agg: pd.DataFrame) -> pd.DataFrame:
    sub = agg[agg["metric"] == "risk_by_hour"].copy()
    if sub.empty:
        return sub
    sub[["hour", "risk_level"]] = sub["dimension"].str.split("#", n=1, expand=True)
    sub["hour"] = pd.to_datetime(sub["hour"], errors="coerce")
    return sub[["hour", "risk_level", "value"]].sort_values("hour")


def ppe_status_table(agg: pd.DataFrame) -> pd.DataFrame:
    sub = agg[agg["metric"] == "ppe_status"].copy()
    if sub.empty:
        return sub
    sub[["item", "status"]] = sub["dimension"].str.split("#", n=1, expand=True)
    return sub.pivot_table(index="item", columns="status", values="value", fill_value=0).reset_index()


def local_summary(obs: pd.DataFrame, frames: pd.DataFrame) -> Dict[str, Any]:
    """Pandas fallback used when no MapReduce output exists yet."""
    if obs.empty:
        return {}
    assigned = obs[obs["assigned"]]
    cats = obs.explode("violations")["violations"].dropna().value_counts().to_dict()
    return {
        "frames_analysed": int(len(frames)),
        "frames_with_violations": int((frames["violations"] > 0).sum()) if not frames.empty else 0,
        "total_workers_detected": int(len(assigned)),
        "compliant_workers": int((assigned["n_violations"] == 0).sum()),
        "observations_with_violations": int((obs["n_violations"] > 0).sum()),
        "total_violations": int(obs["n_violations"].sum()),
        "average_risk_score": round(float(obs["risk_score"].mean()), 3),
        "violations_by_category": cats,
        "violations_by_risk_level": obs["risk_level"].value_counts().reindex(RISK_ORDER, fill_value=0).to_dict(),
    }


def annotated_images(outputs: Path = OUTPUTS) -> List[Path]:
    return sorted(p for p in outputs.glob("*/annotated/*.jpg"))


def load_metrics(outputs: Path = OUTPUTS) -> Dict[str, Any]:
    m = outputs / "metrics"
    out: Dict[str, Any] = {}
    if (m / "test_metrics.json").exists():
        out["overall"] = json.loads((m / "test_metrics.json").read_text())
    for name in ("per_class_metrics", "violation_image_level", "dataset_class_distribution"):
        if (m / f"{name}.csv").exists():
            out[name] = pd.read_csv(m / f"{name}.csv")
    out["plots"] = sorted((m / "plots").glob("*.png")) if (m / "plots").exists() else []
    return out
