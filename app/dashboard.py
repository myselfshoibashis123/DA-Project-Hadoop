"""Streamlit dashboard for Manufacturing Safety Analytics.

Run:  streamlit run app/dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import data_loader as dl  # noqa: E402
from computer_vision.config import load_config  # noqa: E402

RISK_COLORS = {"None": "#3caa3c", "Low": "#e6d200", "Medium": "#ff8c00", "High": "#dc2828"}

st.set_page_config(page_title="Manufacturing Safety Analytics", page_icon="🦺", layout="wide")


@st.cache_data(ttl=30)
def _load_all():
    return dl.load_observations(), dl.load_frames(), dl.load_mapreduce(), dl.load_metrics()


@st.cache_resource
def _pipeline():
    from computer_vision.detector import weights_available
    from computer_vision.pipeline import SafetyPipeline
    cfg = load_config()
    if not weights_available(cfg):
        return None
    return SafetyPipeline(cfg)


def provenance_banner(mr):
    if mr is None:
        st.warning("No MapReduce output found in `outputs/analytics/`. Headline numbers below are "
                   "computed directly with pandas from the detection records. Run "
                   "`bash hadoop/run_mapreduce.sh` (Hadoop) or `bash hadoop/run_local_simulation.sh`.")
        return
    p = mr["summary"].get("_provenance", {})
    mode = p.get("execution_mode", "unknown")
    when = p.get("generated_at", "?")
    if p.get("is_real_hadoop"):
        st.success(f"Aggregates computed by a **Hadoop MapReduce job** ({mode}) at {when}. Source: `{p.get('source')}`")
    else:
        st.info(f"Aggregates computed by the **local MapReduce simulation** (same mapper/reducer, "
                f"no Hadoop cluster) at {when}. Run `bash hadoop/run_mapreduce.sh` for a real Hadoop run.")


def kpis(summary):
    workers = summary.get("total_workers_detected", 0)
    compliant = summary.get("compliant_workers", 0)
    risk = summary.get("violations_by_risk_level", {})
    c = st.columns(6)
    c[0].metric("Frames analysed", summary.get("frames_analysed", 0))
    c[1].metric("Workers detected", workers, help="Worker observations (one per person per analysed frame)")
    c[2].metric("Total violations", summary.get("total_violations", 0))
    c[3].metric("Compliance rate", f"{100 * compliant / workers:.1f}%" if workers else "n/a",
                help="Share of worker observations with no violation")
    c[4].metric("High-risk observations", risk.get("High", 0))
    c[5].metric("Avg. risk score", summary.get("average_risk_score", 0))


def tab_overview(obs, frames, mr):
    provenance_banner(mr)
    summary = mr["summary"] if mr else dl.local_summary(obs, frames)
    if not summary:
        st.info("No results yet. Run `python scripts/run_pipeline.py --input data/sample/images`.")
        return
    kpis(summary)
    left, right = st.columns(2)
    cats = pd.DataFrame(list(summary.get("violations_by_category", {}).items()), columns=["category", "count"])
    if not cats.empty:
        left.plotly_chart(px.bar(cats, x="category", y="count", title="Violations by category",
                                 color="category"), width="stretch")
    risk = pd.DataFrame([(k, summary.get("violations_by_risk_level", {}).get(k, 0)) for k in dl.RISK_ORDER],
                        columns=["risk_level", "count"])
    right.plotly_chart(px.pie(risk, names="risk_level", values="count", color="risk_level",
                              color_discrete_map=RISK_COLORS, title="Worker observations by risk level",
                              category_orders={"risk_level": dl.RISK_ORDER}), width="stretch")

    if mr:
        agg = mr["aggregates"]
        rbh = dl.risk_by_hour(agg)
        if not rbh.empty:
            st.plotly_chart(px.bar(rbh, x="hour", y="value", color="risk_level", color_discrete_map=RISK_COLORS,
                                   category_orders={"risk_level": dl.RISK_ORDER},
                                   title="Risk levels over time (per hour, from MapReduce)",
                                   labels={"value": "observations"}), width="stretch")
        a, b = st.columns(2)
        cam = dl.aggregate(agg, "violations_by_camera")
        if not cam.empty:
            a.plotly_chart(px.bar(cam, x="dimension", y="value", title="Violations by camera",
                                  labels={"dimension": "camera", "value": "violations"}), width="stretch")
        ppe = dl.ppe_status_table(agg)
        if not ppe.empty:
            long = ppe.melt(id_vars="item", var_name="status", value_name="count")
            b.plotly_chart(px.bar(long, x="item", y="count", color="status", title="PPE status per item",
                                  color_discrete_map={"worn": "#3caa3c", "missing": "#dc2828",
                                                      "unknown": "#9e9e9e"}), width="stretch")
        with st.expander("Raw MapReduce aggregates"):
            st.dataframe(agg, width="stretch", hide_index=True)
    elif not obs.empty:
        ts = obs[obs["n_violations"] > 0].set_index("timestamp").resample("h")["n_violations"].sum().reset_index()
        st.plotly_chart(px.line(ts, x="timestamp", y="n_violations", markers=True,
                                title="Violations over time (pandas)"), width="stretch")


def tab_records(obs):
    if obs.empty:
        st.info("No detection records found under outputs/*/detections/.")
        return
    c1, c2, c3 = st.columns(3)
    cams = c1.multiselect("Camera", sorted(obs["camera_id"].unique()))
    levels = c2.multiselect("Risk level", dl.RISK_ORDER, default=["Low", "Medium", "High"])
    cats = sorted({v for vs in obs["violations"] for v in vs})
    cat = c3.multiselect("Violation category", cats)
    df = obs.copy()
    if cams:
        df = df[df["camera_id"].isin(cams)]
    if levels:
        df = df[df["risk_level"].isin(levels)]
    if cat:
        df = df[df["violations"].apply(lambda vs: any(v in vs for v in cat))]
    cols = ["timestamp", "camera_id", "source", "frame_index", "worker_id", "assigned", "risk_level",
            "risk_score", "violations_str", "zones_str", "forklift_gap", "confidence", "output_folder"]
    st.caption(f"{len(df)} worker observations")
    st.dataframe(df.sort_values(["risk_score", "timestamp"], ascending=[False, True])[cols],
                 width="stretch", hide_index=True)
    st.download_button("Download as CSV", df[cols].to_csv(index=False), "safety_violations.csv", "text/csv")


def tab_gallery():
    imgs = dl.annotated_images()
    if not imgs:
        st.info("No annotated images yet.")
        return
    folders = sorted({p.parent.parent.name for p in imgs})
    folder = st.selectbox("Output folder", folders)
    sel = [p for p in imgs if p.parent.parent.name == folder]
    cols = st.columns(3)
    for i, p in enumerate(sel):
        cols[i % 3].image(str(p), caption=p.name, width="stretch")


def tab_live():
    st.write("Upload a site image to run the full pipeline (detection → PPE association → zones → risk).")
    pipe = _pipeline()
    if pipe is None:
        st.error("Model weights not found. See README → 'Model weights'.")
        return
    up = st.file_uploader("Image", type=["jpg", "jpeg", "png"])
    camera = st.text_input("Camera ID", "cam_upload")
    if up is None:
        return
    img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.error("Could not decode image.")
        return
    from datetime import datetime
    with st.spinner("Analysing..."):
        res = pipe.process_frame(img, source=up.name, camera_id=camera, frame_index=0,
                                 timestamp=datetime.now(), timestamp_source="upload")
    st.image(cv2.cvtColor(res["annotated"], cv2.COLOR_BGR2RGB), width="stretch")
    if res["observations"]:
        df = pd.DataFrame(res["observations"])
        df["violations"] = df["violations"].apply(", ".join)
        df["ppe_status"] = df["ppe_status"].apply(lambda d: ", ".join(f"{k}:{v}" for k, v in d.items()))
        st.dataframe(df[["worker_id", "assigned", "risk_level", "risk_score", "violations", "zones",
                         "ppe_status", "confidence"]], width="stretch", hide_index=True)
    else:
        st.info("No workers or violations detected.")
    st.caption("Uploaded images are analysed in memory only; they are not added to the stored results.")


def tab_model(metrics):
    if "overall" not in metrics:
        st.info("No evaluation results. Run `python scripts/evaluate_model.py`.")
    else:
        o = metrics["overall"]
        st.write(f"Held-out **test** split, {o.get('images')} images, weights `{o.get('weights')}`, "
                 f"imgsz {o.get('imgsz')}.")
        c = st.columns(5)
        for col, k in zip(c, ["precision", "recall", "f1", "mAP50", "mAP50_95"]):
            col.metric(k, f"{o[k]:.3f}")
    if "per_class_metrics" in metrics:
        pc = metrics["per_class_metrics"]
        st.plotly_chart(px.bar(pc, x="class", y=["precision", "recall", "mAP50"], barmode="group",
                               title="Per-class metrics (test split)"), width="stretch")
        st.dataframe(pc, width="stretch", hide_index=True)
    if "violation_image_level" in metrics:
        st.subheader("Image-level violation detection (test split)")
        st.dataframe(metrics["violation_image_level"], width="stretch", hide_index=True)
    if "dataset_class_distribution" in metrics:
        d = metrics["dataset_class_distribution"]
        st.plotly_chart(px.bar(d, x="class", y="instances", color="split", barmode="group",
                               title="Dataset class distribution"), width="stretch")
    if metrics.get("plots"):
        with st.expander("Evaluation plots"):
            cols = st.columns(2)
            for i, p in enumerate(metrics["plots"]):
                cols[i % 2].image(str(p), caption=p.name, width="stretch")


def main():
    st.title("🦺 Manufacturing Safety Analytics")
    st.caption("Computer-vision PPE & restricted-zone monitoring with Hadoop MapReduce analytics")
    if st.sidebar.button("Reload data"):
        st.cache_data.clear()
    obs, frames, mr, metrics = _load_all()
    cfg = load_config()
    st.sidebar.markdown("**Risk weights**")
    st.sidebar.json(cfg["risk"]["weights"], expanded=False)
    st.sidebar.markdown(f"Thresholds: Medium ≥ {cfg['risk']['thresholds']['medium']}, "
                        f"High ≥ {cfg['risk']['thresholds']['high']}")
    st.sidebar.markdown(f"Required PPE: {', '.join(cfg['ppe_policy']['required'])}")
    tabs = st.tabs(["Overview", "Violation records", "Annotated frames", "Analyse an image", "Model metrics"])
    with tabs[0]:
        tab_overview(obs, frames, mr)
    with tabs[1]:
        tab_records(obs)
    with tabs[2]:
        tab_gallery()
    with tabs[3]:
        tab_live()
    with tabs[4]:
        tab_model(metrics)


if __name__ == "__main__":
    main()
