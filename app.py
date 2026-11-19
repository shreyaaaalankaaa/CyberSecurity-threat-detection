from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data import FEATURE_COLUMNS, TARGET, generate_demo_dataset, validate_dataset
from src.inference import predict_record
from src.modeling import save_model, train_and_compare

st.set_page_config(page_title="Network Threat Detection", page_icon="🛡️", layout="wide")
st.title("Network Threat Detection")
st.caption("Train, compare, and inspect machine-learning models for binary network-traffic classification.")

with st.sidebar:
    st.header("Data source")
    source = st.radio("Choose a dataset", ["Built-in demonstration data", "Upload CSV"])
    samples = st.slider("Demo rows", 1000, 15000, 5000, 1000, disabled=source != "Built-in demonstration data")
    upload = st.file_uploader("CSV file", type="csv", disabled=source != "Upload CSV")
    st.info("The built-in dataset is synthetic and intended only for demonstrations. Use a recognized benchmark for formal evaluation.")

if source == "Built-in demonstration data":
    raw = generate_demo_dataset(samples)
elif upload is not None:
    try:
        raw = pd.read_csv(upload)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        st.error(f"Could not read the CSV file: {exc}")
        st.stop()
else:
    st.info("Upload a CSV file to continue.")
    st.stop()

try:
    df = validate_dataset(raw)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

normal_count = int((df[TARGET] == 0).sum())
attack_count = int((df[TARGET] == 1).sum())
a, b, c = st.columns(3)
a.metric("Connections", f"{len(df):,}")
b.metric("Normal", f"{normal_count:,}")
c.metric("Attack", f"{attack_count:,}")

with st.expander("Dataset preview", expanded=False):
    st.dataframe(df.head(100), use_container_width=True)

if st.button("Train and compare models", type="primary"):
    with st.spinner("Training models..."):
        st.session_state.results = train_and_compare(df)

results = st.session_state.get("results")
if results:
    metrics_df = pd.DataFrame([{"model": r.model_name, **r.metrics} for r in results])
    st.subheader("Model comparison")
    st.dataframe(metrics_df.style.format({c: "{:.3f}" for c in metrics_df.columns if c != "model"}), use_container_width=True)
    st.plotly_chart(px.bar(metrics_df, x="model", y=["precision", "recall", "f1", "roc_auc"], barmode="group", range_y=[0, 1]), use_container_width=True)

    best = results[0]
    st.success(f"Best model by F1 score: {best.model_name}")
    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        cm = pd.DataFrame(best.confusion, index=["Actual normal", "Actual attack"], columns=["Predicted normal", "Predicted attack"])
        st.plotly_chart(px.imshow(cm, text_auto=True, aspect="auto"), use_container_width=True)
    with right:
        st.subheader("Most influential features")
        st.plotly_chart(px.bar(best.feature_importance.sort_values("importance"), x="importance", y="feature", orientation="h"), use_container_width=True)

    path = save_model(best)
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/metrics.json").write_text(json.dumps({r.model_name: r.metrics for r in results}, indent=2))
    st.caption(f"Saved the selected model to `{path}`.")

    st.subheader("Inspect one connection")
    defaults = df[FEATURE_COLUMNS].iloc[0].to_dict()
    with st.form("prediction"):
        cols = st.columns(2)
        record = {}
        for i, name in enumerate(FEATURE_COLUMNS):
            with cols[i % 2]:
                value = defaults[name]
                if name in ["protocol_type", "service", "flag"]:
                    options = sorted(df[name].astype(str).unique())
                    record[name] = st.selectbox(name, options, index=options.index(str(value)) if str(value) in options else 0)
                elif name in ["wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "count", "srv_count", "dst_host_count", "dst_host_srv_count"]:
                    record[name] = st.number_input(name, min_value=0, value=int(value), step=1)
                else:
                    record[name] = st.number_input(name, min_value=0.0, value=float(value), format="%.4f")
        submitted = st.form_submit_button("Classify connection")
    if submitted:
        prediction = predict_record(best.pipeline, record)
        st.json(prediction)
