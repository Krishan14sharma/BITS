"""Streamlit demo for session-level purchase intention classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from train_models import (
    FEATURE_COLUMNS,
    POSITIVE_LABEL,
    TARGET_COLUMN,
    probability_for_purchase,
    score_predictions,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
DEFAULT_TEST_CSV = ROOT / "test_data.csv"
SCHEMA_PATH = MODEL_DIR / "preprocess_info.json"

st.set_page_config(
    page_title="Shopper Purchase Intention Lab",
    page_icon="🛒",
    layout="wide",
)


@st.cache_resource
def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


@st.cache_resource
def load_fitted_models(schema: dict) -> dict:
    loaded = {}
    for display_name, filename in schema["model_files"].items():
        loaded[display_name] = joblib.load(MODEL_DIR / filename)
    return loaded


def prepare_session_frame(raw: pd.DataFrame) -> pd.DataFrame:
    prepared = raw.copy()
    missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in prepared.columns]
    if missing:
        raise ValueError(
            "Uploaded CSV is missing required columns: " + ", ".join(missing)
        )
    prepared[TARGET_COLUMN] = prepared[TARGET_COLUMN].astype(bool)
    prepared["Weekend"] = prepared["Weekend"].astype(str)
    return prepared


def plot_confusion(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=[False, True])
    figure, axis = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=["No purchase", "Purchase"],
        yticklabels=["No purchase", "Purchase"],
        ax=axis,
    )
    axis.set_xlabel("Predicted session outcome")
    axis.set_ylabel("Actual session outcome")
    axis.set_title("Confusion matrix on uploaded sessions")
    figure.tight_layout()
    return figure


def plot_metric_bars(metrics_by_model: dict[str, dict[str, float]]):
    chart_rows = []
    for model_name, scores in metrics_by_model.items():
        chart_rows.append({"Model": model_name, "Metric": "F1", "Score": scores["F1"]})
        chart_rows.append({"Model": model_name, "Metric": "AUC", "Score": scores["AUC"]})
    chart_frame = pd.DataFrame(chart_rows)
    figure, axis = plt.subplots(figsize=(8.5, 3.6))
    sns.barplot(data=chart_frame, x="Model", y="Score", hue="Metric", ax=axis)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score on test sessions")
    axis.set_title("F1 and AUC across purchase-intention models")
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    return figure


schema = load_schema()
fitted_models = load_fitted_models(schema)

st.title("Shopper Purchase Intention Lab")
st.write(
    "This app scores e-commerce **sessions** (not individual shoppers) and predicts "
    "whether the visit will end in a purchase (`Revenue = True`). "
    "Five scikit-learn classifiers were trained on the UCI Online Shoppers "
    "Purchasing Intention dataset. Upload the assignment test CSV, pick a model, "
    "and inspect live metrics plus error patterns."
)

with st.sidebar:
    st.header("Evaluation setup")
    uploaded_file = st.file_uploader(
        "Upload test sessions (CSV)",
        type=["csv"],
        help="Use only held-out test rows. The file must include the Revenue label.",
    )
    selected_model = st.selectbox(
        "Classifier to inspect",
        options=list(schema["model_files"].keys()),
        index=list(schema["model_files"].keys()).index("Random Forest (Ensemble)"),
    )
    st.caption(
        f"Training used {schema['n_train']} sessions; bundled test split has "
        f"{schema['n_test']} sessions and {schema['n_features']} input features."
    )

if uploaded_file is not None:
    raw_sessions = pd.read_csv(uploaded_file)
    source_label = "uploaded CSV"
else:
    raw_sessions = pd.read_csv(DEFAULT_TEST_CSV)
    source_label = "bundled test_data.csv"

try:
    sessions = prepare_session_frame(raw_sessions)
except ValueError as error:
    st.error(str(error))
    st.stop()

session_features = sessions[FEATURE_COLUMNS]
session_labels = sessions[TARGET_COLUMN]
pipeline = fitted_models[selected_model]
predictions = pipeline.predict(session_features)
purchase_scores = probability_for_purchase(pipeline, session_features)
live_metrics = score_predictions(session_labels, predictions, purchase_scores)

st.subheader(f"Live scores · {selected_model}")
st.caption(f"Evaluated on {len(sessions)} rows from {source_label}.")

metric_cols = st.columns(6)
for column, metric_name in zip(
    metric_cols, ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
):
    column.metric(metric_name, f"{live_metrics[metric_name]:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Classification report**")
    report_text = classification_report(
        session_labels,
        predictions,
        labels=[False, True],
        target_names=["No purchase", "Purchase"],
        digits=4,
        zero_division=0,
    )
    st.code(report_text, language="text")
with right:
    st.markdown("**Confusion matrix**")
    st.pyplot(plot_confusion(session_labels, predictions), clear_figure=True)

st.subheader("How all five models compare on this file")
comparison_rows = {}
for model_name, model_pipeline in fitted_models.items():
    model_pred = model_pipeline.predict(session_features)
    model_score = probability_for_purchase(model_pipeline, session_features)
    comparison_rows[model_name] = score_predictions(session_labels, model_pred, model_score)

comparison_table = pd.DataFrame(comparison_rows).T[
    ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
]
st.dataframe(comparison_table.style.format("{:.4f}"), use_container_width=True)
st.pyplot(plot_metric_bars(comparison_rows), clear_figure=True)

st.caption(
    "Positive class is a completed purchase. Precision, recall, F1, and MCC are "
    "computed for that class so a high accuracy from always predicting "
    "'no purchase' does not hide missed buyers."
)
