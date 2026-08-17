"""Train five purchase-intention classifiers and persist artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/"
    "online_shoppers_intention.csv"
)
LOCAL_DATA = ROOT / "data" / "online_shoppers_intention.csv"
MODEL_DIR = ROOT / "model"
TEST_CSV = ROOT / "test_data.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20
POSITIVE_LABEL = True

NUMERIC_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]
CATEGORICAL_COLUMNS = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
TARGET_COLUMN = "Revenue"

MODEL_SPECS = {
    "Logistic Regression": (
        "logistic_regression.pkl",
        LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    ),
    "Decision Tree": (
        "decision_tree.pkl",
        DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    ),
    "kNN": (
        "knn.pkl",
        KNeighborsClassifier(n_neighbors=11, weights="distance"),
    ),
    "Naive Bayes": (
        "naive_bayes.pkl",
        GaussianNB(),
    ),
    "Random Forest (Ensemble)": (
        "random_forest.pkl",
        RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    ),
}


def load_session_table() -> pd.DataFrame:
    if LOCAL_DATA.exists():
        sessions = pd.read_csv(LOCAL_DATA)
    else:
        LOCAL_DATA.parent.mkdir(parents=True, exist_ok=True)
        sessions = pd.read_csv(DATA_URL)
        sessions.to_csv(LOCAL_DATA, index=False)
    sessions[TARGET_COLUMN] = sessions[TARGET_COLUMN].astype(bool)
    sessions["Weekend"] = sessions["Weekend"].astype(str)
    return sessions


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric_scale", StandardScaler(), NUMERIC_COLUMNS),
            (
                "category_onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )


def densify(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return matrix


def build_model_pipeline(estimator) -> Pipeline:
    steps = [("encode_sessions", build_preprocessor())]
    if isinstance(estimator, GaussianNB):
        steps.append(("to_dense", FunctionTransformer(densify, validate=False)))
    steps.append(("classifier", estimator))
    return Pipeline(steps)


def score_predictions(y_true, y_pred, y_score) -> dict[str, float]:
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "AUC": round(float(roc_auc_score(y_true, y_score)), 4),
        "Precision": round(
            float(precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
            4,
        ),
        "Recall": round(
            float(recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
            4,
        ),
        "F1": round(
            float(f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
            4,
        ),
        "MCC": round(float(matthews_corrcoef(y_true, y_pred)), 4),
    }


def probability_for_purchase(fitted_pipeline: Pipeline, features: pd.DataFrame):
    classes = fitted_pipeline.named_steps["classifier"].classes_
    proba = fitted_pipeline.predict_proba(features)
    positive_index = list(classes).index(POSITIVE_LABEL)
    return proba[:, positive_index]


def train_and_persist() -> dict:
    sessions = load_session_table()
    features = sessions[FEATURE_COLUMNS]
    labels = sessions[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    test_frame = x_test.copy()
    test_frame[TARGET_COLUMN] = y_test.values
    test_frame.to_csv(TEST_CSV, index=False)

    metrics_by_model: dict[str, dict[str, float]] = {}
    for display_name, (filename, estimator) in MODEL_SPECS.items():
        pipeline = build_model_pipeline(estimator)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        purchase_scores = probability_for_purchase(pipeline, x_test)
        metrics_by_model[display_name] = score_predictions(y_test, predictions, purchase_scores)
        joblib.dump(pipeline, MODEL_DIR / filename)
        print(f"{display_name}: {metrics_by_model[display_name]}")

    schema = {
        "feature_columns": FEATURE_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "target_column": TARGET_COLUMN,
        "positive_label": POSITIVE_LABEL,
        "model_files": {name: spec[0] for name, spec in MODEL_SPECS.items()},
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "n_features": len(FEATURE_COLUMNS),
        "n_rows_full": int(len(sessions)),
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics_by_model, indent=2))
    (MODEL_DIR / "preprocess_info.json").write_text(json.dumps(schema, indent=2))
    return metrics_by_model


if __name__ == "__main__":
    train_and_persist()
