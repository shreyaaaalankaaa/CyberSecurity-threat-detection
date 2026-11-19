from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import FEATURE_COLUMNS, TARGET

CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL]

@dataclass
class TrainingResult:
    model_name: str
    pipeline: Pipeline
    metrics: dict[str, float]
    confusion: list[list[int]]
    feature_importance: pd.DataFrame


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])


def model_catalog(random_state: int = 42) -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=220, max_depth=18, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=140, learning_rate=0.06, max_depth=3, random_state=random_state),
    }


def _importance_table(pipeline: Pipeline) -> pd.DataFrame:
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    estimator = pipeline.named_steps["classifier"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        values = np.zeros(len(names))
    return pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=False).head(15)


def train_and_compare(df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42) -> list[TrainingResult]:
    X, y = df[FEATURE_COLUMNS], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    results: list[TrainingResult] = []
    for name, estimator in model_catalog(random_state).items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator),
        ])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob),
        }
        results.append(TrainingResult(name, pipeline, metrics, confusion_matrix(y_test, pred).tolist(), _importance_table(pipeline)))
    return sorted(results, key=lambda r: r.metrics["f1"], reverse=True)


def save_model(result: TrainingResult, path: str | Path = "models/best_model.joblib") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model_name": result.model_name, "pipeline": result.pipeline, "metrics": result.metrics}, path)
    return path


def load_model(path: str | Path = "models/best_model.joblib") -> dict[str, Any]:
    return joblib.load(path)
