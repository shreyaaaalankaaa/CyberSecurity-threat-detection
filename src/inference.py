from __future__ import annotations
import pandas as pd
from .data import FEATURE_COLUMNS


def predict_record(pipeline, record: dict) -> dict:
    missing = [c for c in FEATURE_COLUMNS if c not in record]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")
    frame = pd.DataFrame([{k: record[k] for k in FEATURE_COLUMNS}])
    probability = float(pipeline.predict_proba(frame)[0, 1])
    return {
        "prediction": "attack" if probability >= 0.5 else "normal",
        "attack_probability": round(probability, 4),
        "risk_level": "high" if probability >= 0.8 else "medium" if probability >= 0.5 else "low",
    }
