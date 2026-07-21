from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "same_srv_rate", "diff_srv_rate", "dst_host_count", "dst_host_srv_count",
]
TARGET = "label"


def generate_demo_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Create a reproducible NSL-KDD-shaped demo dataset.

    This is intended for local demonstrations and tests. It is not a substitute
    for a benchmark dataset in published evaluation.
    """
    rng = np.random.default_rng(random_state)
    attack = rng.binomial(1, 0.38, n_samples)
    protocol = rng.choice(["tcp", "udp", "icmp"], n_samples, p=[0.68, 0.22, 0.10])
    service = rng.choice(["http", "private", "domain_u", "smtp", "ftp_data", "other"], n_samples)
    flag = rng.choice(["SF", "S0", "REJ", "RSTR"], n_samples, p=[0.61, 0.20, 0.12, 0.07])

    df = pd.DataFrame({
        "duration": rng.exponential(2 + attack * 4),
        "protocol_type": protocol,
        "service": service,
        "flag": flag,
        "src_bytes": rng.lognormal(6.2 + attack * 0.7, 1.1),
        "dst_bytes": rng.lognormal(5.8 - attack * 0.5, 1.2),
        "wrong_fragment": rng.poisson(attack * 0.35),
        "urgent": rng.binomial(2, 0.01 + attack * 0.03),
        "hot": rng.poisson(0.15 + attack * 1.4),
        "num_failed_logins": rng.poisson(attack * 0.45),
        "logged_in": rng.binomial(1, 0.72 - attack * 0.40),
        "count": rng.poisson(16 + attack * 35),
        "srv_count": rng.poisson(12 + attack * 20),
        "serror_rate": np.clip(rng.beta(1 + attack * 5, 5), 0, 1),
        "srv_serror_rate": np.clip(rng.beta(1 + attack * 4, 5), 0, 1),
        "rerror_rate": np.clip(rng.beta(1 + attack * 3, 7), 0, 1),
        "same_srv_rate": np.clip(rng.beta(5 - attack * 2, 2 + attack * 2), 0, 1),
        "diff_srv_rate": np.clip(rng.beta(1 + attack * 2, 6 - attack * 2), 0, 1),
        "dst_host_count": rng.integers(1, 256, n_samples),
        "dst_host_srv_count": rng.integers(1, 256, n_samples),
    })

    # Add modest label noise so the task is not artificially perfect.
    signal = (
        attack + (df["serror_rate"] > 0.65).astype(int)
        + (df["num_failed_logins"] > 1).astype(int)
        + (df["flag"].isin(["S0", "REJ"])).astype(int)
    )
    labels = (signal >= 2).astype(int)
    flip = rng.random(n_samples) < 0.04
    df[TARGET] = np.where(flip, 1 - labels, labels)
    return df


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(FEATURE_COLUMNS + [TARGET]) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    clean = df[FEATURE_COLUMNS + [TARGET]].copy()
    clean[TARGET] = clean[TARGET].replace({"normal": 0, "attack": 1, "anomaly": 1}).astype(int)
    if not set(clean[TARGET].unique()).issubset({0, 1}):
        raise ValueError("label must contain binary values: 0/1 or normal/attack")
    return clean
