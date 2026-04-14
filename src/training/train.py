from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

FEATURES = [
    "age",
    "prior_no_shows",
    "lead_time_hours",
    "num_previous_appointments",
    "is_weekend",
    "distance_km",
]


def make_synthetic_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.integers(0, 90, n),
            "prior_no_shows": rng.integers(0, 8, n),
            "lead_time_hours": rng.uniform(0, 720, n),
            "num_previous_appointments": rng.integers(0, 20, n),
            "is_weekend": rng.integers(0, 2, n),
            "distance_km": rng.uniform(0.1, 50, n),
        }
    )

    # A simple, explainable signal for no-show risk.
    logit = (
        0.03 * df["lead_time_hours"]
        + 0.45 * df["prior_no_shows"]
        + 0.12 * df["distance_km"]
        + 0.35 * df["is_weekend"]
        - 0.015 * df["age"]
        - 2.2
    )
    prob = 1 / (1 + np.exp(-logit))
    df["target"] = (rng.uniform(0, 1, n) < prob).astype(int)
    return df


def dataframe_hash(df: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(payload).hexdigest()


def build_pipeline() -> Pipeline:
    numeric_features = [
        "age",
        "prior_no_shows",
        "lead_time_hours",
        "num_previous_appointments",
        "distance_km",
    ]
    categorical_features = ["is_weekend"]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def main() -> None:
    df = make_synthetic_data()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["target"])

    X_train = train_df[FEATURES]
    y_train = train_df["target"]
    X_test = test_df[FEATURES]
    y_test = test_df["target"]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, pred)
    report = classification_report(y_test, pred, output_dict=True)

    model_path = MODEL_DIR / "no_show_model.joblib"
    joblib.dump(pipeline, model_path)

    metadata = {
        "model_name": "no_show_risk_classifier",
        "model_version": "v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_hash": dataframe_hash(df),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "features": FEATURES,
        "metrics": {
            "roc_auc": float(roc_auc),
            "f1": float(f1),
            "precision_pos": float(report["1"]["precision"]),
            "recall_pos": float(report["1"]["recall"]),
        },
    }

    metadata_path = MODEL_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    # Keep a copy of baseline data for the drift demo.
    baseline_path = DATA_DIR / "processed" / "baseline_sample.csv"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    df.sample(1000, random_state=42)[FEATURES].to_csv(baseline_path, index=False)

    print(json.dumps(metadata["metrics"], indent=2))
    print(f"Saved model to {model_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
