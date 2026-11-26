#!/usr/bin/env python3
"""Canonical training script for the corrected stress-level model.

The original notebook kept the target column inside the feature matrix, so the
old best_model.pkl learned from `stress_level` itself. This script:
  * uses exactly the 13 fields declared in schema/UserInput.py,
  * excludes the target,
  * fits StandardScaler and the classifier together in a Pipeline,
  * persists the fitted Pipeline (scaler + model) as Model/best_model.pkl,
  * writes measured holdout metrics to Model/metrics.json.

Run from the repository root:  python Model/train.py
"""
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
FEATURES = [
    "anxiety_level", "mental_health_history", "depression", "headache",
    "blood_pressure", "breathing_problem", "noise_level", "study_load",
    "future_career_concerns", "peer_pressure", "extracurricular_activities",
    "bullying", "social_support",
]

df = pd.read_csv(ROOT / "ModelTrain/StressLevelDataset.csv")
missing = set(FEATURES) - set(df.columns)
if missing:
    raise ValueError(f"Dataset is missing expected fields: {sorted(missing)}")
X, y = df[FEATURES], df["stress_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=300, random_state=SEED)),
])
model.fit(X_train, y_train)

prob = model.predict_proba(X_test)
pred = model.predict(X_test)
metrics = {
    "seed": SEED,
    "features": FEATURES,
    "target_excluded": True,
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "accuracy": round(float(accuracy_score(y_test, pred)), 4),
    "balanced_accuracy": round(float(balanced_accuracy_score(y_test, pred)), 4),
    "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
    "macro_roc_auc_ovr": round(float(
        roc_auc_score(y_test, prob, multi_class="ovr", average="macro")), 4),
}

artifact = ROOT / "Model/best_model.pkl"
with open(artifact, "wb") as f:
    pickle.dump(model, f)
(ROOT / "Model/metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps(metrics, indent=2))
print(f"Saved {artifact}")
