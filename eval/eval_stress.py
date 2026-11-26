#!/usr/bin/env python3
"""Reproducible evaluation of the corrected stress-level model.

The committed artifact is a fitted Pipeline (scaler + classifier) trained on the
13 fields in schema/UserInput.py, with the target excluded. This script re-runs
the same stratified 80/20 holdout (seed 42) and compares the committed pipeline
with two simple baselines on the identical feature set.
"""
import json
import pickle
import warnings
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "ModelTrain/StressLevelDataset.csv").exists() else HERE.parent
df = pd.read_csv(ROOT / "ModelTrain/StressLevelDataset.csv")


def metrics(y_true, y_pred, y_prob=None):
    out = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
    }
    if y_prob is not None:
        out["macro_roc_auc_ovr"] = round(float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")), 4)
    return out


with open(ROOT / "Model/best_model.pkl", "rb") as f:
    saved = pickle.load(f)
fit_features = [str(n) for n in getattr(saved, "feature_names_in_", [])]
y = df["stress_level"]
X = df[fit_features]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

saved_pred = saved.predict(Xte)
committed = metrics(yte, saved_pred, saved.predict_proba(Xte))

baselines = {}
for name, estimator in {
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=SEED),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=SEED),
    "KNeighbors": KNeighborsClassifier(),
}.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    pipe.fit(Xtr, ytr)
    prob = pipe.predict_proba(Xte)
    baselines[name] = metrics(yte, pipe.predict(Xte), prob)

result = {
    "evaluated_at": pd.Timestamp.utcnow().isoformat(),
    "seed": SEED,
    "dataset": {"rows": int(len(df)), "features_used": len(fit_features),
                "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()}},
    "committed_pipeline": {
        "artifact_type": type(saved).__name__,
        "fitted_feature_names": fit_features,
        "target_used_as_input": "stress_level" in fit_features,
        "metrics_on_stratified_holdout": committed,
    },
    "baselines_same_features": baselines,
    "notes": [
        "The original notebook kept stress_level in x; Model/train.py is the corrected canonical training script.",
        "The committed artifact now contains a fitted scaler, so inference only calls transform() inside the pipeline.",
        "Holdout: stratified 80/20, seed 42, 220 rows.",
    ],
}
out = ROOT / "eval_results.json"
out.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
