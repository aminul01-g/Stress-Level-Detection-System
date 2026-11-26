#!/usr/bin/env python3
"""Measured audit of the Stress-Level-Detection-System committed model.

Produces three evidence blocks:
  1. exact feature names the pickled model was fitted on,
  2. its accuracy when the training feature set (which includes the target) is
     supplied -- demonstrating target leakage,
  3. a leak-free benchmark on the same holdout for honest comparison.
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

# 1) Saved model on its own fitted feature set, notebook's 80/20 split.
X_leak = df[fit_features]
y = df["stress_level"]
Xtr, Xte, ytr, yte = train_test_split(X_leak, y, test_size=0.2, random_state=SEED)
scaler = StandardScaler().fit(Xtr)
Xte_scaled = pd.DataFrame(scaler.transform(Xte), columns=fit_features, index=Xte.index)
saved_pred = saved.predict(Xte_scaled)
saved_result = metrics(yte, saved_pred)

# 2) Leak-free benchmark: target excluded, stratified split, scaling in-pipeline.
features_clean = [c for c in df.columns if c != "stress_level"]
Xc = df[features_clean]
Xtr2, Xte2, ytr2, yte2 = train_test_split(
    Xc, y, test_size=0.2, stratify=y, random_state=SEED)
models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=SEED),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=SEED),
    "KNeighbors": KNeighborsClassifier(),
}
benchmark = {}
for name, estimator in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    pipe.fit(Xtr2, ytr2)
    prob = pipe.predict_proba(Xte2)
    benchmark[name] = metrics(yte2, pipe.predict(Xte2), prob)

result = {
    "evaluated_at": pd.Timestamp.utcnow().isoformat(),
    "seed": SEED,
    "dataset": {"rows": int(len(df)), "columns": int(df.shape[1]),
                "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()}},
    "committed_model": {
        "fitted_feature_names": fit_features,
        "target_used_as_input": "stress_level" in fit_features,
        "metrics_on_its_own_feature_set": saved_result,
    },
    "leak_free_benchmark": {
        "features_used": len(features_clean),
        "target_excluded": True,
        "split": "stratified 80/20, seed 42",
        "models": benchmark,
    },
    "notes": [
        "The training notebook keeps stress_level in x, so the committed model was fitted on the target.",
        "Model/predict.py passes 13 API fields as a raw array and calls scaler.fit_transform() per request;",
        "the scaler is never fitted on training data, so live predictions are not comparable to training.",
        "The pickle was created with sklearn 1.3.2; evaluation used 1.9.1.",
    ],
}
out = ROOT / "eval_results.json"
out.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
