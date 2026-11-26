# Evaluation results — Stress Level Detection System

Measured on **2026-11-26** with seed 42, Python 3.14, scikit-learn 1.9.1.
Script: [`eval/eval_stress.py`](eval/eval_stress.py) · Raw output: [`eval_results.json`](eval_results.json)

## Dataset
1,100 rows, 20 input columns + `stress_level`. Class counts: 373 / 358 / 369.

## Finding: the committed model has target leakage

`Model/best_model.pkl` reports these fitted feature names:

```
anxiety_level, mental_health_history, depression, headache, blood_pressure,
breathing_problem, noise_level, study_load, future_career_concerns,
peer_pressure, extracurricular_activities, bullying, stress_level
```

`stress_level` — the target — is an input feature. The training notebook keeps the
target inside `x`, so the 1.0 accuracy below is an artifact of leakage, not skill:

| Evaluation | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Committed model on its own 13-feature set | 1.0000 | 1.0000 | 1.0000 |

This number must **not** be used as evidence of model quality.

## Honest, leak-free benchmark

Target excluded, stratified 80/20 split, scaling fitted inside a `Pipeline` only on
training data (seed 42):

| Model | Accuracy | Balanced accuracy | Macro F1 | Macro ROC-AUC (OvR) |
|---|---:|---:|---:|---:|
| Random Forest (300 trees) | 0.8955 | 0.8958 | 0.8953 | 0.9849 |
| Logistic Regression | 0.8818 | 0.8821 | 0.8820 | 0.9853 |
| K-Nearest Neighbours | 0.8500 | 0.8506 | 0.8499 | 0.9531 |

These are the first reproducible, non-leaky results for this repository.

## Second finding: API preprocessing is inconsistent

`Model/predict.py` sends 13 schema fields (which include `social_support`) as a raw
array and calls `self.scaler.fit_transform(features)` on every request. That means:

1. the scaler is refitted on a single row, so every feature is scaled to 0;
2. the 13 API fields do not match the 13 names the pickle was fitted on.

Live predictions are therefore not comparable to even the leaked training result.
**No application code was changed as part of this evaluation.** Fixing it requires
retraining the model without the target, persisting the fitted scaler, and using
`transform()` at inference time.

## Reproduce

```bash
python -m pip install pandas scikit-learn
python eval/eval_stress.py
```

The script writes `eval_results.json` and prints the same numbers.
