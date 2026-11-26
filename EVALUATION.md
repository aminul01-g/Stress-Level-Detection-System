# Evaluation results — Stress Level Detection System

Measured on **2026-11-26** with seed 42, Python 3.14, scikit-learn 1.9.1.
Script: [`eval/eval_stress.py`](eval/eval_stress.py) · Raw output: [`eval_results.json`](eval_results.json)

## Status: corrected

Two defects found in the original repository were fixed in this pass:

1. The old `Model/best_model.pkl` was fitted with `stress_level` (the target) as an
   input feature, because the training notebook left the target in `x`. Its 1.0000
   accuracy was leakage, not performance.
2. `Model/predict.py` created a fresh `StandardScaler` and called `fit_transform()`
   on every single request, and its 13 API fields did not match the pickle's fitted
   feature names. Live predictions were not comparable to training.

The corrected setup:

* [`Model/train.py`](Model/train.py) is now the canonical training script. It uses
  exactly the 13 fields declared in `schema/UserInput.py`, excludes the target,
  fits `StandardScaler` and `RandomForestClassifier(300, seed=42)` in one
  `Pipeline`, and persists that pipeline as `Model/best_model.pkl`.
* [`Model/predict.py`](Model/predict.py) builds a one-row DataFrame in training
  feature order and calls the persisted pipeline. No scaler is created or fitted
  at inference time.
* `Model/metrics.json` records the measured holdout metrics from the training run.

## Dataset
1,100 rows; 13 input features; 3 classes (373 / 358 / 369).
Holdout: stratified 80/20, seed 42, 220 rows.

## Measured performance of the committed, corrected pipeline

| Metric | Value |
|---|---:|
| Accuracy | 0.8818 |
| Balanced accuracy | 0.8821 |
| Macro F1 | 0.8817 |
| Macro ROC-AUC (OvR) | 0.9879 |

Same-feature baselines on the identical holdout:

| Model | Accuracy | Balanced accuracy | Macro F1 | Macro ROC-AUC (OvR) |
|---|---:|---:|---:|---:|
| K-Nearest Neighbours | 0.8864 | 0.8869 | 0.8862 | 0.9718 |
| Random Forest (committed) | 0.8818 | 0.8821 | 0.8817 | 0.9879 |
| Logistic Regression | 0.8682 | 0.8685 | 0.8681 | 0.9798 |

The committed model is competitive with these simple baselines and its ROC-AUC is
the highest. These numbers replace the invalid 1.0 from the leaky artifact.

## Verification performed

* `Model/train.py` was executed; it wrote the new artifact and metrics.
* `Model/predict.py` was imported and exercised with a valid `UserInput`: the
  response had the expected keys, a valid class, probabilities summing to 1,
  and identical outputs on repeated calls.
* `eval/eval_stress.py` was re-run from a fresh clone of the published repository.

## Limitations

* The original notebook still contains the leaky experiment; `Model/train.py`
  supersedes it. Do not cite the notebook's 1.0 accuracy.
* A scikit-learn pickle is version-sensitive. The artifact was created with 1.9.1;
  `requirements.txt` now constrains `scikit-learn>=1.9,<2`.
* No external validation cohort was used.

## Reproduce

```bash
python -m pip install -r requirements.txt
python Model/train.py          # retrain and rewrite the artifact + metrics
python eval/eval_stress.py     # re-evaluate the committed artifact
```
