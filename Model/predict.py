"""Inference for the corrected stress-level model.

The persisted artifact is a fitted Pipeline (StandardScaler + RandomForest).
The scaler is fitted once during training and reused here via transform();
never refit it on a single request.
"""
import pickle
from pathlib import Path

import pandas as pd
from fastapi import HTTPException

from schema.UserInput import UserInput

FEATURES = [
    "anxiety_level", "mental_health_history", "depression", "headache",
    "blood_pressure", "breathing_problem", "noise_level", "study_load",
    "future_career_concerns", "peer_pressure", "extracurricular_activities",
    "bullying", "social_support",
]


class StressPredictor:
    def __init__(self):
        artifact = Path(__file__).with_name("best_model.pkl")
        try:
            with open(artifact, "rb") as f:
                self.model = pickle.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    def preprocess_data(self, data: UserInput) -> pd.DataFrame:
        """Build a one-row frame in the exact feature order used at training."""
        values = [getattr(data, name) for name in FEATURES]
        return pd.DataFrame([values], columns=FEATURES)

    def predict(self, data: UserInput) -> dict:
        try:
            processed = self.preprocess_data(data)
            prediction = int(self.model.predict(processed)[0])
            probability = self.model.predict_proba(processed)[0]

            stress_levels = {0: "Low Stress", 1: "Medium Stress", 2: "High Stress"}
            return {
                "stress_level": prediction,
                "stress_level_label": stress_levels.get(prediction, "Unknown"),
                "probability": {
                    "low_stress": float(probability[0]),
                    "medium_stress": float(probability[1]) if len(probability) > 1 else 0.0,
                    "high_stress": float(probability[2]) if len(probability) > 2 else 0.0,
                },
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
