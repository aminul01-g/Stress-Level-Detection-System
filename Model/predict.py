import pickle
import numpy as np
import os
from fastapi import HTTPException
from schema.UserInput import UserInput
from sklearn.preprocessing import StandardScaler

class StressPredictor:
    def __init__(self):
        try:
            # Load the trained model
            with open('Model/best_model.pkl', 'rb') as file:
                self.model = pickle.load(file)
            self.scaler = StandardScaler()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        
    def preprocess_data(self, data: UserInput) -> np.ndarray:
        # Convert input data to feature array in correct order
        features = np.array([[
            data.anxiety_level,
            data.mental_health_history,
            data.depression,
            data.headache,
            data.blood_pressure,
            data.breathing_problem,
            data.noise_level,
            data.study_load,
            data.future_career_concerns,
            data.peer_pressure,
            data.extracurricular_activities,
            data.bullying,
            data.social_support
        ]])
        
        # Scale the features
        return self.scaler.fit_transform(features)
    
    def predict(self, data: UserInput) -> dict:
        try:
            # Preprocess the input data
            processed_data = self.preprocess_data(data)
            
            # Make prediction
            prediction = self.model.predict(processed_data)[0]
            probability = self.model.predict_proba(processed_data)[0]
            
            # Create response
            stress_levels = {
                0: "Low Stress",
                1: "Medium Stress",
                2: "High Stress"
            }
            
            return {
                "stress_level": int(prediction),
                "stress_level_label": stress_levels[prediction],
                "probability": {
                    "low_stress": float(probability[0]),
                    "medium_stress": float(probability[1]) if len(probability) > 1 else 0.0,
                    "high_stress": float(probability[2]) if len(probability) > 2 else 0.0
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")