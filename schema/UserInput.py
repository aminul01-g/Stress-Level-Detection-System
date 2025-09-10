from pydantic import BaseModel, Field, field_validator
from typing import Literal, Annotated 

class UserInput(BaseModel):
    anxiety_level: int = Field(..., ge=0, le=21, description="Anxiety level score (0-21)")
    mental_health_history: int = Field(..., ge=0, le=1, description="Mental health history (0: No, 1: Yes)")
    depression: int = Field(..., ge=0, le=27, description="Depression score (0-27)")
    headache: int = Field(..., ge=0, le=5, description="Headache intensity (0-5)")
    blood_pressure: int = Field(..., ge=0, le=3, description="Blood pressure level (0-3)")
    breathing_problem: int = Field(..., ge=0, le=5, description="Breathing problem severity (0-5)")
    noise_level: int = Field(..., ge=0, le=5, description="Environmental noise level (0-5)")
    study_load: int = Field(..., ge=0, le=5, description="Academic study load (0-5)")
    future_career_concerns: int = Field(..., ge=0, le=5, description="Level of future career concerns (0-5)")
    peer_pressure: int = Field(..., ge=0, le=5, description="Level of peer pressure (0-5)")
    extracurricular_activities: int = Field(..., ge=0, le=5, description="Level of involvement in extracurricular activities (0-5)")
    bullying: int = Field(..., ge=0, le=5, description="Experience of bullying (0-5)")
    social_support: int = Field(..., ge=0, le=3, description="Level of social support (0-3)")

    class Config:
        schema_extra = {
            "example": {
                "anxiety_level": 14,
                "mental_health_history": 0,
                "depression": 11,
                "headache": 2,
                "blood_pressure": 1,
                "breathing_problem": 4,
                "noise_level": 2,
                "study_load": 2,
                "future_career_concerns": 3,
                "peer_pressure": 3,
                "extracurricular_activities": 3,
                "bullying": 2
            }
        }