# Stress Level Detection System - Build and Deployment Process

This document outlines the complete process of building, testing, and deploying the Stress Level Detection System, including both the FastAPI backend and Streamlit frontend.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Project Setup](#project-setup)
- [Backend Development](#backend-development)
- [Frontend Development](#frontend-development)
- [Docker Deployment](#docker-deployment)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before starting, ensure you have the following installed:
- Python 3.11 or higher
- pip (Python package manager)
- Docker
- Git

You can check your installations with:
```bash
python --version
pip --version
docker --version
git --version
```

## Project Setup

1. **Create Project Directory**
```bash
mkdir Project
cd Project
```

2. **Set Up Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install Required Packages**
```bash
pip install fastapi uvicorn scikit-learn numpy pandas streamlit plotly requests
```

## Backend Development

1. **Create Project Structure**
```bash
mkdir -p Project1/{Model,schema,ModelTrain}
cd Docker1
```

2. **Create Schema (UserInput.py)**
```python
# schema/UserInput.py
from pydantic import BaseModel, Field

class UserInput(BaseModel):
    anxiety_level: int = Field(..., ge=0, le=21)
    mental_health_history: int = Field(..., ge=0, le=1)
    depression: int = Field(..., ge=0, le=27)
    headache: int = Field(..., ge=0, le=5)
    blood_pressure: int = Field(..., ge=0, le=3)
    breathing_problem: int = Field(..., ge=0, le=5)
    noise_level: int = Field(..., ge=0, le=5)
    study_load: int = Field(..., ge=0, le=5)
    future_career_concerns: int = Field(..., ge=0, le=5)
    peer_pressure: int = Field(..., ge=0, le=5)
    extracurricular_activities: int = Field(..., ge=0, le=5)
    bullying: int = Field(..., ge=0, le=5)
    social_support: int = Field(..., ge=0, le=3)
```

3. **Create Prediction Module (predict.py)**
```python
# Model/predict.py
import pickle
import numpy as np
from fastapi import HTTPException
from schema.UserInput import UserInput
from sklearn.preprocessing import StandardScaler

class StressPredictor:
    def __init__(self):
        try:
            with open('Model/best_model.pkl', 'rb') as file:
                self.model = pickle.load(file)
            self.scaler = StandardScaler()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
```

4. **Create FastAPI Application (app.py)**
```python
# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schema.UserInput import UserInput
from Model.predict import StressPredictor

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])
predictor = StressPredictor()

@app.post("/predict")
async def predict_stress(data: UserInput):
    return predictor.predict(data)
```

## Frontend Development

1. **Create Streamlit Frontend (frontend.py)**
```python
# frontend.py
import streamlit as st
import requests
import plotly.express as px

# Set up page configuration
st.set_page_config(page_title="Stress Level Detection", page_icon="🧠", layout="wide")
```

2. **Run Frontend Development Server**
```bash
streamlit run frontend.py
```

## Docker Deployment

1. **Create Dockerfile**
```dockerfile
# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. **Create Requirements File**
```bash
pip freeze > requirements.txt
```

3. **Build and Push Docker Image**
```bash
# Build the image
docker build -t stress-level-detection-system .

# Tag the image
docker tag stress-level-detection-system username/stress-level-detection-system:latest

# Login to Docker Hub
docker login

# Push the image
docker push username/stress-level-detection-system:latest
```

4. **Run the Container**
```bash
docker run -d -p 8000:8000 username/stress-level-detection-system:latest
```

## Testing

1. **Test FastAPI Backend**
- Open browser and navigate to: http://localhost:8000/docs
- Try the /predict endpoint with sample data

2. **Test Streamlit Frontend**
- Open browser and navigate to: http://localhost:8501
- Input test values and verify predictions

## Troubleshooting

### Common Issues and Solutions

1. **ModuleNotFoundError**
```bash
# Solution: Activate virtual environment and reinstall packages
source .venv/bin/activate
pip install -r requirements.txt
```

2. **Port Already in Use**
```bash
# Solution: Kill the process using the port
sudo lsof -i :8000  # Find process
sudo kill -9 <PID>  # Kill process
```

3. **Docker Push Access Denied**
```bash
# Solution: Ensure proper login and image tagging
docker logout
docker login
docker tag image_name username/image_name:latest
```

4. **Model Loading Error**
- Verify model file path is correct
- Ensure scikit-learn versions match between training and deployment

### Best Practices

1. **Virtual Environment**
- Always use virtual environment
- Keep requirements.txt updated

2. **Docker**
- Use .dockerignore for unnecessary files
- Build optimized images
- Tag images properly

3. **Testing**
- Test API endpoints before deployment
- Validate input data ranges
- Monitor system resources

## Additional Resources

- FastAPI Documentation: https://fastapi.tiangolo.com/
- Streamlit Documentation: https://docs.streamlit.io/
- Docker Documentation: https://docs.docker.com/
- Scikit-learn Documentation: https://scikit-learn.org/stable/

## Support

For additional support:
1. Check the project's GitHub issues
2. Review the error logs
3. Contact the development team

Remember to always check logs for detailed error messages when troubleshooting:
```bash
docker logs container_name
uvicorn app:app --reload --log-level debug
```
