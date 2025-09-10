from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schema.UserInput import UserInput
from Model.predict import StressPredictor

# Initialize FastAPI app
app = FastAPI(
    title="Stress Level Detection API",
    description="API for predicting stress levels based on various factors",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the stress predictor
predictor = StressPredictor()

@app.get("/")
async def root():
    return {
        "message": "Welcome to Stress Level Detection API",
        "status": "active",
        "endpoints": {
            "/predict": "POST - Make stress level predictions",
            "/": "GET - This help message"
        }
    }

@app.post("/predict")
async def predict_stress(data: UserInput):
    try:
        # Make prediction
        result = predictor.predict(data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))