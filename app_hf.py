import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from fastapi import FastAPI, HTTPException
import json
from schema.UserInput import UserInput
from Model.predict import StressPredictor
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from threading import Thread

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

@app.get("/api")
async def root():
    return {
        "message": "Welcome to Stress Level Detection API",
        "status": "active"
    }

@app.post("/api/predict")
async def predict_stress(data: UserInput):
    try:
        # Make prediction
        result = predictor.predict(data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Set page configuration
st.set_page_config(
    page_title="Stress Level Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }
    .stProgress .st-bo {
        background-color: #FF4B4B;
    }
    div.block-container {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/mental-health.png", width=100)
    st.title("Navigation")
    page = st.radio("Go to", ["Home", "About"])

def home_page():
    # Header
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("🧠 Stress Level Detection System")
        st.subheader("Monitor and analyze your stress levels")

    # Input form
    with st.form("stress_detection_form"):
        st.write("### Enter Your Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            anxiety_level = st.slider("Anxiety Level", 0, 21, 10)
            mental_health_history = st.selectbox("Mental Health History", ["No", "Yes"])
            depression = st.slider("Depression Level", 0, 27, 13)
            headache = st.slider("Headache Intensity", 0, 5, 2)
            noise_level = st.slider("Noise Level", 0, 5, 2)
            
        with col2:
            blood_pressure = st.slider("Blood Pressure Level", 0, 3, 1)
            breathing_problem = st.slider("Breathing Problem", 0, 5, 2)
            study_load = st.slider("Study Load", 0, 5, 2)
            future_career_concerns = st.slider("Future Career Concerns", 0, 5, 3)
            peer_pressure = st.slider("Peer Pressure", 0, 5, 2)
            
        with col3:
            extracurricular_activities = st.slider("Extracurricular Activities", 0, 5, 2)
            bullying = st.slider("Bullying Experience", 0, 5, 0)
            social_support = st.slider("Social Support", 0, 3, 2)
            
        with col2:
            headache = st.slider("Headache Frequency", 0, 5, 2)
            blood_pressure = st.slider("Blood Pressure Level", 0, 5, 2)
            sleep_quality = st.slider("Sleep Quality", 0, 5, 3)
            breathing_problem = st.slider("Breathing Problem Level", 0, 5, 2)
            
        # Convert mental health history to binary
        mental_health_history = 1 if mental_health_history == "Yes" else 0
        
        submitted = st.form_submit_button("Analyze Stress Level")
        
        if submitted:
            # Prepare data for prediction
            data = {
                "anxiety_level": anxiety_level,
                "mental_health_history": 1 if mental_health_history == "Yes" else 0,
                "depression": depression,
                "headache": headache,
                "blood_pressure": blood_pressure,
                "breathing_problem": breathing_problem,
                "noise_level": noise_level,
                "study_load": study_load,
                "future_career_concerns": future_career_concerns,
                "peer_pressure": peer_pressure,
                "extracurricular_activities": extracurricular_activities,
                "bullying": bullying,
                "social_support": social_support
            }
            
            try:
                # Make prediction using the local predictor
                result = predictor.predict(UserInput(**data))
                
                # Display results
                st.success("Analysis Complete!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### Results")
                    st.write(f"**Predicted Stress Level:** {result['stress_level']}")
                    st.write(f"**Confidence Score:** {result['confidence_score']:.2f}")
                    
                with col2:
                    # Create a gauge chart for stress level
                    fig = px.pie(values=[result['stress_level'], 100-result['stress_level']], 
                               names=['Stress', 'Normal'],
                               hole=0.7,
                               color_discrete_sequence=['#FF4B4B', '#E8E8E8'])
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig)
                
                # Save to history
                if 'history' not in st.session_state:
                    st.session_state.history = []
                
                st.session_state.history.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'data': data,
                    'result': result
                })
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

def about_page():
    st.title("About Stress Level Detection System")
    st.write("""
    This system uses machine learning to predict stress levels based on various physical and psychological factors.
    It takes into account:
    - Anxiety levels
    - Self-esteem
    - Mental health history
    - Depression levels
    - Physical symptoms (headache, blood pressure, sleep quality, breathing problems)
    
    The system provides instant analysis and helps track stress levels over time.
    """)

# Page routing
if page == "Home":
    home_page()
elif page == "About":
    about_page()

def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Run FastAPI in a separate thread
if __name__ == "__main__":
    Thread(target=run_fastapi, daemon=True).start()