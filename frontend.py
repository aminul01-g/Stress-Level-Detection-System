import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
from datetime import datetime

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
    page = st.radio("Go to", ["Home", "About", "History"])

def home_page():
    # Header
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("🧠 Stress Level Detection System")
        st.subheader("Monitor and analyze your stress levels")
    
    with col2:
        st.image("https://img.icons8.com/color/96/000000/brain.png", width=150)

    # Main form
    st.markdown("### 📝 Please fill in your details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        anxiety_level = st.slider("Anxiety Level (0-21)", 0, 21, 10)
        depression = st.slider("Depression Score (0-27)", 0, 27, 13)
        headache = st.slider("Headache Intensity (0-5)", 0, 5, 2)
        blood_pressure = st.slider("Blood Pressure Level (0-3)", 0, 3, 1)
        breathing_problem = st.slider("Breathing Problem (0-5)", 0, 5, 2)
        noise_level = st.slider("Noise Level (0-5)", 0, 5, 2)
    
    with col2:
        mental_health_history = st.selectbox("Mental Health History", ["No", "Yes"])
        study_load = st.slider("Study Load (0-5)", 0, 5, 2)
        future_career_concerns = st.slider("Future Career Concerns (0-5)", 0, 5, 3)
        peer_pressure = st.slider("Peer Pressure (0-5)", 0, 5, 2)
        extracurricular_activities = st.slider("Extracurricular Activities (0-5)", 0, 5, 3)
        bullying = st.slider("Bullying Experience (0-5)", 0, 5, 1)
        social_support = st.slider("Social Support (0-3)", 0, 3, 2)

    if st.button("Predict Stress Level", key="predict"):
        # Prepare the data
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

        # Make prediction request
        try:
            response = requests.post("http://localhost:8000/predict", json=data)
            result = response.json()

            # Display result with nice formatting
            st.markdown("### 📊 Results")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Stress Level", result["stress_level_label"])

            with col2:
                # Create donut chart for probabilities
                probs = result["probability"]
                fig = px.pie(
                    values=list(probs.values()),
                    names=list(probs.keys()),
                    hole=0.6,
                    title="Probability Distribution"
                )
                fig.update_traces(textinfo='percent+label')
                st.plotly_chart(fig)

            with col3:
                # Show recommendations based on stress level
                st.markdown("### 📋 Recommendations")
                if result["stress_level"] == 0:
                    st.success("Keep up your good stress management!")
                elif result["stress_level"] == 1:
                    st.warning("Consider some relaxation techniques.")
                else:
                    st.error("Please consider consulting a professional.")

            # Save to history
            if 'history' not in st.session_state:
                st.session_state.history = []
            
            st.session_state.history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data": data,
                "result": result
            })

        except Exception as e:
            st.error(f"Error occurred: {str(e)}")

def about_page():
    st.title("About Stress Level Detection System")
    st.markdown("""
    ### 🎯 Purpose
    This system is designed to help individuals monitor and manage their stress levels 
    using machine learning technology. By analyzing various factors that contribute to 
    stress, we provide insights and recommendations for better mental health management.

    ### 🔍 How it Works
    1. **Data Collection**: Users input various metrics related to their current state
    2. **Analysis**: Our ML model processes these inputs
    3. **Prediction**: The system predicts the stress level
    4. **Recommendations**: Based on the results, personalized suggestions are provided

    ### 📊 Features
    - Real-time stress level prediction
    - Probability distribution analysis
    - Historical tracking
    - Personalized recommendations
    """)

def history_page():
    st.title("History")
    if 'history' not in st.session_state or not st.session_state.history:
        st.info("No history available yet. Make some predictions first!")
        return

    for i, entry in enumerate(reversed(st.session_state.history)):
        with st.expander(f"Prediction {i+1} - {entry['timestamp']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Input Data")
                for key, value in entry['data'].items():
                    st.write(f"**{key}:** {value}")
            
            with col2:
                st.markdown("### Results")
                st.write(f"**Stress Level:** {entry['result']['stress_level_label']}")
                st.write("**Probabilities:**")
                for k, v in entry['result']['probability'].items():
                    st.write(f"- {k}: {v:.2%}")

# Page routing
if page == "Home":
    home_page()
elif page == "About":
    about_page()
else:
    history_page()