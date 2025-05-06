import streamlit as st
import pandas as pd
import joblib

# Load your trained ML model
model = joblib.load("traffic_accident_model.pkl")  # Replace with your model filename

st.title("AI-Driven Traffic Accident Prediction")

st.markdown("""
This app predicts the likelihood or severity of a traffic accident based on input features like weather, road conditions, time, etc.
""")

# Example input fields – customize as per your model features
weather = st.selectbox("Weather Condition", ["Clear", "Rain", "Snow", "Fog"])
road_surface = st.selectbox("Road Surface Condition", ["Dry", "Wet", "Snow", "Ice"])
light_condition = st.selectbox("Lighting Condition", ["Daylight", "Darkness - lights lit", "Darkness - no lighting"])
hour = st.slider("Hour of Day", 0, 23)

# Convert categorical inputs to numeric if needed (dummy encoding, label encoding, etc.)
# For simplicity, we'll use label encoding manually:
weather_dict = {"Clear": 0, "Rain": 1, "Snow": 2, "Fog": 3}
road_dict = {"Dry": 0, "Wet": 1, "Snow": 2, "Ice": 3}
light_dict = {"Daylight": 0, "Darkness - lights lit": 1, "Darkness - no lighting": 2}

features = [[
    weather_dict[weather],
    road_dict[road_surface],
    light_dict[light_condition],
    hour
]]

# Prediction
if st.button("Predict Accident Risk"):
    prediction = model.predict(features)
    st.success(f"Predicted Accident Severity: {prediction[0]}")
