import pandas as pd
import joblib

def load_model():
    model = joblib.load('model.joblib')
    encoder = joblib.load('encoder.joblib')
    return model, encoder

def preprocess_input(temp, humidity, visibility, wind_speed, weather_cond, encoder):
    encoded_weather = encoder.transform([weather_cond])[0]
    return [[temp, humidity, visibility, wind_speed, encoded_weather]]
