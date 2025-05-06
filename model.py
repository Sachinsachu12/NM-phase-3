import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

def load_and_train_model(data_path='dataset/accidents.csv'):
    df = pd.read_csv(data_path)
    
    # Simple feature selection (you can expand this)
    df = df[['Severity', 'Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Weather_Condition']].dropna()

    # Encode categorical features
    le = LabelEncoder()
    df['Weather_Condition'] = le.fit_transform(df['Weather_Condition'])

    X = df.drop('Severity', axis=1)
    y = df['Severity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, 'model.joblib')
    joblib.dump(le, 'encoder.joblib')
    return model
