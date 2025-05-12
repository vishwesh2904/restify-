import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
    df = pd.read_csv("data/insomnia_synthetic.csv").dropna()

    # Features and Labels
    from utils.helper import FEATURE_COLS
    X = df[FEATURE_COLS]
    y = df["Insomnia Level"]

    # Scaling features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Label Encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train Model (RandomForest as an example)
    model = RandomForestClassifier()
    model.fit(X_scaled, y_encoded)

    # Save model, label encoder, and scaler
    joblib.dump(model, "models/insomnia_model.pkl")
    joblib.dump(label_encoder, "models/label_encoder.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("Model, label encoder, and scaler saved successfully!")
    return model.score(X_scaled, y_encoded)
