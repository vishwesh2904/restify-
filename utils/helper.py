import os
import random
import pandas as pd
import joblib
import base64
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE

# --- Constants ---
FEATURE_COLS = [
    "Insomnia Severity", "Sleep Quality", "Depression Level", "Sleep Hygiene",
    "Negative Thoughts About Sleep", "Bedtime Worrying", "Stress Level",
    "Coping Skills", "Emotion Regulation", "Age"
]

# --- Data Generation ---
import numpy as np

def generate_data(n_samples=300):
    import numpy as np
    data = []
    # Define feature means and stddevs per insomnia level class
    class_feature_stats = {
        "No Insomnia": {
            "Insomnia Severity": (0.5, 0.3),
            "Sleep Quality": (4.0, 0.3),
            "Depression Level": (0.5, 0.3),
            "Sleep Hygiene": (4.0, 0.3),
            "Negative Thoughts About Sleep": (0.2, 0.2),
            "Bedtime Worrying": (0.3, 0.2),
            "Stress Level": (1.0, 0.5),
            "Coping Skills": (4.0, 0.3),
            "Emotion Regulation": (4.0, 0.3),
            "Age": (40, 10)
        },
        "Mild": {
            "Insomnia Severity": (1.5, 0.5),
            "Sleep Quality": (3.0, 0.5),
            "Depression Level": (1.0, 0.5),
            "Sleep Hygiene": (3.0, 0.5),
            "Negative Thoughts About Sleep": (0.8, 0.4),
            "Bedtime Worrying": (1.0, 0.4),
            "Stress Level": (2.0, 0.7),
            "Coping Skills": (3.0, 0.5),
            "Emotion Regulation": (3.0, 0.5),
            "Age": (45, 12)
        },
        "Moderate": {
            "Insomnia Severity": (2.5, 0.5),
            "Sleep Quality": (2.0, 0.5),
            "Depression Level": (2.0, 0.5),
            "Sleep Hygiene": (2.5, 0.5),
            "Negative Thoughts About Sleep": (1.5, 0.5),
            "Bedtime Worrying": (1.8, 0.5),
            "Stress Level": (3.0, 0.7),
            "Coping Skills": (2.0, 0.5),
            "Emotion Regulation": (2.0, 0.5),
            "Age": (50, 15)
        },
        "Severe": {
            "Insomnia Severity": (3.5, 0.5),
            "Sleep Quality": (1.0, 0.5),
            "Depression Level": (3.0, 0.5),
            "Sleep Hygiene": (1.5, 0.5),
            "Negative Thoughts About Sleep": (2.5, 0.5),
            "Bedtime Worrying": (2.5, 0.5),
            "Stress Level": (4.0, 0.5),
            "Coping Skills": (1.0, 0.5),
            "Emotion Regulation": (1.0, 0.5),
            "Age": (55, 15)
        }
    }



    samples_per_class = n_samples // 4
    for level, stats in class_feature_stats.items():
        for _ in range(samples_per_class):
            responses = []
            for feature in FEATURE_COLS:
                mean, std = stats[feature]
                val = np.random.normal(mean, std)
                if feature != "Age":
                    val = max(0, min(4, val))
                    val = round(val)
                else:
                    val = max(18, min(80, val))
                    val = int(val)
                responses.append(val)
            total = sum(responses[:-1])
            data.append(responses + [total, level])

    df = pd.DataFrame(data, columns=FEATURE_COLS + ["Total Score", "Insomnia Level"])
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/insomnia_synthetic.csv", index=False)
    return df

# --- Train Initial Model ---
def train_and_save_model(df):
    X = df[FEATURE_COLS]
    y = df["Insomnia Level"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=5,
        max_depth=2,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    model.fit(X_scaled, y_encoded)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/insomnia_model.pkl")
    joblib.dump(label_encoder, "models/label_encoder.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    accuracy = model.score(X_scaled, y_encoded)
    y_pred = model.predict(X_scaled)
    f1 = f1_score(y_encoded, y_pred, average='weighted')
    cm = confusion_matrix(y_encoded, y_pred)

    print("="*40)
    print("Model Evaluation Metrics:")
    print("="*40)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"F1 Score (weighted): {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("="*40)

    return accuracy, f1, cm

# --- Load Model ---
import joblib

def load_model():
    import os
    # Load the model, label encoder, and scaler
    model_path = "models/insomnia_model.pkl"
    label_encoder_path = "models/label_encoder.pkl"
    scaler_path = "models/scaler.pkl"

    # Check if files exist in 'models/' directory, else fallback to 'model/' directory
    if not (os.path.exists(model_path) and os.path.exists(label_encoder_path) and os.path.exists(scaler_path)):
        model_path = "models/insomnia_model.pkl"
        label_encoder_path = "models/label_encoder.pkl"
        scaler_path = "models/scaler.pkl"

    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
    scaler = joblib.load(scaler_path)

    # Validate label encoder classes against current dataset labels
    import pandas as pd
    df = pd.read_csv("data/insomnia_synthetic.csv")
    dataset_labels = set(df["Insomnia Level"].unique())
    encoder_labels = set(label_encoder.classes_)

    if not dataset_labels.issubset(encoder_labels):
        # Retrain label encoder with full dataset labels
        from sklearn.preprocessing import LabelEncoder
        new_label_encoder = LabelEncoder()
        new_label_encoder.fit(df["Insomnia Level"])
        label_encoder = new_label_encoder
        # Optionally, retrain model here or notify user to retrain

    return model, label_encoder, scaler


# --- Retrain with Feedback ---
def retrain_model_with_feedback():
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    import logging
    logging.basicConfig(level=logging.INFO)
    df = pd.read_csv("data/insomnia_synthetic.csv")
    logging.info(f"Data shape before dropping NA: {df.shape}")

    # Instead of dropping NA rows, fill missing values with median of each column
    df = df.fillna(df.median(numeric_only=True))
    logging.info(f"Data shape after filling NA: {df.shape}")

    if df.isnull().values.any():
        raise ValueError("Data still contains missing values after filling. Please check the data source.")

    le = LabelEncoder()
    df["Insomnia Level Encoded"] = le.fit_transform(df["Insomnia Level"])

    X = df[FEATURE_COLS]
    y = df["Insomnia Level Encoded"]

    if X.empty or y.empty:
        raise ValueError("Feature or target data is empty after preprocessing.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    if len(X_resampled) == 0 or len(y_resampled) == 0:
        raise ValueError("No data available after SMOTE resampling.")

    # Stratified train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    rf = RandomForestClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "insomnia_model.pkl")
    label_encoder_path = os.path.join("models", "label_encoder.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")

    joblib.dump(best_model, model_path)
    joblib.dump(le, label_encoder_path)
    joblib.dump(scaler, scaler_path)

    return accuracy

# --- Spotify API Helper ---
def get_spotify_track_link(song_name, artist_name):
    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError("Spotify API credentials are missing. Please set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET.")

    auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

    token_url = "https://accounts.spotify.com/api/token"
    token_headers = {"Authorization": f"Basic {auth_header}"}
    token_data = {"grant_type": "client_credentials"}

    token_response = requests.post(token_url, headers=token_headers, data=token_data)
    if token_response.status_code != 200:
        raise Exception(f"Spotify token request failed: {token_response.json()}")

    access_token = token_response.json()["access_token"]

    search_url = "https://api.spotify.com/v1/search"
    query = f"{song_name} {artist_name}"
    search_headers = {"Authorization": f"Bearer {access_token}"}
    search_params = {"q": query, "type": "track", "limit": 1}

    response = requests.get(search_url, headers=search_headers, params=search_params)
    if response.status_code != 200:
        raise Exception(f"Spotify search failed: {response.json()}")

    try:
        return response.json()["tracks"]["items"][0]["external_urls"]["spotify"]
    except (IndexError, KeyError):
        return None

# --- Append New Feedback ---
def append_to_insomnia_data(new_entry, file_path="data/insomnia_synthetic.csv"):
    os.makedirs("data", exist_ok=True)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([new_entry])
    df.to_csv(file_path, index=False)

# --- Expose Questions ---
def get_questions():
    return FEATURE_COLS
