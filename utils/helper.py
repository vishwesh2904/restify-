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
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE

# --- Constants ---
FEATURE_COLS = [
    "Insomnia Severity", "Sleep Quality", "Depression Level", "Sleep Hygiene",
    "Negative Thoughts About Sleep", "Bedtime Worrying", "Stress Level",
    "Coping Skills", "Emotion Regulation"
]

# --- Data Generation ---
def generate_data(n_samples=300):
    data = []
    for _ in range(n_samples):
        responses = [random.randint(0, 4) for _ in FEATURE_COLS]
        total = sum(responses)
        if total <= 7:
            level = "No Insomnia"
        elif total <= 14:
            level = "Mild"
        elif total <= 21:
            level = "Moderate"
        else:
            level = "Severe"
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

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_scaled, y_encoded)

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/insomnia_model.pkl")
    joblib.dump(label_encoder, "model/label_encoder.pkl")
    joblib.dump(scaler, "model/scaler.pkl")

    accuracy = model.score(X_scaled, y_encoded)
    return accuracy

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
        model_path = "model/insomnia_model.pkl"
        label_encoder_path = "model/label_encoder.pkl"
        scaler_path = "model/scaler.pkl"

    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
    scaler = joblib.load(scaler_path)

    return model, label_encoder, scaler


# --- Retrain with Feedback ---
def retrain_model_with_feedback():
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    df = pd.read_csv("data/insomnia_synthetic.csv").dropna()

    le = LabelEncoder()
    df["Insomnia Level Encoded"] = le.fit_transform(df["Insomnia Level"])

    X = df[FEATURE_COLS]
    y = df["Insomnia Level Encoded"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

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

    os.makedirs("model", exist_ok=True)
    joblib.dump(best_model, "model/insomnia_model.pkl")
    joblib.dump(le, "model/label_encoder.pkl")
    joblib.dump(scaler, "model/scaler.pkl")

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
