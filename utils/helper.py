import pandas as pd
import random
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
from requests.exceptions import ReadTimeout

questions = [
    "Difficulty falling asleep?",
    "Difficulty staying asleep?",
    "Problems waking up too early?",
    "Satisfaction with current sleep pattern?",
    "Interference with daily functioning due to sleep issues?",
    "Noticeability of sleep problems to others?",
    "Worry/distress about current sleep issues?"
]

def generate_data(n_samples=300):
    data = []
    for _ in range(n_samples):
        responses = [random.randint(0, 4) for _ in questions]
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

    df = pd.DataFrame(data, columns=questions + ["Total Score", "Insomnia Level"])
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/insomnia_synthetic.csv", index=False)
    return df

def train_and_save_model(df):
    X = df[questions]
    y = df["Insomnia Level"]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/insomnia_model.pkl")
    joblib.dump(label_encoder, "model/label_encoder.pkl")

def load_model():
    model = joblib.load("model/insomnia_model.pkl")
    label_encoder = joblib.load("model/label_encoder.pkl")
    return model, label_encoder

def get_questions():
    return questions

def get_spotify_client():
    # client_id = os.getenv("SPOTIFY_CLIENT_ID")
    # client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    client_id = "b95c8b59381c48faad3ccbe390ef41a8"
    client_secret ="ba091eaf703e4e9c9773ad5e8d98862c"
    if not client_id or not client_secret:
        raise Exception("Spotify client ID and secret must be set as environment variables.")
    auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    return sp

def get_spotify_track_link(song_name, artist_name):
    sp = get_spotify_client()
    query = f"track:{song_name} artist:{artist_name}"
    try:
        results = sp.search(q=query, type="track", limit=1)
        tracks = results.get("tracks", {}).get("items", [])
        if tracks:
            return tracks[0]["external_urls"]["spotify"]
    except ReadTimeout:
        # Handle timeout gracefully
        return None
    except requests.exceptions.RequestException:
        # Handle other request exceptions gracefully
        return None
    return None

def retrain_model_with_feedback(feedback_csv_path="data/feedback.csv"):
    import pandas as pd
    import joblib
    import os
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    # Load original synthetic data
    synthetic_data_path = "data/insomnia_synthetic.csv"
    if not os.path.exists(synthetic_data_path):
        raise FileNotFoundError(f"Original synthetic data not found at {synthetic_data_path}")

    df_synthetic = pd.read_csv(synthetic_data_path)

    # Load feedback data
    if not os.path.exists(feedback_csv_path):
        raise FileNotFoundError(f"Feedback data not found at {feedback_csv_path}")

    df_feedback = pd.read_csv(feedback_csv_path)

    # Check if feedback data has the required feature columns
    questions = [
        "Difficulty falling asleep?",
        "Difficulty staying asleep?",
        "Problems waking up too early?",
        "Satisfaction with current sleep pattern?",
        "Interference with daily functioning due to sleep issues?",
        "Noticeability of sleep problems to others?",
        "Worry/distress about current sleep issues?"
    ]

    missing_cols = [q for q in questions if q not in df_feedback.columns]
    if missing_cols:
        raise ValueError(f"Feedback data missing required feature columns: {missing_cols}")

    # Prepare feedback data for training
    df_feedback_features = df_feedback[questions]
    df_feedback_labels = df_feedback["Insomnia Level"]

    # Combine synthetic and feedback data
    df_combined_features = pd.concat([df_synthetic[questions], df_feedback_features], ignore_index=True)
    df_combined_labels = pd.concat([df_synthetic["Insomnia Level"], df_feedback_labels], ignore_index=True)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df_combined_labels)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(df_combined_features, y_encoded, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model and label encoder
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/insomnia_model.pkl")
    joblib.dump(label_encoder, "model/label_encoder.pkl")

    # Return training and test accuracy for info
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    return train_acc, test_acc
