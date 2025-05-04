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
    results = sp.search(q=query, type="track", limit=1)
    tracks = results.get("tracks", {}).get("items", [])
    if tracks:
        return tracks[0]["external_urls"]["spotify"]
    return None
