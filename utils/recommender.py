import pandas as pd
from utils.helper import get_spotify_track_link
from utils.spotify_utils import get_spotify_thumbnail


import os

client_id = os.getenv("SPOTIPY_CLIENT_ID")
client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")

if not client_id or not client_secret:
    raise Exception("Spotify API credentials are missing. Please set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET")

def load_lullaby_dataset():
    return pd.read_csv("data/lullaby_songs.csv")

def load_feedback_data():
    try:
        return pd.read_csv("data/feedback.csv")
    except FileNotFoundError:
        return pd.DataFrame()  # No feedback yet

def compute_song_scores(df, feedback_df):
    if feedback_df.empty or "Recommended Song" not in feedback_df.columns:
        df['score'] = 1  # Default score
        return df

    # Calculate average rating for each song
    rating_df = feedback_df.groupby("Recommended Song")["Rating"].mean().reset_index()
    rating_df.columns = ["song_label", "avg_rating"]

    # Match the format used in labeling
    df['song_label'] = df.apply(lambda row: f"{row['song_name']} by {row['artist_name']} (BPM: {row['bpm']})", axis=1)

    df = df.merge(rating_df, how="left", on="song_label")
    df['avg_rating'] = df['avg_rating'].fillna(3)  # Default rating
    df['score'] = df['avg_rating']  # Use this as sorting basis

    return df

from utils.helper import FEATURE_COLS

def predict_insomnia_level(input_data):
    from utils.helper import load_model  # Import the load_model function
    model, label_encoder, scaler = load_model()

    # Explicitly reorder input_data columns to FEATURE_COLS to ensure consistent feature order
    input_data = input_data[FEATURE_COLS]

    # Scale the input data if needed
    scaled_input = scaler.transform(input_data)

    # Get the prediction
    prediction = model.predict(scaled_input)[0]
    insomnia_level = label_encoder.inverse_transform([prediction])[0]

    return insomnia_level



def recommend_song_from_dataset(insomnia_level, num_songs=1):
    df = load_lullaby_dataset()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    feedback_df = load_feedback_data()
    df = compute_song_scores(df, feedback_df)

    if insomnia_level == "No Insomnia":
        return ["No lullaby needed."], [None], [None]

    elif insomnia_level == "Mild":
        filtered = df[(df['energy'] <= 55) & (df['happiness'] >= 30)]
    elif insomnia_level == "Moderate":
        filtered = df[(df['energy'] <= 35) & (df['danceability'] <= 55) & (df['accousticness'] >= 40)]
    else:  # Severe
        filtered = df[(df['energy'] <= 25) & (df['danceability'] <= 40) & (df['accousticness'] >= 60) & (df['happiness'] <= 50)]

    if filtered.empty:
        return ["No suitable lullaby found."], [None], [None]

    # Sort by score
    filtered = filtered.sort_values("score", ascending=False)

    # Safe sampling
    sample_size = min(num_songs, len(filtered))
    selected_songs = filtered.sample(n=sample_size)

    # print(f"Debug: Number of songs requested: {num_songs}")
    # print(f"Debug: Number of songs returned: {sample_size}")

    labels = []
    spotify_links = []
    thumbnails = []

    for _, song in selected_songs.iterrows():
        label = f"{song['song_name']} by {song['artist_name']} (BPM: {song['bpm']})"
        spotify_link = song["spotify_link"]

        # Fallback: generate link if missing
        if not spotify_link or pd.isna(spotify_link) or spotify_link.strip() == "":
            spotify_link = get_spotify_track_link(song['song_name'], song['artist_name'])

        # Extract thumbnail from Spotify link (if possible)
        thumbnail_url = get_spotify_thumbnail(spotify_link)


        labels.append(label)
        spotify_links.append(spotify_link)
        thumbnails.append(thumbnail_url)

    return labels, spotify_links, thumbnails
