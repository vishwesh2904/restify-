import pandas as pd
import random
from utils.helper import get_spotify_track_link

def load_lullaby_dataset():
    return pd.read_csv("data/lullaby_songs.csv")

def recommend_song_from_dataset(insomnia_level, num_songs=1):
    df = load_lullaby_dataset()

    # Sanitize column names to avoid KeyError due to casing/whitespace
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    if insomnia_level == "No Insomnia":
        return ["No lullaby needed."], [None]

    # Adjusted filter thresholds to match dataset scale (0-100+)
    if insomnia_level == "Mild":
        filtered = df[df['energy'] <= 40]
    elif insomnia_level == "Moderate":
        filtered = df[(df['energy'] <= 30) & (df['danceability'] <= 50)]
    else:  # Severe
        filtered = df[(df['energy'] <= 20) & (df['danceability'] <= 40)]

    if filtered.empty:
        return ["No suitable lullaby found."], [None]

    # Sample up to num_songs songs or all if less available
    sample_size = min(num_songs, len(filtered))
    songs = filtered.sample(sample_size)

    labels = []
    spotify_links = []

    for _, song in songs.iterrows():
        label = f"{song['song_name']} by {song['artist_name']} (BPM: {song['bpm']})"
        spotify_link = song["spotify_link"]

        if not spotify_link or pd.isna(spotify_link) or spotify_link.strip() == "":
            # Fetch from Spotify API if missing
            spotify_link = get_spotify_track_link(song['song_name'], song['artist_name'])

        labels.append(label)
        spotify_links.append(spotify_link)

    return labels, spotify_links
