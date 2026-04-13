import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os
import pandas as pd
import time

load_dotenv()

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET")
))

# Genre/style search terms mapped to cultural region + market code
SEARCH_QUERIES = {
    ("US", "US"):           ["hip hop", "pop", "r&b"],
    ("GB", "GB"):           ["uk garage", "brit pop", "grime"],
    ("Nigeria", "NG"):      ["afrobeats", "afropop", "naija"],
    ("Ghana", "GH"):        ["highlife", "azonto", "ghana pop"],
    ("SouthAfrica", "ZA"):  ["amapiano", "kwaito", "south africa house"],
    ("Brazil", "BR"):       ["funk carioca", "sertanejo", "bossa nova"],
    ("Mexico", "MX"):       ["reggaeton", "banda", "norteÃ±o"],
    ("Colombia", "CO"):     ["cumbia", "vallenato", "colombia pop"],
    ("Korea", "KR"):        ["k-pop", "korean indie", "k-rnb"],
    ("Japan", "JP"):        ["j-pop", "city pop", "j-rock"],
    ("India", "IN"):        ["bollywood", "hindi pop", "punjabi"],
    ("France", "FR"):       ["french pop", "chanson", "french rap"],
    ("Germany", "DE"):      ["german pop", "schlager", "deutschrap"],
    ("Spain", "ES"):        ["flamenco pop", "spanish pop", "latin pop"],
}

def search_tracks(query, region, market="US", limit=10):
    tracks = []
    try:
        results = sp.search(q=query, type="track", limit=limit, market=market)
        for item in results["tracks"]["items"]:
            if not item or not item.get("id"):
                continue
            artist = item["artists"][0]
            tracks.append({
                "track_id": item["id"],
                "track_name": item.get("name", ""),
                "artist_name": artist.get("name", ""),
                "artist_id": artist.get("id", ""),
                "popularity": item.get("popularity", 0),
                "region": region
            })
    except Exception as e:
        print(f"  Error searching '{query}' for {region}: {e}")
    return tracks

def get_audio_features(track_ids):
    features = []
    for i in range(0, len(track_ids), 100):
        batch = track_ids[i:i+100]
        try:
            results = sp.audio_features(batch)
            features.extend([f for f in results if f])
        except Exception as e:
            print(f"  Error fetching audio features: {e}")
        time.sleep(0.2)
    return features

def get_artist_data(artist_ids):
    artist_data = {}
    for i in range(0, len(artist_ids), 50):
        batch = artist_ids[i:i+50]
        try:
            results = sp.artists(batch)
            for artist in results["artists"]:
                if artist:
                    artist_data[artist["id"]] = {
                        "artist_genres": ", ".join(artist.get("genres", [])),
                        "artist_followers": artist["followers"]["total"]
                    }
        except Exception as e:
            print(f"  Error fetching artist data: {e}")
        time.sleep(0.2)
    return artist_data

# --- Main Collection Loop ---
all_tracks = []
seen_ids = set()

print("Starting data collection...\n")

for (region, market), queries in SEARCH_QUERIES.items():
    print(f"Collecting tracks for region: {region}")
    for query in queries:
        tracks = search_tracks(query, region, market=market, limit=10)
        for t in tracks:
            if t["track_id"] not in seen_ids:
                seen_ids.add(t["track_id"])
                all_tracks.append(t)
        time.sleep(0.3)

print(f"\nUnique tracks collected: {len(all_tracks)}")

# --- Audio Features ---
print("Fetching audio features...")
track_ids = [t["track_id"] for t in all_tracks]
audio_features = get_audio_features(track_ids)
audio_map = {f["id"]: f for f in audio_features}

# --- Artist Data ---
print("Fetching artist data...")
artist_ids = list(set([t["artist_id"] for t in all_tracks]))
artist_map = get_artist_data(artist_ids)

# --- Merge ---
print("Merging data...")
rows = []
for track in all_tracks:
    af = audio_map.get(track["track_id"], {})
    ar = artist_map.get(track["artist_id"], {})
    rows.append({
        "track_id": track["track_id"],
        "track_name": track["track_name"],
        "artist_name": track["artist_name"],
        "artist_id": track["artist_id"],
        "region": track["region"],
        "popularity": track["popularity"],
        "artist_genres": ar.get("artist_genres", ""),
        "artist_followers": ar.get("artist_followers", 0),
        "danceability": af.get("danceability"),
        "energy": af.get("energy"),
        "loudness": af.get("loudness"),
        "speechiness": af.get("speechiness"),
        "acousticness": af.get("acousticness"),
        "instrumentalness": af.get("instrumentalness"),
        "liveness": af.get("liveness"),
        "valence": af.get("valence"),
        "tempo": af.get("tempo"),
        "duration_ms": af.get("duration_ms"),
        "time_signature": af.get("time_signature"),
    })

df = pd.DataFrame(rows)

os.makedirs("data", exist_ok=True)
df.to_csv("data/starter_data.csv", index=False)

print(f"\nDone! {len(df)} tracks saved to data/starter_data.csv")
print(df.head())