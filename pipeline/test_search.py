import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os

load_dotenv()

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET")
))

results = sp.search(q="afrobeats", type="track", limit=10, market="US")
for item in results["tracks"]["items"]:
    print(item["name"], "-", item["artists"][0]["name"])