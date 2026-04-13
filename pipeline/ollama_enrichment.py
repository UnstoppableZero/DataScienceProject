import pandas as pd
import requests
import json
import time

# loading my dataset
df = pd.read_csv('data/starter_data_enriched.csv')

# genres (no duplicates)
unique_genres = df['cultural_genre'].dropna().unique().tolist()
print(f"Unique genres to enrich: {len(unique_genres)}")

def enrich_genre(genre):
    prompt = f"""You are a music and culture expert. Given this music genre: "{genre}"

Return ONLY a JSON object with no explanation, no markdown, no extra text:
{{
  "cultural_region": "the world region this genre originates from",
  "language": "primary language typically used in this genre",
  "global_reach": "low/medium/high based on how globally mainstream this genre is",
  "cultural_notes": "one sentence describing the cultural origin"
}}"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "deepseek-r1:14b",
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        result = response.json()["response"].strip()

        # taking out the JSON 
        start = result.find("{")
        end = result.rfind("}") + 1
        if start != -1 and end != 0:
            return json.loads(result[start:end])
    except Exception as e:
        print(f"  Error enriching '{genre}': {e}")
    return None

# enrich each unique genre
print("Running Ollama enrichment...\n")
genre_labels = {}

for i, genre in enumerate(unique_genres):
    print(f"[{i+1}/{len(unique_genres)}] Enriching: {genre}")
    label = enrich_genre(genre)
    if label:
        genre_labels[genre] = label
        print(f"  -> {label}")
    time.sleep(0.5)

# mapping back to dataset
df['ollama_cultural_region'] = df['cultural_genre'].map(
    lambda g: genre_labels.get(g, {}).get('cultural_region', '')
)
df['ollama_language'] = df['cultural_genre'].map(
    lambda g: genre_labels.get(g, {}).get('language', '')
)
df['ollama_global_reach'] = df['cultural_genre'].map(
    lambda g: genre_labels.get(g, {}).get('global_reach', '')
)
df['ollama_cultural_notes'] = df['cultural_genre'].map(
    lambda g: genre_labels.get(g, {}).get('cultural_notes', '')
)

# saved
df.to_csv('data/full_data.csv', index=False)
print(f"\nDone! Saved to data/full_data.csv")
print(f"Columns: {df.columns.tolist()}")