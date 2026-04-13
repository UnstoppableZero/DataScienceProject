import pandas as pd
import requests
import json
import time

# Load the enriched dataset
df = pd.read_csv('data/starter_data_enriched.csv')

unique_genres = df['cultural_genre'].dropna().unique().tolist()
print(f"Unique genres to enrich: {len(unique_genres)}")

# Hard-coded labels for reliable/known genres
HARDCODED_LABELS = {
    "afrobeat":   {"cultural_region": "West Africa", "language": "Yoruba/English", "global_reach": "high", "cultural_notes": "Originated in Nigeria, blending African rhythms with jazz and funk."},
    "anime":      {"cultural_region": "Japan", "language": "Japanese", "global_reach": "high", "cultural_notes": "Music tied to Japanese animated media, blending pop and orchestral styles."},
    "blues":      {"cultural_region": "Southern United States", "language": "English", "global_reach": "high", "cultural_notes": "Born from African American work songs and spirituals in the Deep South."},
    "british":    {"cultural_region": "United Kingdom", "language": "English", "global_reach": "high", "cultural_notes": "Reflects the diverse and globally influential UK music scene."},
    "country":    {"cultural_region": "Southern United States", "language": "English", "global_reach": "high", "cultural_notes": "Rooted in rural American folk and blues traditions."},
    "french":     {"cultural_region": "France", "language": "French", "global_reach": "medium", "cultural_notes": "Encompasses chanson, pop, and electronic music from France."},
    "garage":     {"cultural_region": "United Kingdom", "language": "English", "global_reach": "medium", "cultural_notes": "UK garage emerged from London's club scene in the 1990s."},
    "german":     {"cultural_region": "Germany", "language": "German", "global_reach": "medium", "cultural_notes": "Spans classical heritage to modern Deutschrap and electronic music."},
    "hip-hop":    {"cultural_region": "United States", "language": "English", "global_reach": "high", "cultural_notes": "Originated in African American communities in the Bronx, New York."},
    "indian":     {"cultural_region": "South Asia", "language": "Hindi/Tamil/Telugu", "global_reach": "medium", "cultural_notes": "Rooted in India's rich classical and folk musical traditions."},
    "j-pop":      {"cultural_region": "Japan", "language": "Japanese", "global_reach": "high", "cultural_notes": "Japan's dominant pop genre blending Western influences with local aesthetics."},
    "j-rock":     {"cultural_region": "Japan", "language": "Japanese", "global_reach": "medium", "cultural_notes": "Japanese rock blending Western rock with local cultural elements."},
    "jazz":       {"cultural_region": "United States", "language": "English", "global_reach": "high", "cultural_notes": "Emerged from African American communities in New Orleans in the early 20th century."},
    "k-pop":      {"cultural_region": "South Korea", "language": "Korean", "global_reach": "high", "cultural_notes": "Highly produced South Korean pop known for synchronized choreography and fandom culture."},
    "latin":      {"cultural_region": "Latin America", "language": "Spanish", "global_reach": "high", "cultural_notes": "Broad genre spanning reggaeton, salsa, and pop from Latin America."},
    "pagode":     {"cultural_region": "Brazil", "language": "Portuguese", "global_reach": "low", "cultural_notes": "A festive Brazilian samba subgenre originating in Rio de Janeiro."},
    "pop":        {"cultural_region": "United States", "language": "English", "global_reach": "high", "cultural_notes": "Mainstream popular music originating in the US with massive global influence."},
    "r-n-b":      {"cultural_region": "United States", "language": "English", "global_reach": "high", "cultural_notes": "Rhythm and blues originating from African American communities in the US."},
    "reggaeton":  {"cultural_region": "Latin America", "language": "Spanish", "global_reach": "high", "cultural_notes": "Emerged from Puerto Rican hip-hop and Caribbean rhythms in the late 1990s."},
    "salsa":      {"cultural_region": "Caribbean/Latin America", "language": "Spanish", "global_reach": "high", "cultural_notes": "Vibrant dance music blending African, Caribbean, and Latin influences."},
    "samba":      {"cultural_region": "Brazil", "language": "Portuguese", "global_reach": "medium", "cultural_notes": "Deeply rooted in Afro-Brazilian culture, associated with Rio Carnival."},
    "sertanejo":  {"cultural_region": "Brazil", "language": "Portuguese", "global_reach": "low", "cultural_notes": "Brazilian country music rooted in the rural interior regions."},
    "soul":       {"cultural_region": "United States", "language": "English", "global_reach": "high", "cultural_notes": "Emerged from African American gospel and R&B traditions in the 1950s."},
    "spanish":    {"cultural_region": "Spain", "language": "Spanish", "global_reach": "high", "cultural_notes": "Spans flamenco, pop, and rock from Spain with strong Latin crossover."},
}

def enrich_genre(genre):
    # Use hardcoded label if available
    if genre in HARDCODED_LABELS:
        return HARDCODED_LABELS[genre]

    # Fall back to Ollama for any unknown genres
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
            json={"model": "llama3.2:latest", "prompt": prompt, "stream": False},
            timeout=30
        )
        result = response.json()["response"].strip()
        start = result.find("{")
        end = result.rfind("}") + 1
        if start != -1 and end != 0:
            return json.loads(result[start:end])
    except Exception as e:
        print(f"  Error enriching '{genre}': {e}")
    return None

# Enrich each genre
print("Running enrichment...\n")
genre_labels = {}

for i, genre in enumerate(unique_genres):
    print(f"[{i+1}/{len(unique_genres)}] Enriching: {genre}")
    label = enrich_genre(genre)
    if label:
        genre_labels[genre] = label
        print(f"  -> {label['cultural_region']} | {label['language']} | {label['global_reach']}")
    else:
        print(f"  -> MISSING")
    time.sleep(0.2)

# Map back to dataset
df['ollama_cultural_region'] = df['cultural_genre'].map(
    lambda g: genre_labels.get(g, {}).get('cultural_region', ''))
df['ollama_language'] = df['cultural_genre'].map(
    lambda g: genre_labels.get(g, {}).get('language', ''))
df['ollama_global_reach'] = df['cultural_genre'].map(
    lambda g: genre_labels.get(g, {}).get('global_reach', ''))
df['ollama_cultural_notes'] = df['cultural_genre'].map(
    lambda g: genre_labels.get(g, {}).get('cultural_notes', ''))

df.to_csv('data/full_data.csv', index=False)
print(f"\nDone! Saved to data/full_data.csv")
print(f"Columns: {df.columns.tolist()}")