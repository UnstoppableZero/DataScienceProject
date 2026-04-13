import pandas as pd

kaggle = pd.read_csv('data/kaggle_audio_features.csv')

# Map Kaggle genres to cultural regions
region_map = {
    "hip-hop": "US", "pop": "US", "r-n-b": "US", "soul": "US",
    "country": "US", "blues": "US", "jazz": "US",
    "british": "GB", "garage": "GB",
    "afrobeat": "Nigeria", "afropop": "Nigeria",
    "highlife": "Ghana", "afro-funk": "Ghana",
    "amapiano": "SouthAfrica", "kwaito": "SouthAfrica", "south-african": "SouthAfrica",
    "samba": "Brazil", "bossa-nova": "Brazil", "pagode": "Brazil", "sertanejo": "Brazil",
    "reggaeton": "Mexico", "banda": "Mexico", "corrido": "Mexico",
    "cumbia": "Colombia", "salsa": "Colombia",
    "k-pop": "Korea", "korean-pop": "Korea",
    "j-pop": "Japan", "j-rock": "Japan", "anime": "Japan",
    "bollywood": "India", "indian": "India",
    "french": "France", "chanson": "France",
    "german": "Germany", "schlager": "Germany",
    "flamenco": "Spain", "spanish": "Spain", "latin": "Spain",
}

# Assign region based on genre
kaggle['region'] = kaggle['track_genre'].map(region_map)

# Keep only tracks we could map to a region
mapped = kaggle[kaggle['region'].notna()].copy()

# Drop unnecessary columns
mapped = mapped.drop(columns=['Unnamed: 0', 'artists'])
mapped = mapped.rename(columns={'track_genre': 'cultural_genre'})

# Remove duplicates
mapped = mapped.drop_duplicates(subset='track_id')

# Create target variable: top10 = 1 if popularity >= 70
mapped['top10'] = (mapped['popularity'] >= 70).astype(int)

print(f'Total mapped tracks: {len(mapped)}')
print(f'\nRegion distribution:')
print(mapped['region'].value_counts())
print(f'\nTop 10 distribution:')
print(mapped['top10'].value_counts())
print(f'\nColumns: {mapped.columns.tolist()}')

mapped.to_csv('data/starter_data_enriched.csv', index=False)
print('\nSaved to data/starter_data_enriched.csv')