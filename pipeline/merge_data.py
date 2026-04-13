import pandas as pd

our_data = pd.read_csv('data/starter_data.csv')
kaggle = pd.read_csv('data/kaggle_audio_features.csv')

# Keep only the audio feature columns from Kaggle
audio_cols = [
    'track_id', 'duration_ms', 'danceability', 'energy', 'key',
    'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'time_signature', 'track_genre', 'explicit'
]
kaggle_slim = kaggle[audio_cols].drop_duplicates(subset='track_id')

# Drop the empty audio feature columns from our data
our_data_slim = our_data.drop(columns=[
    'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature'
])

# Merge on track_id
merged = our_data_slim.merge(kaggle_slim, on='track_id', how='left')

print(f'Our tracks: {len(our_data_slim)}')
print(f'Matched with audio features: {merged["danceability"].notna().sum()}')
print(f'Unmatched: {merged["danceability"].isna().sum()}')
print(f'\nFinal shape: {merged.shape}')
print(f'\nColumns: {merged.columns.tolist()}')

merged.to_csv('data/starter_data_enriched.csv', index=False)
print('\nSaved to data/starter_data_enriched.csv')