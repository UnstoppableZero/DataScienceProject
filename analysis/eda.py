import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
df = pd.read_csv('data/full_data.csv')

# Output folder
os.makedirs('outputs/eda', exist_ok=True)

# Visual style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.family'] = 'sans-serif'

# Quick summary
print("=== Dataset Summary ===")
print(f"Total tracks: {len(df)}")
print(f"Regions: {df['region'].nunique()}")
print(f"Genres: {df['cultural_genre'].nunique()}")
print(f"Top 10 tracks: {df['top10'].sum()}")
print(f"Non-top 10: {(df['top10'] == 0).sum()}")
print(f"\nRegion distribution:\n{df['region'].value_counts()}")

# --- Chart 1: Track Count by Region ---
region_counts = df['region'].value_counts().reset_index()
region_counts.columns = ['region', 'count']

plt.figure(figsize=(12, 6))
bars = sns.barplot(data=region_counts, x='region', y='count', hue='region', legend=False, palette='Blues_d')
plt.title('Track Representation by Region', fontsize=16, fontweight='bold')
plt.xlabel('Region', fontsize=12)
plt.ylabel('Number of Tracks', fontsize=12)
plt.xticks(rotation=45, ha='right')

for bar, val in zip(bars.patches, region_counts['count']):
    bars.annotate(f'{val}', (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                  ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('outputs/eda/chart1_region_distribution.png', dpi=150)
plt.show()
print("Chart 1 saved.")

# --- Chart 2: Top 10 Hit Rate by Region ---
hit_rate = df.groupby('region')['top10'].mean().reset_index()
hit_rate.columns = ['region', 'hit_rate']
hit_rate['hit_rate_pct'] = hit_rate['hit_rate'] * 100
hit_rate = hit_rate.sort_values('hit_rate_pct', ascending=False)

plt.figure(figsize=(12, 6))
bars = sns.barplot(data=hit_rate, x='region', y='hit_rate_pct', hue='region', legend=False, palette='RdYlGn')
plt.title('Top 10 Hit Rate by Region (%)', fontsize=16, fontweight='bold')
plt.xlabel('Region', fontsize=12)
plt.ylabel('Hit Rate (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')

for bar, val in zip(bars.patches, hit_rate['hit_rate_pct']):
    bars.annotate(f'{val:.1f}%', (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                  ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('outputs/eda/chart2_hit_rate_by_region.png', dpi=150)
plt.show()
print("Chart 2 saved.")

# --- Chart 3: Audio Feature Comparison by Region ---
audio_features = ['danceability', 'energy', 'valence', 'acousticness', 'speechiness']

region_features = df.groupby('region')[audio_features].mean().reset_index()
region_features_melted = region_features.melt(id_vars='region', var_name='feature', value_name='mean_value')

plt.figure(figsize=(14, 7))
sns.barplot(data=region_features_melted, x='region', y='mean_value', hue='feature', palette='tab10')
plt.title('Average Audio Features by Region', fontsize=16, fontweight='bold')
plt.xlabel('Region', fontsize=12)
plt.ylabel('Mean Value (0-1)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('outputs/eda/chart3_audio_features_by_region.png', dpi=150)
plt.show()
print("Chart 3 saved.")

# --- Chart 4: Correlation Heatmap ---
corr_cols = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness',
             'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
corr_matrix = df[corr_cols].corr(numeric_only=True)

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5)
plt.title('Correlation Heatmap — Audio Features vs Popularity', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/eda/chart4_correlation_heatmap.png', dpi=150)
plt.show()
print("Chart 4 saved.")

# --- Chart 5: Popularity Distribution by Region ---
plt.figure(figsize=(14, 7))
sns.boxplot(data=df, x='region', y='popularity', hue='region', legend=False, palette='Set2',
            order=df.groupby('region')['popularity'].median().sort_values(ascending=False).index)
plt.title('Popularity Distribution by Region', fontsize=16, fontweight='bold')
plt.xlabel('Region', fontsize=12)
plt.ylabel('Popularity Score (0-100)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('outputs/eda/chart5_popularity_distribution.png', dpi=150)
plt.show()
print("Chart 5 saved.")

# --- Chart 6: Top 10 vs Non-Top 10 Audio Feature Comparison ---
top10_features = df.groupby('top10')[audio_features].mean().reset_index()
top10_features['label'] = top10_features['top10'].map({0: 'Non-Top 10', 1: 'Top 10'})
top10_melted = top10_features.melt(id_vars='label', value_vars=audio_features,
                                    var_name='feature', value_name='mean_value')

plt.figure(figsize=(12, 6))
sns.barplot(data=top10_melted, x='feature', y='mean_value', hue='label', palette=['#d9534f', '#5cb85c'])
plt.title('Audio Features — Top 10 vs Non-Top 10 Tracks', fontsize=16, fontweight='bold')
plt.xlabel('Audio Feature', fontsize=12)
plt.ylabel('Mean Value (0-1)', fontsize=12)
plt.legend(title='Category')
plt.tight_layout()
plt.savefig('outputs/eda/chart6_top10_vs_nontop10.png', dpi=150)
plt.show()
print("Chart 6 saved.")

# --- Chart 7: Global Reach Distribution (Ollama Labels) ---
reach_counts = df['ollama_global_reach'].value_counts().reset_index()
reach_counts.columns = ['global_reach', 'count']

plt.figure(figsize=(8, 6))
colors = ['#5cb85c', '#f0ad4e', '#d9534f']
plt.pie(reach_counts['count'].tolist(), labels=reach_counts['global_reach'].tolist(),
        autopct='%1.1f%%', colors=colors, startangle=140,
        textprops={'fontsize': 13})
plt.title('Global Reach Distribution (Ollama Cultural Labels)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/eda/chart7_global_reach.png', dpi=150)
plt.show()
print("Chart 7 saved.")

# --- Chart 8: Language Diversity by Region ---
lang_region = df.groupby(['region', 'ollama_language']).size().reset_index(name='count')
top_langs = df['ollama_language'].value_counts().head(6).index.tolist()
lang_region_filtered = lang_region[lang_region['ollama_language'].isin(top_langs)]

plt.figure(figsize=(14, 7))
sns.barplot(data=lang_region_filtered, x='region', y='count', hue='ollama_language', palette='tab10')
plt.title('Language Distribution Across Regions', fontsize=16, fontweight='bold')
plt.xlabel('Region', fontsize=12)
plt.ylabel('Track Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Language', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('outputs/eda/chart8_language_diversity.png', dpi=150)
plt.show()
print("Chart 8 saved.")

print("\n=== EDA Complete -- All charts saved to outputs/eda/ ===")