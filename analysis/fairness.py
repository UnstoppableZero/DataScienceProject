import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data and test set
df = pd.read_csv('data/full_data.csv')
test_set = pd.read_csv('outputs/models/test_set.csv')

# Load best model
xgb = joblib.load('outputs/models/xgboost.pkl')

# Output folder
os.makedirs('outputs/fairness', exist_ok=True)

FEATURES = ['danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

X_test = test_set[FEATURES]
y_test = test_set['top10']
regions = test_set['region']

# Get predictions
y_pred = xgb.predict(X_test)
y_proba = xgb.predict_proba(X_test)[:, 1]  # type: ignore

print("=== Test Set Summary ===")
print(f"Total test samples: {len(X_test)}")
print(f"Regions in test set: {regions.nunique()}")
print(f"\nRegion distribution in test set:")
print(regions.value_counts())

# --- Fairness Metrics ---
metrics = {
    "accuracy": accuracy_score,
    "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0),
    "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=0),
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0),
}

mf = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=regions
)

print("\n=== MetricFrame by Region ===")
print(mf.by_group.to_string())
print(f"\n=== Overall Metrics ===")
print(mf.overall)

dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=regions)
eod = equalized_odds_difference(y_test, y_pred, sensitive_features=regions)

print(f"\n=== Demographic Parity Difference ===")
print(f"  {dpd:.4f}  (0 = perfectly fair, higher = more biased)")
print(f"\n=== Equalized Odds Difference ===")
print(f"  {eod:.4f}  (0 = perfectly fair, higher = more biased)")

mf.by_group.to_csv('outputs/fairness/metrics_by_region.csv')
print("\nMetrics saved.")

# Convert to percentages for charts
metrics_pct = mf.by_group.copy() * 100

# --- Chart 1: Recall by Region (%) ---
recall_by_region = metrics_pct['recall'].reset_index()
recall_by_region.columns = ['region', 'recall_pct']
recall_by_region = recall_by_region.sort_values('recall_pct', ascending=False)
overall_recall_pct = mf.overall['recall'] * 100

plt.figure(figsize=(12, 7))
bars = sns.barplot(data=recall_by_region, x='region', y='recall_pct',
                   hue='region', legend=False, palette='RdYlGn')
plt.axhline(y=overall_recall_pct, color='navy', linestyle='--',
            linewidth=2, label=f"Overall Recall ({overall_recall_pct:.1f}%)")
plt.title('How Often Does the Model Correctly Identify a Hit by Region?',
          fontsize=14, fontweight='bold')
plt.xlabel('Region', fontsize=12)
plt.ylabel('Recall — % of Actual Hits Correctly Predicted', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend()

for bar, val in zip(bars.patches, recall_by_region['recall_pct']):
    bars.annotate(f'{val:.1f}%',  # type: ignore
                  (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3),  # type: ignore
                  ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/fairness/chart1_recall_by_region.png', dpi=150)
plt.show()
print("Chart 1 saved.")

# --- Chart 2: F1 Score by Region (%) ---
f1_by_region = metrics_pct['f1'].reset_index()
f1_by_region.columns = ['region', 'f1_pct']
f1_by_region = f1_by_region.sort_values('f1_pct', ascending=False)
overall_f1_pct = mf.overall['f1'] * 100

plt.figure(figsize=(12, 7))
sns.barplot(data=f1_by_region, x='region', y='f1_pct',
            hue='region', legend=False, palette='coolwarm')
plt.axhline(y=overall_f1_pct, color='navy', linestyle='--',
            linewidth=2, label=f"Overall F1 ({overall_f1_pct:.1f}%)")
plt.title('Overall Model Performance by Region (F1 Score)',
          fontsize=14, fontweight='bold')
plt.xlabel('Region', fontsize=12)
plt.ylabel('F1 Score — Balances Precision & Recall (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/fairness/chart2_f1_by_region.png', dpi=150)
plt.show()
print("Chart 2 saved.")

# --- Chart 3: Fairness Summary ---
plt.figure(figsize=(9, 6))
metric_names = ['Demographic Parity\nDifference', 'Equalized Odds\nDifference']
scores = [dpd, eod]
plain_english = [
    f"Hit prediction rate varies\nby {dpd*100:.1f}% across regions",
    f"Model is {eod*100:.1f}% less accurate\nfor underrepresented regions"
]
colors = ['#d9534f', '#d9534f']

bars_summary = plt.bar(metric_names, scores, color=colors,
                       edgecolor='white', linewidth=1.5, width=0.4)
plt.axhline(y=0.1, color='green', linestyle='--', linewidth=1.5,
            label='Fair threshold (0.10)')
plt.title('Bias Measurement Summary\n(Closer to 0 = More Fair)',
          fontsize=14, fontweight='bold')
plt.ylabel('Bias Score (0 = Fair, 1 = Maximum Bias)', fontsize=11)
plt.ylim(0, 0.5)

for bar, val, note in zip(bars_summary, scores, plain_english):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,  # type: ignore
             f'{val:.4f}', ha='center', fontsize=13, fontweight='bold', color='#d9534f')
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,  # type: ignore
             note, ha='center', fontsize=9, color='white', fontweight='bold')

plt.legend()
plt.tight_layout()
plt.savefig('outputs/fairness/chart3_fairness_summary.png', dpi=150)
plt.show()
print("Chart 3 saved.")

# --- Chart 4: Heatmap (%) ---
plt.figure(figsize=(10, 8))
sns.heatmap(metrics_pct, annot=True, fmt='.1f', cmap='RdYlGn',
            linewidths=0.5, vmin=0, vmax=100,
            annot_kws={"size": 10})
plt.title('All Fairness Metrics by Region (%)', fontsize=14, fontweight='bold')
plt.xlabel('Metric', fontsize=12)
plt.ylabel('Region', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/fairness/chart4_metrics_heatmap.png', dpi=150)
plt.show()
print("Chart 4 saved.")

print("\n=== Phase 4 Complete — Fairness Audit Done ===")