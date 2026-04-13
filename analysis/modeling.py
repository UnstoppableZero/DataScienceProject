import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('data/full_data.csv')

# Output folders
os.makedirs('outputs/models', exist_ok=True)

# Features and target
FEATURES = ['danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

X = df[FEATURES]
y = df['top10']

# Train/test split — stratified to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=== Data Split ===")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Top 10 in test set: {y_test.sum()}")
print(f"Non-top 10 in test set: {(y_test == 0).sum()}")

# --- Model 1: Logistic Regression ---
print("\n=== Logistic Regression ===")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_proba = lr.predict_proba(X_test)[:, 1]

print(classification_report(y_test, lr_preds))
print(f"ROC-AUC: {roc_auc_score(y_test, lr_proba):.4f}")

# --- Model 2: Random Forest ---
print("\n=== Random Forest ===")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, rf_preds))
print(f"ROC-AUC: {roc_auc_score(y_test, rf_proba):.4f}")

# --- Model 3: XGBoost ---
print("\n=== XGBoost ===")
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
xgb = XGBClassifier(n_estimators=100, random_state=42,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric='logloss', verbosity=0)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
xgb_proba = xgb.predict_proba(X_test)[:, 1]

print(classification_report(y_test, xgb_preds))
print(f"ROC-AUC: {roc_auc_score(y_test, xgb_proba):.4f}")

# Save models and test set for fairness audit
joblib.dump(rf, 'outputs/models/random_forest.pkl')
joblib.dump(xgb, 'outputs/models/xgboost.pkl')
joblib.dump(lr, 'outputs/models/logistic_regression.pkl')

X_test_saved = X_test.copy()
X_test_saved['top10'] = y_test.values
X_test_saved['region'] = df.loc[X_test.index, 'region'].values  # type: ignore
X_test_saved.to_csv('outputs/models/test_set.csv', index=False)

print("\nModels and test set saved to outputs/models/")

# --- Confusion Matrices ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, preds, title in zip(axes,
    [lr_preds, rf_preds, xgb_preds],
    ['Logistic Regression', 'Random Forest', 'XGBoost']):
    cm = confusion_matrix(y_test, preds)
    ConfusionMatrixDisplay(cm, display_labels=['Non-Top 10', 'Top 10']).plot(ax=ax, colorbar=False)
    ax.set_title(title, fontsize=13, fontweight='bold')

plt.suptitle('Confusion Matrices — All Models', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/models/confusion_matrices.png', dpi=150)
plt.show()
print("Confusion matrices saved.")

# --- Feature Importance — Random Forest ---
rf_importance = pd.DataFrame({
    'feature': FEATURES,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=rf_importance, x='importance', y='feature', hue='feature', legend=False, palette='Blues_d')
plt.title('Feature Importance — Random Forest', fontsize=16, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/models/feature_importance_rf.png', dpi=150)
plt.show()
print("Random Forest feature importance saved.")

# --- Feature Importance — XGBoost ---
xgb_importance = pd.DataFrame({
    'feature': FEATURES,
    'importance': xgb.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=xgb_importance, x='importance', y='feature', hue='feature', legend=False, palette='Oranges_d')
plt.title('Feature Importance — XGBoost', fontsize=16, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/models/feature_importance_xgb.png', dpi=150)
plt.show()
print("XGBoost feature importance saved.")

# --- Model Comparison Summary ---
summary = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'ROC-AUC': [roc_auc_score(y_test, lr_proba),
                roc_auc_score(y_test, rf_proba),
                roc_auc_score(y_test, xgb_proba)],
    'Top10 Recall': [
        dict(classification_report(y_test, lr_preds, output_dict=True))['1']['recall'],  # type: ignore[index]
        dict(classification_report(y_test, rf_preds, output_dict=True))['1']['recall'],  # type: ignore[index]
        dict(classification_report(y_test, xgb_preds, output_dict=True))['1']['recall'],  # type: ignore[index]
    ]
})
print("\n=== Model Comparison Summary ===")
print(summary.to_string(index=False))
summary.to_csv('outputs/models/model_comparison.csv', index=False)

print("\n=== Phase 3 Complete ===")