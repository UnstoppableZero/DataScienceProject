import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# Page config
st.set_page_config(
    page_title="BlindSpot — Music Bias Audit",
    page_icon="🎵",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('data/full_data.csv')

df = load_data()

# Sidebar navigation
st.sidebar.title("🎵 BlindSpot")
st.sidebar.markdown("*Does Spotify favor certain sounds?*")
page = st.sidebar.radio("Navigate", [
    "Overview",
    "Regional Representation",
    "Model Performance",
    "Fairness Audit"
])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** {len(df):,} tracks")
st.sidebar.markdown(f"**Regions:** {df['region'].nunique()}")
st.sidebar.markdown(f"**Genres:** {df['cultural_genre'].nunique()}")

# --- PAGE: Overview ---
if page == "Overview":
    st.title("🎵 BlindSpot")
    st.subheader("Does Spotify's chart system favor certain cultural sounds over others?")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tracks", f"{len(df):,}")
    col2.metric("Cultural Regions", df['region'].nunique())
    col3.metric("Genres Analyzed", df['cultural_genre'].nunique())
    col4.metric("Top 10 Hits", f"{df['top10'].sum():,}")

    st.markdown("---")
    st.markdown("""
    ### The Problem
    Spotify controls what billions of people hear. With over 600 million users across 180 countries,
    the platform's algorithms shape what music gets discovered — and what gets overlooked.

    **BlindSpot** audits Spotify's ecosystem using real data to answer a simple but important question:
    are certain cultural sounds, languages, or regions systematically disadvantaged in what reaches the top?

    ### How We Built It
    - **Data Collection** — Spotify Web API across 12 cultural regions
    - **Audio Features** — 9 measurable sonic characteristics per track
    - **AI Enrichment** — Local Ollama LLM labels cultural origin, language, and global reach
    - **ML Modeling** — Logistic Regression, Random Forest, and XGBoost trained to predict chart success
    - **Fairness Audit** — Microsoft Fairlearn measures demographic parity and equalized odds across regions

    ### Key Finding
    > Tracks from Nigeria, Brazil, Colombia, and India have **0% recall** — the model never correctly
    identifies their hits. Meanwhile Korea and Mexico reach 36% and 27% respectively.
    The **Equalized Odds Difference is 37.5%** — well above the fair threshold of 10%.
    """)

# --- PAGE: Regional Representation ---
elif page == "Regional Representation":
    st.title("📊 Regional Representation")
    st.markdown("How are different cultural regions represented in our dataset and in chart success?")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Track Count by Region")
        img = Image.open('outputs/eda/chart1_region_distribution.png')
        st.image(img, use_container_width=True)

    with col2:
        st.subheader("Top 10 Hit Rate by Region")
        img = Image.open('outputs/eda/chart2_hit_rate_by_region.png')
        st.image(img, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Audio Features by Region")
        img = Image.open('outputs/eda/chart3_audio_features_by_region.png')
        st.image(img, use_container_width=True)

    with col4:
        st.subheader("Popularity Distribution by Region")
        img = Image.open('outputs/eda/chart5_popularity_distribution.png')
        st.image(img, use_container_width=True)

    st.markdown("---")
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Global Reach Distribution")
        img = Image.open('outputs/eda/chart7_global_reach.png')
        st.image(img, use_container_width=True)

    with col6:
        st.subheader("Language Diversity Across Regions")
        img = Image.open('outputs/eda/chart8_language_diversity.png')
        st.image(img, use_container_width=True)

    st.markdown("---")
    st.subheader("Explore the Data")
    region_filter = st.selectbox("Filter by Region", ["All"] + sorted(df['region'].unique().tolist()))
    if region_filter != "All":
        filtered = df[df['region'] == region_filter]
    else:
        filtered = df

    display_cols = [c for c in ['track_name', 'artist_name', 'region', 'cultural_genre',
                                 'popularity', 'danceability', 'energy', 'valence', 'top10']
                    if c in filtered.columns]
    st.dataframe(filtered[display_cols].head(50), use_container_width=True)

# --- PAGE: Model Performance ---
elif page == "Model Performance":
    st.title("🤖 Model Performance")
    st.markdown("Three machine learning models trained to predict whether a track reaches the global Top 10.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Logistic Regression ROC-AUC", "0.7062")
    col2.metric("Random Forest ROC-AUC", "0.6934")
    col3.metric("XGBoost ROC-AUC", "0.6661")

    st.markdown("""
    > **Key Insight:** Audio features alone are weak predictors of chart success.
    > This suggests other factors — label backing, algorithmic promotion, and cultural origin —
    > play a larger role than the sound itself.
    """)

    st.markdown("---")
    col4, col5 = st.columns(2)

    with col4:
        st.subheader("Confusion Matrices")
        img = Image.open('outputs/models/confusion_matrices.png')
        st.image(img, use_container_width=True)

    with col5:
        st.subheader("Feature Importance — Random Forest")
        img = Image.open('outputs/models/feature_importance_rf.png')
        st.image(img, use_container_width=True)

    st.subheader("Feature Importance — XGBoost")
    img = Image.open('outputs/models/feature_importance_xgb.png')
    st.image(img, use_container_width=True)

# --- PAGE: Fairness Audit ---
elif page == "Fairness Audit":
    st.title("⚖️ Fairness Audit")
    st.markdown("Measuring whether the model treats all cultural regions equally.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    col1.metric("Demographic Parity Difference", "0.1874",
                delta="Above fair threshold (0.10)", delta_color="inverse")
    col2.metric("Equalized Odds Difference", "0.3750",
                delta="Above fair threshold (0.10)", delta_color="inverse")

    st.markdown("""
    > **What this means:** The model is **37.5% less likely** to correctly identify a hit
    > from underrepresented regions. Tracks from Nigeria, Brazil, Colombia, and India
    > have **0% recall** — the model never correctly predicts their hits.
    """)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Recall by Region")
        img = Image.open('outputs/fairness/chart1_recall_by_region.png')
        st.image(img, use_container_width=True)

    with col4:
        st.subheader("F1 Score by Region")
        img = Image.open('outputs/fairness/chart2_f1_by_region.png')
        st.image(img, use_container_width=True)

    st.markdown("---")
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Bias Measurement Summary")
        img = Image.open('outputs/fairness/chart3_fairness_summary.png')
        st.image(img, use_container_width=True)

    with col6:
        st.subheader("All Metrics by Region")
        img = Image.open('outputs/fairness/chart4_metrics_heatmap.png')
        st.image(img, use_container_width=True)

    st.markdown("---")
    st.subheader("Raw Fairness Metrics by Region")
    metrics_df = pd.read_csv('outputs/fairness/metrics_by_region.csv')
    metrics_df.columns = ['Region', 'Accuracy', 'Precision', 'Recall', 'F1']
    metrics_df = metrics_df.set_index('Region')
    metrics_df = (metrics_df * 100).round(1)
    st.dataframe(metrics_df, use_container_width=True)