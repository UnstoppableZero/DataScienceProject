# 🎧 BlindSpot: Music Bias Audit
**Does Streaming Favor Certain Sounds? An Audit of Audio Features and Cultural Representation.**

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20XGBoost-orange.svg)
![Fairness](https://img.shields.io/badge/Fairness-Microsoft%20Fairlearn-green.svg)
![API](https://img.shields.io/badge/API-Spotify%20%7C%20Last.fm-1DB954.svg)

## 📌 Executive Summary
Streaming platforms control music discovery for over 600 million global users. This project investigates whether streaming ecosystems systematically advantage certain musical sounds or cultural origins over others. 

By analyzing the audio features of top-charting tracks across 12 distinct global regions, we built predictive models for chart success and applied fairness auditing frameworks to identify patterns of underrepresentation. This project combines predictive machine learning with responsible AI principles to directly address cultural equity in algorithmic systems.

## 👥 The Team
* **Michael Danquah-Tabbi**
* **Dahmir Mason**
* **Owen Matimu**
* **Zien Reams**

**Institution:** Delaware State University  
**Track:** Tech & Transformation — AI & Emerging Technologies  
**Advisor:** Dr. Fatima Boukari  

---

## 🛠️ Technical Architecture & Tech Stack

### 1. Data Collection
* **Spotify Web API (Spotipy):** Extraction of continuous audio features (Tempo, Energy, Valence, Danceability, etc.) for 50,000+ tracks.
* **Last.fm / MusicBrainz API:** Scraping supplemental metadata for artist origin and raw genre tags.

### 2. AI Enrichment & Processing
* **Ollama (Local LLM):** Processing, cleaning, and clustering highly unstructured, user-generated genre tags into standardized cultural categories.
* **Pandas & NumPy:** Handling missing values, feature scaling, and interaction feature engineering (e.g., energy × danceability).

### 3. Machine Learning Models
* **Scikit-learn & XGBoost:** Training Logistic Regression, Random Forest, and XGBoost classifiers to predict global top 10 chart success based strictly on pure audio profiles.

### 4. Fairness & Bias Auditing
* **Microsoft Fairlearn:** Calculating disparity across regions using metrics such as Demographic Parity and Equal Opportunity / Equalized Odds to measure systemic bias thresholds.

---

## 📂 Repository Structure

```text
├── data/                   # Raw and processed datasets (Ignored in .gitignore)
├── notebooks/              # Jupyter notebooks for EDA and model prototyping
├── src/                    # Source code for the data pipeline
│   ├── data_collection.py  # Spotify & Last.fm API scripts
│   ├── llm_cleaning.py     # Ollama prompt chains for genre clustering
│   ├── models.py           # Model training and evaluation scripts
│   └── bias_audit.py       # Fairlearn implementation
├── presentation/           # Research Day posters and slide decks
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
