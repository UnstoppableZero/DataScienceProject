# BlindSpot: Music Bias Audit 🎵🔍
**Does Streaming Favor Certain Sounds? An Audit of Audio Features and Cultural Representation.**

## 📖 Project Overview
Streaming platforms like Spotify use recommendation algorithms that influence which songs reach global audiences. While audio features are public, it is unclear if tracks from certain cultural or geographic origins are systematically over- or underrepresented on the charts. 

This project aims to predict chart success purely from audio data and apply fairness audits to identify patterns of underrepresentation. It combines predictive machine learning with responsible AI principles, directly addressing questions of cultural equity in algorithmic systems.

**Track:** Tech & Transformation — AI & Emerging Technologies  
**Institution:** Delaware State University (Spring 2026)

---

## 👥 The Team
* **Michael Danquah-Tabbi**
* **Dahmir Mason**
* **Owen Matimu**
* **Zion Reams**

*Advisor: Dr. Fatima Boukari*

---

## 🛠️ Tech Stack & Tooling
* **Data Extraction:** Spotify Web API (audio features), MusicBrainz / Last.fm APIs (geographic metadata)
* **AI Processing:** Local AI (**Ollama**) for cleaning and clustering unstructured, user-generated genre tags into standardized cultural categories
* **Machine Learning:** Logistic Regression, Random Forest, and XGBoost
* **Fairness Auditing:** Microsoft **Fairlearn**
* **Evaluation Metrics:** Precision, Recall, F1-Score, ROC-AUC, Demographic Parity, Equal Opportunity

---

## 🔬 Methodology
1. **Data Pipeline:** Aggregating 50,000+ tracks (2017–2024) and extracting continuous audio features alongside geographic metadata.
2. **LLM Processing:** Utilizing local AI to standardize cultural representation data.
3. **Modeling:** Training classifiers evaluated via 5-fold cross-validation.
4. **Bias Audit:** Leveraging Fairlearn to calculate disparity and evaluate if tracks from regions like Latin America, Africa, and Asia face higher barriers to global chart entry compared to Western tracks with identical audio scores.

---

## 📊 Major Takeaway
**Streaming algorithms may implicitly favor Western sonic profiles, restricting diverse cultural sounds to regional success despite having competitive audio features.** ---

## ⚖️ Discussion & Ethics
We acknowledge that proxy variables like "genre tags" are imperfect representations of cultural identity. This framework aims to highlight systemic architectural patterns rather than assign blame.
