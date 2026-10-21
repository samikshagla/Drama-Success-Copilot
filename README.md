# Drama Success Copilot 🎬

An end-to-end AI/GenAI decision support system designed to predict whether an Asian drama concept (Korean, Chinese, Japanese, Thai) is likely to succeed, explain the key drivers behind the prediction, and generate actionable greenlight recommendations for production teams.

---

## 📌 Problem Statement
Pitching and greenlighting television dramas is high-stakes and historically intuition-driven. Studio executives and screenwriters need evidence-backed intelligence to answer:
- **Will this drama concept succeed with global viewers?**
- **Which specific factors (genres, cast, episode length, airing schedule) drive or hurt the predicted outcome?**
- **What historical dramas are most similar to this idea, and how did they perform?**
- **How can we tweak elements (the "What-If" scenario) to optimize audience engagement?**

---

## 🏗️ System Architecture & Roadmap
The project is built iteratively across 6 core pillars:

1. **Success Predictor (Machine Learning):** Tabular classification/regression (scikit-learn / XGBoost) to estimate audience reception while preventing data leakage.
2. **Explainability Engine (SHAP):** Local and global feature attribution using Shapley values to explain *why* a drama scored high or low.
3. **Similar-Drama Retriever (RAG):** Text embedding and vector search to find historical comparable titles and their performance patterns.
4. **What-If Scenario Simulator:** Interactive sensitivity tool allowing creators to modify inputs (e.g., change genre, swap cast tier) and compare predicted outcomes in real time.
5. **Grounded LLM Agent:** An autonomous decision agent that synthesizes model predictions, SHAP explanations, and retrieved drama precedents into a structured greenlight memorandum.
6. **Full-Stack Production Deployment:** FastAPI backend, containerized with Docker, paired with a web UI and deployed on Hugging Face Spaces.

---

## 📁 Repository Structure
```text
asian_dramas/
├── data/
│   ├── raw/                 # Scraped raw datasets (full data ignored by git)
│   │   └── sample_50_dramas.csv  # 50-row sample tracked for reproducibility
│   └── processed/           # Cleaned and feature-engineered datasets
├── notebooks/               # Jupyter notebooks for EDA and experiments
├── src/                     # Production Python modules (cleaning, modeling, RAG)
├── app/                     # FastAPI backend and frontend UI
├── mdl_scraper/             # Scrapy spiders and scrapers used for data collection
├── .gitignore               # Strict git hygiene (data, secrets, models, venvs)
└── README.md                # Project documentation and milestone tracker
```

---

## 📊 Dataset Description
- **Source:** Scraped from [MyDramaList (MDL)](https://mydramalist.com/).
- **Coverage:** Korean, Chinese, Japanese, and Thai dramas.
- **Scale:** 10,400 drama titles with 20 raw attributes capturing metadata, audience metrics, content tags, genres, and cast composition.
- **Storage Policy:** Large raw files (`>15MB`) and scraper state files are safely excluded from version control via `.gitignore`. A 50-row sample (`data/raw/sample_50_dramas.csv`) is versioned for local smoke tests.

---

## 🚀 Project Milestones
- [x] **Milestone 1:** Repository architecture setup, git hygiene, environment verification, and raw data schema inspection.
- [ ] **Milestone 2:** Exploratory Data Analysis (EDA), missing value strategy, and data leakage audit.
- [ ] **Milestone 3:** Target variable definition and leakage-free feature engineering.
- [ ] **Milestone 4:** Time-based train/test splitting and baseline predictive modeling.
- [ ] **Milestone 5:** Model interpretability with SHAP.
- [ ] **Milestone 6:** RAG retriever for comparable dramas.
- [ ] **Milestone 7:** What-if simulator and LLM agent integration.
- [ ] **Milestone 8:** FastAPI, Dockerization, and Hugging Face Spaces deployment.
