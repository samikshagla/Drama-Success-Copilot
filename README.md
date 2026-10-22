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
Drama-Success-Copilot/
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

### ⚠️ Data Limitations & Coverage Gaps (Provenance Audit)
An audit of the scraper code (`mdl_scraper/`) revealed specific temporal coverage gaps that dictate our modeling decisions:
- **Ascending Date Pagination:** MyDramaList's `so=date` search parameter sorts **chronologically ascending (oldest-to-newest)**. Each country was scraped with a hard cap of 65 pages (1,300 titles) across two runs (`2015–2026` and `2021–2026`).
- **High-Volume Country Truncation & Partial Years:**
  - **China:** Covers 2015–2018, 14 dramas in 2019, and 2021–2022. It stops in **October 2022** (0 dramas in 2020 or 2023–2026).
  - **Japan:** Covers 2015–2017 and 2021–mid 2023. It stops in **July 2023** (0 dramas in 2019–2020 or 2024–2026).
  - **South Korea:** Spans 2015–mid 2025, but 2020 has only 31 titles (**January–February 2020 only**) due to the crawl cutoff.
  - **Thailand:** Broadest and most consistent coverage, spanning 2015–mid 2025 (~200–290 titles/year).
- **Partial Year Warning:** Because coverage is cut off midway through certain years (China 2022 ends ~Oct, Japan 2023 ends ~Jul, Korea 2020 covers only Jan–Feb), **release month and airing season must NOT be used as predictive features without verifying bias**.
- **Year Consistency:** For all 10,238 titles where both `aired_bw` and a title year `(YYYY)` are present, the two years match in 100% of cases.
- **Page Edit Bias Warning:** Missing-metadata indicators (such as credit counts) may reflect post-release crowdsourced edits (popular dramas attract more volunteer contributors on MDL). We will benchmark models with and without these missingness features.
- **RAG Indexing Note:** Titles with empty `synopsis` will be excluded from the semantic vector search / RAG comp retrieval index.

### 🎯 Evaluation & Time-Split Strategy (Decision Log)
- **Validation Split Scheme:** **Option 1 (Train: $\le$ 2021, Test: $\ge$ 2022)**.
  - **Train Set:** 6,970 dramas (67.0%) — China 29.0%, Japan 25.6%, South Korea 23.4%, Thailand 22.0%.
  - **Test Set:** 3,430 dramas (33.0%) — Thailand 31.2%, South Korea 28.3%, Japan 23.8%, China 16.8%.
  - Ensures all 4 countries are represented in both training and testing, mitigating the risk of `country` acting as an accidental proxy for time.
- **Reporting Requirement:** Evaluation metrics must be reported **both overall and broken down per country** to detect country-specific performance disparities.
- **Input Feature Restriction:** `release_year` is strictly reserved for temporal train/test splitting and will **NOT** be fed into the model as an input feature.

---

## ⚙️ Setup & Installation

### 1. Environment Setup
Clone the repository and set up a Python 3.11 virtual environment:
```bash
# Clone the repository
git clone https://github.com/samikshagla/Drama-Success-Copilot.git
cd Drama-Success-Copilot

# Create virtual environment with Python 3.11
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# Install core data science dependencies
python -m pip install -r requirements.txt
```

*(Note: Scraper dependencies are archived separately in `mdl_scraper/requirements-scraper.txt`.)*

### 2. Jupyter Kernel Registration
Register the virtual environment as an ipykernel:
```bash
python -m ipykernel install --user --name drama-success-copilot --display-name "Python (Drama Success Copilot)"
```
In Jupyter Notebook or VS Code, select **`Python (Drama Success Copilot)`** from the kernel picker at the top right.

### 3. Data File Placement
The full scraped dataset (`dramas_raw.csv`, ~15.35 MB) is excluded from version control to maintain clean repository hygiene.
Place the raw data file directly in:
```
data/raw/dramas_raw.csv
```
*(A 50-row smoke-test sample `data/raw/sample_50_dramas.csv` is tracked in the repository for quick testing and pipeline verification.)*

---

## 🚀 Project Milestones
- [x] **Milestone 1:** Repository architecture setup, git hygiene, environment verification, and raw data schema inspection.
- [ ] **Milestone 2:** Exploratory Data Analysis (EDA), missing value strategy, and data leakage audit.
  - [x] **Steps 2A & 2B:** Data sanity checks, date parsing, and provenance audit (scraping gaps, ascending pagination analysis, and time-split design).
  - [x] **Step 2C:** Data cleaning, leakage-safe exclusions, pilot/special flagging, and credit count indicators (`notebooks/02_cleaning.ipynb`).
  - [ ] **Step 2D:** Leakage review (column classification, pre/post-release analysis of tags, cast, crew, synopsis, and temporal track-record rules).
  - [ ] **Step 2E:** Target variable definition, distribution analysis, and scope cutoff selection.
- [ ] **Milestone 3:** Target variable definition and leakage-free feature engineering.
- [ ] **Milestone 4:** Time-based train/test splitting and baseline predictive modeling.
- [ ] **Milestone 5:** Model interpretability with SHAP.
- [ ] **Milestone 6:** RAG retriever for comparable dramas.
- [ ] **Milestone 7:** What-if simulator and LLM agent integration.
- [ ] **Milestone 8:** FastAPI, Dockerization, and Hugging Face Spaces deployment.

