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

### 🛡️ Pre-Release Leakage Audit & Feature Decisions (Step 2D)
From the standpoint of an executive pitching a drama before filming, all attributes were audited in `notebooks/03_leakage_review.ipynb` and categorized:

- **Valid Pre-Release Candidate Inputs:**
  - **Production Parameters:** `country`, `original_network`, `episodes` (contracted episode order; *caveat:* MDL records actual aired episodes, so cancelled/shortened shows may have fewer—the Copilot interface will accept the user's planned order), `aired_on` (scheduled day/slot), `content_rating`, `is_short_run` (`episodes <= 4`, pending scope decision in Step 2E).
  - **Creative Pitch:** `genres` (pitch genres), `synopsis` (concept logline/premise; sampled across 30 random titles showing 0 spoilers/recaps and 0 "Source:" lines, with weak +0.1492 correlation to watchers).
  - **Attached Talent:** `director`, `screenwriter`, `cast_summary` (talent features must strictly use `[Main Role]` lead actors only).

- **Display / Identifier Only (Excluded from Inputs):**
  - `name`: Working project title; ends with `"(YYYY)"` which leaks `release_year`. Mark as display/identifier only; any NLP/text use must explicitly strip the `"(YYYY)"` suffix.
  - `native_title`: Non-English script title; redundant with `name`.

- **Strictly Excluded Post-Release Leakage:**
  - **Target Variables:** `rating`, `watchers`, `popularity`, `is_unranked`.
  - **Broadcast & Lifecycle Leakage:** `end_date` (leaks cancellations, run length, or hiatuses), `aired_bw`, `also_known_as`.
  - **Crowdsourced User Tags (`tags`):** Excluded from input features. Tags reveal mid-to-end plot spoilers (`Death`, `Murder`, `Infidelity`, `Tragic Ending`), and total tag count correlates (+0.3022) with `watchers` due to community engagement bias.
  - **Format / Audit Flags:** `is_special_or_pilot` (audit-only flag; superseded by `is_short_run`).
  - **Page Edit Missingness Flags:** `n_missing_credits`, `episodes_was_missing`, `synopsis_was_missing`, `tags_was_missing`, `genres_was_missing`.
  - **Evaluation Split Variable:** `release_year` is strictly reserved for time splitting.

- **Quantifying the "Unknown" Leakage (Page Completeness Bias):**
  Presence of `"Unknown"` or `"Not Yet Rated"` carries a strong post-release page-completeness signal:

  | Field | Value | Unknown Count | Median Watchers (Unk) | Median Rating (Unk) | Known Count | Median Watchers (Known) | Median Rating (Known) |
  | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
  | `director` | Unknown | 4,653 (44.7%) | 120 | 7.10 | 5,747 (55.3%) | 905 | 7.50 |
  | `screenwriter` | Unknown | 5,490 (52.8%) | 147 | 7.20 | 4,910 (47.2%) | 1,054 | 7.50 |
  | `original_network` | Unknown | 2,060 (19.8%) | 133 | 7.00 | 8,340 (80.2%) | 461 | 7.40 |
  | `aired_on` | Unknown | 2,233 (21.5%) | 67 | 7.10 | 8,167 (78.5%) | 562 | 7.40 |
  | `cast_summary` | Unknown | 637 (6.1%) | 25 | 6.80 | 9,763 (93.9%) | 386 | 7.40 |
  | `content_rating` | Not Yet Rated | 6,541 (62.9%) | 164 | 7.30 | 3,859 (37.1%) | 2,118 | 7.50 |

  *Implication & Modeling Plan:* Obscure dramas are left unedited by the community. These values carry the same page-completeness signal as the missingness flags we excluded. **Milestone 4 will explicitly benchmark models trained with "Unknown" kept versus masked/regularized.**

- **Temporal Prior Engineering Rules & Scraper Limitations:**
  - Talent track records (actor hit rates, director averages) must use an **expanding historical window** restricted to titles released **strictly before** the drama's premiere date ($\text{release\_date}_{\text{prior}} < \text{release\_date}_{\text{current}}$). Global dataset-wide aggregations are strictly prohibited.
  - *Data Limitation:* Because of scraper temporal gaps (China 2019–2020 and 2023+, Japan 2019–2020 and 2024+), talent track-record priors will be computed from an incomplete historical corpus and must be treated as proxy estimates.

### 🎯 Target Definition & Dataset Scope (Step 2E)
Formally established in `notebooks/04_target_and_scope.ipynb` and persisted to `data/processed/dramas_labeled.csv`:

- **Target Metric (Relative MDL Audience Reach):**
  - **Definition:** Percentile rank of `watchers` calculated within each `(country, release_year)` cohort (`target_pct`).
  - **Decision Tiers (`target_tier`):**
    - **High (Top 25%):** $\text{target\_pct} \ge 0.75$ *(Commercial hit relative to country-year peers)*
    - **Medium (Middle 50%):** $0.25 \le \text{target\_pct} < 0.75$ *(Moderate mainstream performance)*
    - **Low (Bottom 25%):** $\text{target\_pct} < 0.25$ *(Underperforming relative to country-year peers)*
  - *Construction Property:* The 25% High, 50% Medium, 25% Low class distribution holds **by construction** across all countries, years, and train/test splits.
  - *Core Limitation:* The target specifically measures **relative audience reach on MyDramaList**. It is not a proxy for domestic broadcast TV ratings, production profitability, or critical acclaim.

- **Dataset Scope (Minimum 6 Episodes):**
  - Applied **before** computing percentiles. Eliminates short-run web series, pilots, and specials ($< 6$ episodes), and drops 18 rows with missing episode counts (2,784 titles removed).

- **Small-Cohort Exclusion Rule (< 50 Dramas):**
  - Any cohort containing fewer than 50 dramas after the scope filter is excluded to prevent noisy ranking thresholds:
    - `China 2019`: 13 titles
    - `Japan 2018`: 47 titles
    - `South Korea 2020`: 21 titles (Jan–Feb only)
  - *Removed by small-cohort rule:* 81 titles.
  - **Final Scoped & Labeled Dataset:** **7,535 dramas** (Train $\le 2021$: 5,010; Test $\ge 2022$: 2,525).
  - All retained cohorts contain at least 118 dramas (zero cohorts between 50 and 100).
  - *Truncated Cohort Limitation:* Due to scraper cutoffs, `China 2022` (ends ~October) and `Japan 2023` (ends ~July) are truncated cohorts covering only earlier-airing titles in those years.

- **Evaluation Plan (Milestone 4):**
  - **Metrics:** Report overall accuracy, **macro-F1**, and **balanced accuracy** across the 3 tiers (High, Medium, Low), as well as a binary **"High vs. not High"** secondary view.
  - **Dual Test Evaluation:** Models will be evaluated on both:
    1. **Full Test Set** ($N = 2,525$).
    2. **`complete_info` Subset** ($N = 1,022$ test dramas where director, writer, network, and cast are all listed):
       - *Test breakdown by country:* South Korea: 336 (44.7%), Thailand: 344 (43.5%), Japan: 241 (53.3%), China: 101 (19.0%).
       - *(Overall dataset complete_info counts:* South Korea: 934 / 48.8%, Thailand: 816 / 42.7%, Japan: 759 / 60.5%, China: 535 / 21.8%; total $N = 3,044$).
  - **Country Breakdown:** All metrics will be reported overall and broken down per country.
  - **Baselines:**
    1. **Majority-Class Baseline:** Predict "Medium" for all titles (**50.0% accuracy**).
    2. **Metadata-Presence Baseline:** An ablation model trained using *only* the Unknown-signals (director, screenwriter, network, cast listed or not) to isolate how much predictive signal is driven by page-completeness bias alone.

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
- [x] **Milestone 2:** Exploratory Data Analysis (EDA), missing value strategy, and data leakage audit.
  - [x] **Steps 2A & 2B:** Data sanity checks, date parsing, and provenance audit (scraping gaps, ascending pagination analysis, and time-split design).
  - [x] **Step 2C:** Data cleaning, leakage-safe exclusions, pilot/special flagging, and credit count indicators (`notebooks/02_cleaning.ipynb`).
  - [x] **Step 2D:** Leakage review (column classification, pre/post-release analysis of tags, cast, crew, synopsis, and temporal track-record rules; `notebooks/03_leakage_review.ipynb`).
  - [x] **Step 2E:** Target variable definition, distribution analysis, and scope cutoff selection (`notebooks/04_target_and_scope.ipynb`).
- [ ] **Milestone 3:** Leakage-free feature engineering.
- [ ] **Milestone 4:** Time-based train/test splitting and baseline predictive modeling.
- [ ] **Milestone 5:** Model interpretability with SHAP.
- [ ] **Milestone 6:** RAG retriever for comparable dramas.
- [ ] **Milestone 7:** What-if simulator and LLM agent integration.
- [ ] **Milestone 8:** FastAPI, Dockerization, and Hugging Face Spaces deployment.

