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
  - **Production Parameters:** `country`, `original_network` (*note: network is a strong proxy for country*), `episodes` (contracted episode order; *caveat:* MDL records actual aired episodes, so cancelled/shortened shows may have fewer—the Copilot interface will accept the user's planned order), `aired_on` (scheduled day/slot), `content_rating` (*note: "Not Yet Rated" is a placeholder for 62.6% of train dramas and is isolated in the missingness group; the 3 rare "R" ratings are merged into 18+*).
  - **Creative Pitch:** `genres` (pitch genres), `synopsis` (concept logline/premise; sampled across 30 random titles showing 0 spoilers/recaps and 0 "Source:" lines, with weak +0.1492 correlation to watchers).
  - **Attached Talent:** `director`, `screenwriter`, `cast_summary` (talent features must strictly use `[Main Role]` lead actors only).

- **Display / Identifier Only (Excluded from Inputs):**
  - `name`: Working project title; ends with `"(YYYY)"` which leaks `release_year`. Mark as display/identifier only; any NLP/text use must explicitly strip the `"(YYYY)"` suffix.
  - `native_title`: Non-English script title; redundant with `name`.

- **Strictly Excluded Post-Release Leakage:**
  - **Target Variables:** `rating`, `watchers`, `popularity`, `is_unranked`.
  - **Broadcast & Lifecycle Leakage:** `end_date` (leaks cancellations, run length, or hiatuses), `aired_bw`, `also_known_as`.
  - **Crowdsourced User Tags (`tags`):** Excluded from input features. Many tags describe plot developments (for example Death, Murder, Infidelity), and the number of tags per drama tracks popularity (r = +0.30), so tags are excluded as inputs.
  - **Format / Audit Flags:** `is_special_or_pilot` (audit-only flag; superseded by 6+ episode scope cutoff).
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

### 🛠️ Leakage-Free Feature Engineering (Milestone 3, Steps 1–4)
Implemented in `src/features.py` and validated in `notebooks/05_feature_engineering.ipynb`:

- **Step 1: Temporal Train/Test Split:**
  - Cutoff year: 2021. Train: $2015–2021$ ($N = 5,010$, 66.5%); Test: $2022–2025$ ($N = 2,525$, 33.5%).
  - Non-leakage guarantee: all transformations, vocabularies, groupings, and global shrinkage means are fitted strictly on Train and applied to Test.

- **Step 2: Basic Features (Core vs. Missingness):**
  - **Core Group ($N = 41$):** `episodes`, `country` (single categorical string; one-hot / categorical encoding deferred to train-only model preprocessor), `original_network` (57 train broadcasters with $\ge 10$ titles + `"Other"`; Unknown maps to `NaN`), `content_rating` (`15+`, `13+`, `G`, `18+`; rare "R" merged into "18+"; "Not Yet Rated" maps to `NaN`), 7 weekday indicators (`aired_on_monday` ... `sunday`; Unknown maps to `NaN`), and 30 multi-hot pitch genres (appearing in $\ge 30$ train titles; Unknown maps to `NaN`).
  - **Missingness Ablation Group ($N = 4$):** `aired_on_is_missing`, `content_rating_is_missing`, `network_is_missing`, `genre_is_missing`. Isolated from Core so that no combination of core columns encodes missingness except `NaN`.

- **Switchable Feature Groups in `src/features.py`:**
  - `core` ($N = 41$): Basic pitch-time features (episodes, country, original_network, content_rating, weekdays, multi-hot genres).
  - `talent_reach` ($N = 6$): Attached talent track-record reach (max/mean prior reach for director, screenwriter, lead cast).
  - `talent_experience` ($N = 8$): All-time and 5-year prior title counts ($n_{\text{prior}}$) for director, screenwriter, and cast. **Set to OFF by default** (ablation only) because raw experience features heavily track premiere year rather than talent quality.
  - `text` ($N = 1$): `synopsis_length`. **Set to OFF by default** (ablation only) because it is page-completeness-prone (synopsis missing ~15% in train; length correlated +0.15 with watchers, and +0.18 in train / +0.21 in test with `target_pct` on non-missing rows).
  - `missingness` ($N = 4$): Binary indicators for missing metadata (aired_on, content_rating, network, genre). **Set to OFF by default** (reserved for page-completeness bias ablation benchmarking).
  - Default active modeling features: `DEFAULT_FEATURE_GROUPS = ['core', 'talent_reach']` ($N = 47$).

- **Step 3: Attached Talent Track Records (Expanding-Window Priors):**
  - **Talent Features ($N = 14$):**
    - Director: `director_n_prior_max`, `director_n_prior_5yr_max`, `director_prior_reach_max`, `director_prior_reach_mean`.
    - Screenwriter: `screenwriter_n_prior_max`, `screenwriter_n_prior_5yr_max`, `screenwriter_prior_reach_max`, `screenwriter_prior_reach_mean`.
    - Lead Cast: `cast_n_prior_max`, `cast_n_prior_5yr_max`, `cast_n_prior_mean`, `cast_n_prior_5yr_mean`, `cast_prior_reach_max`, `cast_prior_reach_mean` (strictly using `[Main Role]` entries; dramas with no `[Main Role]` receive `NaN`).
  - **Conservative 12-Month Rule:** Only prior dramas satisfying $\text{eff\_pri\_date} \le \text{eff\_cur\_date} - 365\text{ days}$ qualify. If `start_date` is missing, Jan 1 of `release_year` is used for the current drama, and Dec 31 of `release_year` is used for candidate priors.
  - **Fixed 5-Year Lookback Variant:** The 5-year window ($\text{cutoff} - 5\text{ years} < \text{prior} \le \text{cutoff}$) only equalizes the history length from 2020 onward. In train, 2015–2019 are still left-censored (the 5-year and all-time counts are mathematically identical for those years because the corpus begins in 2015), so the $n_{\text{prior}}$ features still correlate with release year in train (about $+0.40$). Furthermore, 2019–2020 scraper gaps thin out the observable history for 2021 dramas.
  - **Unknown vs. Debut Distinction:** If a talent credit is `"Unknown"`, all its features are `NaN` (including $n_{\text{prior}}$). If talent is known but has 0 qualifying priors (debut), $n_{\text{prior}} = 0$ and prior reach = `NaN`.
  - **Empirical Bayes Smoothing:** $\hat{\mu} = \frac{n \cdot \bar{y} + m \cdot \mu_{\text{global}}}{n + m}$ with $\mu_{\text{global}} = 0.5022$ (Train mean). While $m = 1$ achieved slightly higher correlation ($+0.608$ vs. $+0.584$ for cast) and lower RMSE ($0.224$ vs. $0.227$) on 2020–2021 train dramas, $m = 3$ was kept as a conservative hyperparameter choice for stability against extreme singletons (since 52.4% of lead actors in train appear only once).
  - **Two Test Prior Variants:**
    - **Variant 1 (V1 — Deployment-Style Evaluation):** Prior pool for test dramas includes Train plus earlier Test dramas ($\le \text{current} - 365\text{ days}$). Evaluates continuous real-world model deployment ($r = +0.59$ for cast reach).
    - **Variant 2 (V2 — Static Train-Only Benchmark):** Prior pool for test dramas is strictly limited to the Train set ($\le 2021$). Completely passive evaluation benchmark ($r = +0.57$ for cast reach).
    - Both feature tables are exported to `data/processed/` (`train_features_v1.csv`, `test_features_v1.csv`, `train_features_v2.csv`, `test_features_v2.csv`) and gitignored.

- **Step 4: Synopsis Text Cleaning & Feature Extraction:**
  - **Text Cleaning (`synopsis_clean`):** Stripped `(YYYY)` release year patterns, bare mentions of the drama's own `release_year` (31 dramas affected; 0 remaining), `Source:` attribution lines (e.g. `(Source: ...)`, `Source: Baidu`), MDL `~~` metadata note lines (e.g. `~~ Adapted from...`, `~~ Remake of...`), `Edit Translation` markers, translation credit lines, and redundant whitespace.
  - **Adaptation Signal Note:** The removed `"~~ Adapted from / Remake of"` lines represent a valid pre-release signal (intellectual property / adaptation) and will be considered as a possible binary feature (`is_adaptation`) in a later experiment.
  - **Text Feature (`synopsis_length`):** Number of words in `synopsis_clean`, strictly set to `NaN` when synopsis is missing/empty (NOT 0, ensuring missingness is not encoded as zero length). Correlates $+0.1775$ in train ($N = 4,245$) and $+0.2081$ in test ($N = 2,252$) with `target_pct` on non-missing rows.
  - **Retrieval Index Corpus:** Full cleaned synopses for all 7,535 labeled dramas exported to `data/processed/synopses_clean.csv` (gitignored), keyed by `url` and `name`. Dramas with empty synopses ($N = 1,038$, 13.8%) will be excluded from the retrieval vector index in Milestone 6.
  - **Text Embedding Policy:** SBERT embeddings are NOT built at this stage. Semantic embeddings are reserved for the retriever in Milestone 6, and may be tested as an optional experiment in Milestone 4 using dimensionality reduction fitted on train only.

- **Critical Data Limitations & Caveats for Talent Features:**
  1. **String-Matching Without Person IDs:** Talent names are matched strictly on normalized romanized strings (`lowercase`, stripped parentheses, collapsed whitespace, unaccented). No unique MDL person IDs or URLs exist in this dataset; common romanized names may belong to different real-world individuals (homonym collisions), or split across transliteration variants.
  2. **Scrape-Time Snapshot Labels & Cumulative Growth:** Track records use `target_pct` computed at scrape time. Although the 12-month rule prevents using concurrent or future titles, prior-drama labels include later popularity growth (snapshot at scrape time), which may inflate talent-reach correlations compared to views known at the exact pitch date.
  3. **Ablation Nature of Prior Experience ($n_{\text{prior}}$):** Raw experience count ($n_{\text{prior}}$) exhibits weak and unstable correlation with target reach ($r \approx -0.10 \text{ to } +0.13$). Because the corpus starts in 2015, left-censoring causes earlier titles to mechanically have fewer recorded priors, confounding $n_{\text{prior}}$ with premiere year.
  4. **Reach NaN Patterns & Tree Proxy Risk:** Reach NaN patterns vary significantly with release year (100% in 2015, dropping to 10% in 2020) and country (screenwriter reach is 86.1% NaN in China vs. 55.3% in Thailand). Consequently, a tree model could use reach NaN status as a proxy for premiere time, country, or page completeness rather than true talent quality. Milestone 4 will compare models with and without `talent_reach`, and explore dropping 2015 as a prior-only burn-in year.
  5. **Non-Comparable Correlations (Sample Selection Skew):** All reported correlations between talent priors and `target_pct` are computed strictly on rows where the feature is not `NaN`. Because rows with non-NaN priors are heavily skewed towards popular, well-documented dramas with established industry talent, these correlations are not comparable with correlations evaluated on the full dataset.
  6. **Inherent Missingness Signal in Talent Credits:** ~60% of dramas lack complete credits (`complete_info == 0`). Incomplete titles have 87%–93% missingness in director/screenwriter priors and 55% in cast priors, meaning talent features inherently carry page-completeness signal.
  7. **Incomplete Historical Corpus:** Due to scraper temporal gaps (China 2019–2020, Japan 2018–2020), priors come from an incomplete historical sample and serve as proxy estimates.

---

---

---

## 📈 Milestone 4 Results: Baselines, Feature Ablations, and Evaluation

### Plain-Language Summary
1. Pre-release pitch parameters (broadcaster network, country, genre) and attached talent track records (prior reach of director, writer, and lead cast) predict relative MDL audience-reach tier significantly better than chance (ROC-AUC 0.863 on Full Test, 0.811 on complete metadata).
2. The talent reach lift is clear on the full test set (+0.011 ROC-AUC, +0.024 macro-F1) but not established for ROC-AUC on complete_info (-0.015 [-0.034, +0.004]), while PR-AUC on complete_info rises +0.036 [+0.005, +0.069].
3. The page-completeness baseline is constant (0.500) on strict by construction, and scores 0.569 on complete_info, demonstrating that metadata completeness carries strong post-release confounding on uncurated pages.
4. Models trained exclusively on complete-information dramas achieve comparable test ROC-AUC (0.808–0.810 vs 0.821), supporting the conclusion that the model learns legitimate pre-release relationships rather than relying on training-set missingness shortcuts.
5. In ranking feature groups, attached talent reach, broadcaster network, and genre carry most of the signal.
6. The model's predictions barely depend on episode count (internal validation permutation drop: -0.0026 ROC-AUC), meaning what-if tools should not rely on episode count to alter predicted reach.
7. Post-hoc probability calibration using isotonic regression (selected via lowest pooled out-of-fold validation Brier score of 0.1106) reduces test Brier score from 0.1309 to 0.1273 (+0.0035 improvement, paired bootstrap excludes 0) against a base-rate Brier baseline of 0.1888.
8. What-if adjustments in subsequent tools must be interpreted as associational forecasting benchmarks, not causal interventions.

---

### 1. Foundational Baselines vs. Headline Model

| Model / Baseline | Full Test Accuracy | Full Test Macro-F1 | Full Test ROC-AUC [base: 0.253] | Full Test PR-AUC [base: 0.253] | complete_info ROC-AUC [base: 0.389] | strict ROC-AUC [base: 0.441] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class (Medium)** | 0.499 [0.479, 0.521] | 0.222 [0.215, 0.229] | 0.500 [0.500, 0.500] | 0.253 [0.237, 0.270] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| **Stratified Random** | 0.365 [0.345, 0.381] | 0.324 [0.306, 0.342] | 0.509 [0.485, 0.533] | 0.257 [0.237, 0.277] | 0.505 [0.471, 0.538] | 0.503 [0.463, 0.548] |
| **Country-Only Baseline** | 0.499 [0.479, 0.521] | 0.222 [0.215, 0.229] | 0.501 [0.477, 0.526] | 0.253 [0.237, 0.270] | 0.408 [0.375, 0.442] | 0.428 [0.385, 0.473] |
| **Page Completeness Only** | 0.528 [0.509, 0.548] | 0.502 [0.483, 0.524] | 0.736 [0.714, 0.754] | 0.408 [0.375, 0.437] | 0.569 [0.538, 0.599] | 0.500 [0.500, 0.500] |
| **Headline Model (core + talent_reach)** | **0.659 [0.637, 0.679]** | **0.655 [0.634, 0.676]** | **0.863 [0.847, 0.878]** | **0.676 [0.638, 0.710]** | **0.811 [0.787, 0.835]** | **0.821 [0.788, 0.854]** |

*Notes on Baselines:*
- Country-only is near chance by construction because the target is standardized as a percentile within each country-year cohort. On `complete_info`, Country-only yields 0.408 ROC-AUC as a selection artifact (High-tier proportion differs by country inside that slice: 44.7% in Korea vs 22.9% in Japan), not true negative signal.
- The page-completeness baseline is constant on strict by construction (0.500) because all missingness flags are zero there, and scores 0.569 on `complete_info`.

---

### 2. Headline Model Performance Across Slices & Countries
*Headline configuration: HistGradientBoosting on `core + talent_reach` (47 features, V2 static train-only priors, declared before test evaluation).*

| Slice / Country | $N$ | High Base Rate | Accuracy [95% CI] | Macro-F1 [95% CI] | ROC-AUC (High) [95% CI] | PR-AUC (High) [95% CI] | Spearman ($r_s$) [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Test Set** | **2,525** | **0.253** | **0.659 [0.637, 0.679]** | **0.655 [0.634, 0.676]** | **0.863 [0.847, 0.878]** | **0.676 [0.638, 0.710]** | **+0.751 [0.735, 0.768]** |
| **complete_info Subset** | **1,022** | **0.389** | **0.681 [0.654, 0.709]** | **0.675 [0.639, 0.709]** | **0.811 [0.787, 0.835]** | **0.739 [0.702, 0.779]** | **+0.664 [0.627, 0.700]** |
| **complete_info_strict Subset** | **635** | **0.441** | **0.676 [0.638, 0.711]** | **0.643 [0.584, 0.697]** | **0.821 [0.788, 0.854]** | **0.792 [0.745, 0.836]** | **+0.676 [0.625, 0.723]** |
| South Korea (Full Test) | 751 | 0.253 | 0.670 [0.634, 0.703] | 0.669 [0.633, 0.701] | 0.888 [0.862, 0.911] | 0.752 [0.690, 0.807] | +0.788 [0.758, 0.814] |
| Thailand (Full Test) | 790 | 0.254 | 0.623 [0.589, 0.656] | 0.623 [0.587, 0.656] | 0.851 [0.820, 0.876] | 0.595 [0.527, 0.675] | +0.751 [0.722, 0.781] |
| China (Full Test) | 532 | 0.248 | 0.695 [0.658, 0.733] | 0.687 [0.647, 0.724] | 0.886 [0.856, 0.911] | 0.717 [0.636, 0.786] | +0.774 [0.735, 0.804] |
| Japan (Full Test) | 452 | 0.250 | 0.662 [0.617, 0.701] | 0.643 [0.594, 0.686] | 0.829 [0.783, 0.870] | 0.632 [0.543, 0.718] | +0.700 [0.651, 0.740] |
| South Korea (`complete_info`) | 336 | 0.426 | 0.702 [0.655, 0.750] | 0.566 [0.448, 0.732] | 0.813 [0.764, 0.859] | 0.785 [0.716, 0.849] | +0.663 [0.591, 0.724] |
| Thailand (`complete_info`) | 344 | 0.317 | 0.622 [0.576, 0.672] | 0.629 [0.578, 0.683] | 0.817 [0.771, 0.856] | 0.669 [0.585, 0.751] | +0.719 [0.651, 0.763] |
| China (`complete_info`) | 101 | 0.624 | 0.733 [0.648, 0.812] | 0.703 [0.441, 0.851] | 0.797 [0.708, 0.885] | 0.861 [0.771, 0.929] | +0.531 [0.378, 0.683] |
| Japan (`complete_info`) | 241 | 0.344 | 0.722 [0.660, 0.776] | 0.720 [0.644, 0.775] | 0.805 [0.750, 0.861] | 0.688 [0.597, 0.793] | +0.647 [0.560, 0.721] |
| South Korea (`strict`) | 312 | 0.455 | 0.692 [0.639, 0.744] | 0.575 [0.439, 0.720] | 0.802 [0.754, 0.848] | 0.793 [0.737, 0.853] | +0.639 [0.564, 0.705] |
| Thailand (`strict`) | 253 | 0.368 | 0.632 [0.577, 0.690] | 0.626 [0.565, 0.695] | 0.830 [0.779, 0.878] | 0.729 [0.641, 0.813] | +0.722 [0.643, 0.783] |
| China (`strict`)* | 40 | 0.700 | 0.750 [0.600, 0.888] | 0.667 [0.473, 0.847] | 0.759 [0.606, 0.910] | 0.897 [0.781, 0.971] | +0.558 [0.296, 0.745] |
| Japan (`strict`)* | 30 | 0.567 | 0.767 [0.600, 0.900] | 0.753 [0.576, 0.900] | 0.846 [0.680, 0.973] | 0.889 [0.756, 0.984] | +0.564 [0.210, 0.792] |

*(\*Note: China and Japan strict sample sizes are too small for reliable conclusions).*

---

### 3. Feature Group Ablations & Paired Bootstrap Tests

| Feature Configuration | Full Test ROC-AUC [base: 0.253] | Full Test PR-AUC [base: 0.253] | complete_info ROC-AUC [base: 0.389] | complete_info PR-AUC [base: 0.389] | strict ROC-AUC [base: 0.441] | strict PR-AUC [base: 0.441] | $\Delta$ ROC vs Headline (Full Test) | $\Delta$ Macro-F1 vs Headline (Full Test) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. core** | 0.852 [0.836, 0.867] | 0.642 [0.606, 0.682] | 0.796 [0.770, 0.828] | 0.704 [0.660, 0.751] | 0.807 [0.772, 0.839] | 0.745 [0.685, 0.798] | -0.011 [-0.019, -0.001] * | -0.021 [-0.039, -0.003] * |
| **2. core + talent_reach (Headline)** | **0.863 [0.847, 0.878]** | **0.676 [0.638, 0.710]** | **0.811 [0.787, 0.835]** | **0.739 [0.702, 0.779]** | **0.821 [0.788, 0.854]** | **0.792 [0.745, 0.836]** | **[Reference]** | **[Reference]** |
| **3. core + reach + missingness** | 0.867 [0.851, 0.882] | 0.686 [0.649, 0.721] | 0.818 [0.792, 0.843] | 0.751 [0.714, 0.791] | 0.825 [0.792, 0.857] | 0.796 [0.748, 0.841] | +0.004 [-0.000, +0.008] | +0.012 [+0.001, +0.024] * |
| **4. core + reach + talent_exp** | 0.878 [0.863, 0.892] | 0.694 [0.658, 0.727] | 0.834 [0.808, 0.859] | 0.760 [0.723, 0.801] | 0.839 [0.810, 0.868] | 0.803 [0.752, 0.846] | +0.015 [+0.010, +0.021] * | +0.015 [+0.002, +0.029] * |
| **5. core + reach + text** | 0.868 [0.853, 0.884] | 0.691 [0.654, 0.727] | 0.816 [0.792, 0.841] | 0.748 [0.712, 0.788] | 0.830 [0.800, 0.861] | 0.803 [0.753, 0.845] | +0.006 [+0.000, +0.011] * | +0.006 [-0.009, +0.022] |
| **6. talent_reach only** | 0.769 [0.747, 0.788] | 0.516 [0.474, 0.555] | 0.721 [0.691, 0.750] | 0.595 [0.548, 0.647] | 0.730 [0.689, 0.770] | 0.661 [0.602, 0.724] | -0.094 [-0.110, -0.079] * | -0.193 [-0.217, -0.171] * |
| **7. all groups** | 0.882 [0.868, 0.895] | 0.709 [0.674, 0.743] | 0.838 [0.813, 0.861] | 0.767 [0.731, 0.807] | 0.845 [0.815, 0.873] | 0.809 [0.760, 0.850] | +0.019 [+0.012, +0.026] * | +0.017 [+0.002, +0.036] * |
| **8. Drop 2015 burn-in** | 0.867 [0.852, 0.881] | 0.682 [0.646, 0.718] | 0.813 [0.788, 0.837] | 0.741 [0.705, 0.783] | 0.820 [0.786, 0.851] | 0.789 [0.743, 0.832] | +0.004 [+0.000, +0.008] * | -0.006 [-0.017, +0.004] |
| **9. V1 rolling priors** | 0.868 [0.852, 0.883] | 0.688 [0.652, 0.724] | 0.821 [0.796, 0.847] | 0.753 [0.717, 0.791] | 0.830 [0.799, 0.859] | 0.803 [0.759, 0.844] | +0.005 [+0.002, +0.009] * | +0.001 [-0.006, +0.007] |
| **10. Linear Reference (LogisticReg)** | 0.864 [0.848, 0.879] | 0.673 [0.638, 0.711] | 0.818 [0.794, 0.843] | 0.741 [0.703, 0.785] | 0.833 [0.802, 0.862] | 0.797 [0.748, 0.842] | +0.001 [-0.007, +0.009] | -0.037 [-0.052, -0.019] * |

---

### 4. Shortcut-Free Training Experiments

| Training Split | Evaluation Slice | Accuracy [95% CI] | Macro-F1 [95% CI] | ROC-AUC (High) [95% CI] | PR-AUC (High) [95% CI] | Spearman ($r_s$) [95% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Trained on ALL** ($N=5,010$) | `complete_info` test ($N=1,022$) | 0.681 [0.654, 0.709] | 0.675 [0.639, 0.709] | 0.811 [0.787, 0.835] | 0.739 [0.702, 0.779] [base: 0.389] | +0.664 [0.627, 0.700] |
| **Trained on complete_info** ($N=2,022$) | `complete_info` test ($N=1,022$) | 0.651 [0.623, 0.682] | 0.633 [0.593, 0.668] | 0.801 [0.776, 0.830] | 0.727 [0.688, 0.773] [base: 0.389] | +0.647 [0.610, 0.685] |
| **Trained on ALL** ($N=5,010$) | `strict` test ($N=635$) | 0.676 [0.638, 0.711] | 0.643 [0.584, 0.697] | 0.821 [0.788, 0.854] | 0.792 [0.745, 0.836] [base: 0.441] | +0.676 [0.625, 0.723] |
| **Trained on complete_info** ($N=2,022$) | `strict` test ($N=635$) | 0.647 [0.605, 0.686] | 0.596 [0.533, 0.649] | 0.810 [0.777, 0.843] | 0.767 [0.716, 0.815] [base: 0.441] | +0.661 [0.612, 0.709] |
| **Trained on strict** ($N=1,072$) | `strict` test ($N=635$) | 0.661 [0.625, 0.698] | 0.584 [0.523, 0.638] | 0.808 [0.774, 0.841] | 0.763 [0.713, 0.813] [base: 0.441] | +0.654 [0.603, 0.701] |

*Interpretation:*
These results **support** the conclusion that the model learns legitimate pre-release relationships rather than relying on missingness shortcuts. Because `complete_info` still allows missing `content_rating`, `aired_on`, and `genres`, the model trained strictly on `complete_info_strict` ($N=1,072$) and evaluated on `strict` test ($N=635$) is the cleanest test: its ROC-AUC (0.808) is within sampling error of the model trained on all data (0.821), showing that missingness shortcuts during training are not required to achieve strong ranking performance on clean data.

---

### 5. Signal Decomposition & Validation Permutation Importance

| Feature Sub-group | Isolated ROC-AUC (Full Test) | Isolated Macro-F1 (Full Test) | LOO Retrained ROC-AUC (Full Test) | Mean Drop in Val ROC-AUC ($\pm$ std) | Mean Drop in Val Macro-F1 ($\pm$ std) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **talent_reach** (6 features) | 0.769 | 0.516 | 0.852 | **+0.0965 ($\pm 0.0084$)** | **+0.1079 ($\pm 0.0084$)** |
| **genres** (30 features) | 0.745 | 0.473 | 0.849 | +0.0417 ($\pm 0.0050$) | +0.0491 ($\pm 0.0053$) |
| **content_rating** (1 feature) | 0.689 | 0.361 | 0.849 | +0.0416 ($\pm 0.0045$) | +0.0283 ($\pm 0.0074$) |
| **network + country** (2 features) | 0.740 | 0.449 | 0.852 | +0.0376 ($\pm 0.0071$) | +0.0685 ($\pm 0.0115$) |
| **weekdays** (7 features) | 0.636 | 0.414 | 0.852 | +0.0290 ($\pm 0.0038$) | +0.0162 ($\pm 0.0051$) |
| **episodes** (1 feature) | 0.582 | 0.278 | 0.865 | -0.0026 ($\pm 0.0018$) | -0.0053 ($\pm 0.0046$) |

*Methodological Note:* Group permutation importance (evaluating drops on a fixed model on internal validation) and leave-one-out retraining answer different questions: permutation measures how much the existing model relies on the group in the presence of other features, while LOO measures whether the model can compensate by retraining using remaining correlated features. Both analyses confirm that **the model's predictions barely depend on episode count**, so what-if tools should not rely on episode count to alter predicted success.

---

### 6. Probability Calibration: Equal-Count Deciles & Out-of-Fold Selection
*Calibrators were fitted strictly on pooled out-of-fold validation predictions from expanding-window folds (<=2017 -> 2018, <=2018 -> 2019, <=2019 -> 2020-2021, N = 2,609). Calibrator choice was determined by the rule of **lowest out-of-fold Brier score** before evaluating on test data:*
- Out-of-Fold Raw Brier: `0.1188`
- Out-of-Fold Platt Brier: `0.1137`
- Out-of-Fold Isotonic Brier: `0.1106` *(Selected: Isotonic Regression)*

*Evaluated on Full Test (N=2,525) using 10 equal-count (decile) bins (N ~ 252-253 dramas per bin). The Brier score baseline of predicting the test base rate (0.2527 * 0.7473) is **0.1888**:*

| Metric / Decile Bin | Raw Model | Platt Scaling | Isotonic Regression (Selected) |
| :--- | :---: | :---: | :---: |
| **Full Test Brier Score** (baseline = 0.1888) | 0.1309 | 0.1296 | **0.1273** |
| **Equal-Count ECE (Deciles)** | 0.0487 | 0.0515 | **0.0360** |
| **Brier Improvement vs Raw** (500 resamples) | [Reference] | +0.0013 [-0.0018, +0.0044] | **+0.0035 [+0.0006, +0.0064] \*** |
| **ECE Improvement vs Raw** (500 resamples) | [Reference] | -0.0028 [-0.0157, +0.0174] | +0.0126 [-0.0045, +0.0336] |
| Decile 1 ($N=253$) | Pred: 0.008, Obs: 0.008 (+0.001) | Pred: 0.043, Obs: 0.012 (+0.031) | Pred: 0.000, Obs: 0.000 (+0.000) |
| Decile 2 ($N=252$) | Pred: 0.021, Obs: 0.004 (+0.017) | Pred: 0.055, Obs: 0.008 (+0.047) | Pred: 0.005, Obs: 0.007 (-0.001) |
| Decile 3 ($N=253$) | Pred: 0.044, Obs: 0.020 (+0.024) | Pred: 0.076, Obs: 0.012 (+0.064) | Pred: 0.021, Obs: 0.016 (+0.005) |
| Decile 4 ($N=252$) | Pred: 0.081, Obs: 0.079 (+0.001) | Pred: 0.108, Obs: 0.083 (+0.025) | Pred: 0.032, Obs: 0.070 (-0.038) |
| Decile 5 ($N=252$) | Pred: 0.141, Obs: 0.143 (-0.002) | Pred: 0.158, Obs: 0.147 (+0.011) | Pred: 0.075, Obs: 0.108 (-0.032) |
| Decile 6 ($N=253$) | Pred: 0.226, Obs: 0.237 (-0.011) | Pred: 0.230, Obs: 0.237 (-0.007) | Pred: 0.129, Obs: 0.203 (-0.074) |
| Decile 7 ($N=252$) | Pred: 0.350, Obs: 0.286 (+0.064) | Pred: 0.339, Obs: 0.286 (+0.053) | Pred: 0.211, Obs: 0.267 (-0.055) |
| Decile 8 ($N=253$) | Pred: 0.516, Obs: 0.411 (+0.105) | Pred: 0.490, Obs: 0.407 (+0.083) | Pred: 0.312, Obs: 0.346 (-0.034) |
| Decile 9 ($N=252$) | Pred: 0.708, Obs: 0.571 (+0.137) | Pred: 0.669, Obs: 0.583 (+0.086) | Pred: 0.537, Obs: 0.570 (-0.033) |
| Decile 10 ($N=253$) | Pred: 0.891, Obs: 0.767 (+0.125) | Pred: 0.840, Obs: 0.763 (+0.077) | Pred: 0.809, Obs: 0.730 (+0.079) |

*Calibration Diagnostics & Limitations:*
- Raw model probabilities are well-ordered across deciles but overconfident in upper deciles D7–D10 (e.g. Decile 9 predicts 70.8% High vs 57.1% observed).
- Isotonic regression fitted on pooled expanding-window validation significantly improves Brier score (+0.0035, 95% CI: [+0.0006, +0.0064], distinguishable from noise) and reduces equal-count ECE from 4.87% to 3.60%.
- The calibrator was selected using Brier scores computed on the same out-of-fold predictions it was fitted on, which favors flexible methods (isotonic); the test-set improvement (Brier +0.0035 [+0.0006, +0.0064]) is the evidence that matters, and leave-one-fold-out selection would be cleaner.
- The out-of-fold predictions come from models trained on smaller windows than the final model.

---

### 7. Methodological Guardrails & Limitations
1. **Headline Feature Selection:** The headline (`core + talent_reach`, V2 priors) was declared before test evaluation and is retained even though "all groups" scores slightly higher. Including `talent_experience` and `text` introduces premiere-year confounding and page-completeness bias.
2. **Multiple Comparisons Caution:** Many comparisons were run across feature groups and subsets; results with an interval that barely excludes zero should be treated as suggestive rather than definitive.
3. **Metric Stability on Subsets:** On `complete_info` and strict complete info, Macro-F1 is unstable because the Low class represents only ~7.5%–9.5% of the dramas. Therefore, ROC-AUC, PR-AUC, and Spearman correlation are the primary metrics on these slices.
4. **Text Feature Finding:** Adding `synopsis_length` **does not help** (it lowers complete_info Macro-F1 by -0.048). As hypothesized, synopsis presence acts primarily as a page-completeness artifact rather than a substantive predictor of pitch quality.
5. **Associational vs. Causal Interpretation:** All predictions and what-if scenario simulations in subsequent milestones are associational forecasting tools, not causal interventions.
6. **Bootstrap Independence Limitation:** 500 bootstrap resamples treat dramas as independent observations even though dramas frequently share talent, directors, and sequel universes.

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
- [x] **Milestone 3:** Leakage-free feature engineering (`notebooks/05_feature_engineering.ipynb`, `src/features.py`).
- [x] **Milestone 4:** Time-based train/test splitting and baseline predictive modeling.
- [ ] **Milestone 5:** Model interpretability with SHAP.
- [ ] **Milestone 6:** RAG retriever for comparable dramas.
- [ ] **Milestone 7:** What-if simulator and LLM agent integration.
- [ ] **Milestone 8:** FastAPI, Dockerization, and Hugging Face Spaces deployment.

