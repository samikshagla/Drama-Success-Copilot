### 1. What Was Assessed
Production assessment for proposed drama originating in South Korea. Planned release schedule targets premiere on 2024-05-01 with 12 planned episodes; scheduled for broadcast on Wednesday, Thursday. Network not specified, content rating not specified, and planned genres: Crime, Action. Individual creator names and narrative synopsis text are excluded from assessment.

### 2. Likelihood Band & Historical Track Record
Model-Assigned Likelihood Band: **Unlikely**
Empirical benchmark: **7.0% [95% CI: 5.8%, 8.4%] (N=1,436; observed on test dramas released 2022-2025, relative to dramas of the same country and year)** among the 2,525 evaluated test dramas (released 2022-2025, 6+ episodes, all four countries) that the model placed in this band. High = the top quarter of MyDramaList watchers among dramas of the same country and release year.

### 3. Main Predictive Drivers
- **Top Positive Feature Groups:** None identified.
- **Top Negative Feature Groups:** `genres` (negative), `network` (negative).
- Country is a cohort baseline and planned episode count had little influence on this assessment.
- Talent track record: no qualifying history found for the attached talent, so talent could not contribute to this assessment

### 4. Advisory Warnings & Data Hygiene Alerts
- *Alert:* Broadcaster network is not specified. The model predicts lower likelihood of the High tier when this field is left blank than when a typical broadcaster is attached; specify a network for an accurate assessment.
- *Alert:* Content rating is not specified. The model predicts lower likelihood of the High tier when this field is left blank than when a typical rating is attached; specify a content rating if known.
- *Alert:* No qualifying prior credits found within the 12-month rule. The model cannot distinguish a debut creator from an unlisted track record; prior reach is treated as unknown.

### 5. Comparable Historical Dramas
| Rank | Title | Release Year | Country | Broadcaster | Reach Tier | URL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Mouse (2021) | 2021 | South Korea | tvN | High | https://mydramalist.com/58953-mouse |
| 2 | Cheo Yong Season 2 (2015) | 2015 | South Korea | OCN | Medium | https://mydramalist.com/13559-cheo-yong-2 |
| 3 | Vampire Detective (2016) | 2016 | South Korea | OCN | High | https://mydramalist.com/9125-vampire-detective |

> [!NOTE]
> **Retrieval Scope Caveat:** comparables show similar premises; their tiers are not a probability for this pitch.

### 6. Sensitivity Simulation (What-If Analysis)
- **Attachment Changes Evaluated:** Network: jTBC; Content rating: 15+

| State | Likelihood Band | Empirical Benchmark (Observed Test Rate) |
| :--- | :--- | :--- |
| Before | Unlikely | 7.0% [95% CI: 5.8%, 8.4%] (N=1,436; observed on test dramas released 2022-2025, relative to dramas of the same country and year) |
| After | Possible | 31.7% [95% CI: 27.9%, 35.8%] (N=527; observed on test dramas released 2022-2025, relative to dramas of the same country and year) |

- **Blank-Field Effect:** Part of this change reflects the model's lower likelihood of the High tier when a field is blank; most of the lift comes from filling in the blank fields, not from the specific choices. Modifications filled unlisted fields (network, content_rating). About two thirds of the raw gain comes from the fields being filled in rather than specific choices, reflecting the baseline statistical penalty for unlisted metadata in the training corpus.
- **Calibration Step Alert:** band edges are approximate: the calibrator is a step function.

> [!IMPORTANT]
> **Sensitivity Caveat:** Results are associational, not causal; changing attachments does not guarantee audience reach shifts.

### 7. Methodological Limitations & Governance Notice
Forecasts and attributions reflect observational correlations from historical training data, not causal treatment effects, commercial earnings predictions, or performance guarantees. Actual audience reach is subject to script execution, marketing investments, competition, and platform scheduling.