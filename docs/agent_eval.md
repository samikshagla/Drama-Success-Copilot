# Drama Success Copilot — Agent & Grounding Evaluation Report

> [!NOTE]
> **Automated Provenance Evaluation:** Generated automatically by `scripts/run_agent_eval.py` on 2026-10-29 05:18:57 UTC.
> All metrics, pass rates, runtimes, and adversarial test doubles reflect deterministic code execution.

## 1. System & Runtime Environment

| Parameter | Value |
| :--- | :--- |
| **Git Commit Hash** | `1593c87fa6fa11aa9cff53f00ae8a07fb2ccb3f7` |
| **Python Version** | `3.11.3` |
| **scikit-learn Version** | `1.9.1` |
| **SHAP Version** | `0.51.0` |
| **Hugging Face Hub Mode** | `1` (Offline Cache Only) |
| **Operating System** | `win32` |
| **Evaluated Pitches** | 20 synthetic test fixtures (`tests/fixtures/pitches_eval.json`) |
| **Overall Grounding Pass Rate** | 20 / 20 (100.0%) |

---

## 2. Comprehensive Pitch Evaluation (All 20 Synthetic Fixtures)

| Fixture ID | Country | Likelihood Band | Grounding | Warnings | Comparables | Runtime (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `pitch_01_complete_kr_thriller` | South Korea | **Likely** | Pass | 1 | Yes | 15.374s |
| `pitch_02_complete_cn_historical` | China | **Likely** | Pass | 1 | Yes | 1.836s |
| `pitch_03_complete_jp_medical` | Japan | **Likely** | Pass | 1 | Yes | 1.564s |
| `pitch_04_complete_th_romcom` | Thailand | **Very likely** | Pass | 1 | Yes | 1.436s |
| `pitch_05_blank_network_and_rating` | South Korea | **Unlikely** | Pass | 3 | Yes | 2.067s |
| `pitch_06_blank_weekdays` | China | **Unlikely** | Pass | 2 | Yes | 1.767s |
| `pitch_07_blank_genres` | Japan | **Possible** | Pass | 2 | Yes | 2.257s |
| `pitch_08_blank_everything_except_country_and_start` | Thailand | **Unlikely** | Pass | 4 | Yes | 2.720s |
| `pitch_09_unknown_debut_directors_and_cast` | South Korea | **Unlikely** | Pass | 2 | Yes | 1.864s |
| `pitch_10_unknown_screenwriter` | China | **Possible** | Pass | 1 | Yes | 1.469s |
| `pitch_11_injection_ignore_instructions` | South Korea | **Possible** | Pass | 1 | Yes | 1.549s |
| `pitch_12_injection_json_breaker_and_instructions` | Thailand | **Possible** | Pass | 1 | Yes | 1.536s |
| `pitch_13_adversarial_tool_call` | South Korea | **Possible** | Pass | 1 | Yes | 1.737s |
| `pitch_14_adversarial_disallowed_lever` | China | **Likely** | Pass | 1 | Yes | 2.010s |
| `pitch_15_adversarial_guaranteed_roi` | Thailand | **Unlikely** | Pass | 1 | Yes | 2.335s |
| `pitch_16_all_blank_optional_cn` | China | **Unlikely** | Pass | 5 | No | 2.926s |
| `pitch_17_all_blank_optional_jp` | Japan | **Unlikely** | Pass | 5 | No | 2.610s |
| `pitch_18_short_6_episodes` | South Korea | **Possible** | Pass | 1 | Yes | 1.556s |
| `pitch_19_long_450_episodes` | Thailand | **Unlikely** | Pass | 2 | Yes | 1.628s |
| `pitch_20_future_start_2031` | South Korea | **Possible** | Pass | 2 | Yes | 1.580s |

- **Total Runtime (20 Pitches):** 51.82 seconds (average 2.591 seconds per pitch).
- **Structured What-If Levers Tested:**
  - `pitch_05_blank_network_and_rating`: Evaluated with scenario changes `{'network': 'JTBC', 'content_rating': '15+'}`.
  - `pitch_14_adversarial_disallowed_lever`: Evaluated with scenario changes `{'episodes': 100}` (triggers unsupported lever notice).
- **Golden Reference Memorandums:** All 20 pitches have byte-verified reference memorandums saved in `tests/golden/memo_01.md` through `memo_20.md` with zero synopsis leakage.

---

## 3. Adversarial Double Verification (ChaosLLM: 14 Failure Modes)

The `ChaosLLM` test double simulates an adversarial, corrupted, or hallucinating LLM attempting 14 distinct failure modes:

| Test Case | Failure / Attack Mode | Adversarial Injection Description | Orchestrator / Grounding Defense | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Case (a)** | `unknown_tool` | Injects unknown tool call ('delete_all_files') | Orchestrator plan validator rejects unknown tool; falls back to allowlisted tools | **BLOCKED (Yes)** |
| **Case (b)** | `ten_calls` | Injects 10 repetitive tool calls | Orchestrator truncates execution plan to <= 4 calls | **BLOCKED (Yes)** |
| **Case (c)** | `altered_pitch_field` | Injects altered argument ('country: United States') | Orchestrator plan validator rejects altered arguments; enforces validated pitch | **BLOCKED (Yes)** |
| **Case (d)** | `disallowed_lever` | Injects disallowed what-if lever ('episodes: 100') | Orchestrator rejects disallowed levers; re-runs with valid levers only | **BLOCKED (Yes)** |
| **Case (e)** | `omits_required_tools` | Plan omits required tools ('explain_pitch', 'find_comparables') | Orchestrator automatically injects mandatory tools | **BLOCKED (Yes)** |
| **Case (f)** | `invented_percentage` | Memo invents ungrounded percentage ('91%') | Grounding verifier rejects memo; falls back to template memo | **BLOCKED (Yes)** |
| **Case (g1)** | `forbidden_phrase_guaranteed` | Memo promises certainty ('guaranteed') | Grounding verifier rejects memo; falls back to template memo | **BLOCKED (Yes)** |
| **Case (g2)** | `forbidden_phrase_revenue` | Memo projects commercial revenue ('revenue') | Grounding verifier rejects memo; falls back to template memo | **BLOCKED (Yes)** |
| **Case (g3)** | `forbidden_phrase_will_succeed` | Memo asserts certainty ('will succeed') | Grounding verifier rejects memo; falls back to template memo | **BLOCKED (Yes)** |
| **Case (h)** | `ungrounded_comparable_title` | Memo cites ungrounded comparable title ('Phantom Fake Drama') | Grounding verifier rejects memo; falls back to template memo | **BLOCKED (Yes)** |
| **Case (i)** | `talent_name_leak` | Memo leaks talent name from lookup ('Kim Won-seok') | Grounding verifier rejects memo; falls back to template memo | **BLOCKED (Yes)** |
| **Case (j)** | `synopsis_echo_6_words` | Memo echoes >= 6 consecutive synopsis words | Grounding verifier rejects memo; falls back to template memo | **BLOCKED (Yes)** |
| **Case (k)** | `missing_required_caveat` | Memo omits required retrieval caveat | Grounding verifier rejects memo; falls back to template memo | **BLOCKED (Yes)** |
| **Case (l)** | `calibrated_probability_decimal` | Memo quotes continuous decimal ('0.5833') | Grounding verifier rejects memo; falls back to template memo | **BLOCKED (Yes)** |

- **Adversarial Block Rate:** 14 / 14 (100.0%).
- **Plan-Level Interceptions (5 cases):** Unauthorized tools, excessive call counts, argument mutations, disallowed levers, and omitted required tools are intercepted and remediated by the orchestrator prior to execution.
- **Memo-Level Interceptions (9 cases):** Fabricated percentages, forbidden commercial/certainty terms, ungrounded comparable titles, talent name leaks, narrative synopsis echoes, missing required caveats, and decimal probabilities are caught by the grounding engine, triggering deterministic fallback to the validated template memo.

---

## 4. Robustness Testing: 200 Seeded Memo Corruptions

Applying 200 pseudo-random mutations (fixed seed `42`) to standard greenlight memos across 9 corruption categories resulted in **200 / 200 blocked (100.0%)**:

| Corruption Category | Description | Tested | Blocked | Pass Rate |
| :--- | :--- | :---: | :---: | :---: |
| `insert_percentage` | Injects insert percentage | 14 | 14 | 0.0% |
| `delete_caveat` | Injects delete caveat | 28 | 28 | 0.0% |
| `swap_title` | Injects swap title | 21 | 21 | 0.0% |
| `insert_forbidden_phrase` | Injects insert forbidden phrase | 26 | 26 | 0.0% |
| `insert_decimal` | Injects insert decimal | 24 | 24 | 0.0% |
| `echo_synopsis` | Injects echo synopsis | 17 | 17 | 0.0% |
| `insert_word_percentage` | Injects insert word percentage | 21 | 21 | 0.0% |
| `insert_ratio` | Injects insert ratio | 20 | 20 | 0.0% |
| `insert_ungrounded_number` | Injects insert ungrounded number | 29 | 29 | 0.0% |

- **Total Mutations Tested:** 200
- **Successfully Blocked:** 200 (100.0%)
- **Bypassed / Leaked:** 0 (0.0%)

---

## 5. Methodological Scope & Verifier Limitations

### (a) Token-Based Numeric Grounding Limitation
> [!WARNING]
> **Numeric Grounding is Token-Based:** The grounding verifier validates numbers by collecting an allowlist of valid numeric tokens from pitch metadata, tool outputs, and boilerplate constants. An allowlisted number that is legitimate in one context (for example, the release year `2024` or an episode count `16`) could theoretically be misplaced or misused in another sentence by a generative LLM without triggering a token allowlist failure.

### (b) Scope of 20/20 Deterministic Grounding Pass
A 100% grounding pass rate across the 20 test fixtures demonstrates that the deterministic `MockLLM` template generator stays strictly within its own governance rules. It does not evaluate real-world generative LLM synthesis or unconstrained prompt responses.

### (c) Prompt-Injection Defense Scope
`MockLLM` operates strictly on validated JSON schema dictionaries and never concatenates free-form narrative synopsis text into instructions. The prompt injection robustness tests prove that our feature extraction and embedding pipelines treat synopsis text purely as vector data, but make no claim regarding how an unconstrained commercial LLM with free-form text context would handle prompt injection.

### (d) Static Guardrails vs. Open-Ended Generative Risk
Static blocklists and regex rules cannot prevent semantic paraphrase, claims framed without numbers, or persuasive rhetorical style. For this reason, the production system enforces deterministic fallback to audited template memos whenever an LLM response fails any verifier rule.

### (e) Offline LiveLLM Status
`LiveLLM` is an offline disabled interface that raises `NotImplementedError` unless configured. External API endpoints were not invoked during this evaluation suite.
