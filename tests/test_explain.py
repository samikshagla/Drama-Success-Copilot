"""
Unit and Integration Tests for src.explain (Milestone 5)
=========================================================
Tests cover:
1. SHAP additivity within 1e-9 on 5 diverse dramas.
2. JSON-serializability and presence of all required dictionary keys.
3. Debut talent (no prior history) producing NaN reach and standard warning.
4. Blank broadcaster network producing warning and missing encoding.
5. Seed consistency: two random seeds agree on top driver group and sign for 3 dramas.
6. Feature builder fidelity: build_feature_row reproduces all 47 features for 20 test dramas.
7. 12-month rule enforcement: prior credits starting within 365 days of planned start are strictly excluded.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.explain import (
    load_artifacts,
    explain_drama,
    build_feature_row,
    GROUP_DEFS,
)
from src.modeling import load_modeling_data
from src.features import (
    load_labeled_data,
    prepare_talent_columns,
    build_talent_inverted_index,
    temporal_train_test_split,
)

repo_root = Path(__file__).resolve().parent.parent

REQUIRED_MODEL_FILES = [
    repo_root / "models/headline_model.joblib",
    repo_root / "models/isotonic_calibrator.joblib",
    repo_root / "models/background_sample.joblib",
    repo_root / "models/talent_lookup.joblib",
    repo_root / "models/metadata.joblib",
]
REQUIRED_DATA_FILES = [
    repo_root / "data/processed/dramas_labeled.csv",
    repo_root / "data/processed/train_features_v2.csv",
    repo_root / "data/processed/test_features_v2.csv",
]

missing_models = [p.relative_to(repo_root).as_posix() for p in REQUIRED_MODEL_FILES if not p.exists()]
missing_data = [p.relative_to(repo_root).as_posix() for p in REQUIRED_DATA_FILES if not p.exists()]
missing_artifacts = missing_models + missing_data

pytestmark = pytest.mark.skipif(
    bool(missing_artifacts),
    reason=(
        f"Missing artifacts or data required for tests: {', '.join(missing_artifacts)}. "
        "Follow README 'Rebuilding artifacts and running tests' section to generate them."
    ),
)


@pytest.fixture(scope="module")
def artifacts():
    """Load cached model artifacts once for the test session."""
    return load_artifacts()


@pytest.fixture(scope="module")
def train_only_artifacts(artifacts):
    """Artifacts equipped with train-only talent lookup matching test_df construction."""
    labeled_df = load_labeled_data("data/processed/dramas_labeled.csv")
    train_raw, _ = temporal_train_test_split(labeled_df)
    train_prep = prepare_talent_columns(train_raw)
    exact_global_mean = float(train_prep["target_pct"].mean())
    d_idx, s_idx, c_idx = build_talent_inverted_index(train_prep)

    train_only_lookup = {
        "director": {p: [(d, float(t)) for d, t, _, _ in ent] for p, ent in d_idx.items()},
        "screenwriter": {p: [(d, float(t)) for d, t, _, _ in ent] for p, ent in s_idx.items()},
        "cast": {p: [(d, float(t)) for d, t, _, _ in ent] for p, ent in c_idx.items()},
        "_d_idx": d_idx,
        "_s_idx": s_idx,
        "_c_idx": c_idx,
        "global_mean": exact_global_mean,
        "smoothing_m": 3.0,
    }
    art_train = artifacts.copy()
    art_train["talent_lookup"] = train_only_lookup
    meta = art_train["metadata"].copy()
    meta["global_mean"] = exact_global_mean
    art_train["metadata"] = meta
    return art_train


DIVERSE_5_DRAMAS = [
    {
        "title": "Who Rules the World",
        "country": "China",
        "network": "Tencent Video",
        "genres": ["Romance", "Wuxia", "Fantasy"],
        "content_rating": "13+",
        "weekdays": ["Monday", "Tuesday", "Wednesday"],
        "episodes": 40,
        "director": "Yin Tao",
        "cast": ["Yang Yang", "Zhao Lu Si"],
        "planned_start_date": "2022-04-18"
    },
    {
        "title": "The Royal Bracelet",
        "country": "Thailand",
        "network": "Channel 7",
        "genres": ["Historical", "Drama"],
        "content_rating": "13+",
        "weekdays": ["Monday", "Tuesday", "Wednesday", "Thursday"],
        "episodes": 30,
        "director": "Thanapol Phangdee",
        "cast": ["Thanwa Suriyajak", "Tubtim Anyarin Terathananpat"],
        "planned_start_date": "2024-01-15"
    },
    {
        "title": "Korean Hit Pitch",
        "country": "South Korea",
        "network": "tvN",
        "genres": ["Comedy", "Romance"],
        "content_rating": "15+",
        "weekdays": ["Saturday", "Sunday"],
        "episodes": 16,
        "director": "Kim Won-seok",
        "cast": ["IU", "Lee Sun-kyun"],
        "planned_start_date": "2024-05-01"
    },
    {
        "title": "Japanese Mystery Pitch",
        "country": "Japan",
        "network": "TBS",
        "genres": ["Mystery", "Thriller"],
        "content_rating": "15+",
        "weekdays": ["Sunday"],
        "episodes": 10,
        "planned_start_date": "2024-10-01"
    },
    {
        "title": "Minimal Incomplete Pitch",
        "country": "South Korea",
        "network": None,
        "genres": [],
        "content_rating": None,
        "weekdays": [],
        "episodes": 12,
        "planned_start_date": "2024-06-01"
    }
]


def test_shap_additivity_5_dramas(artifacts):
    """(a) SHAP additivity: base_value + sum(group_attributions) == raw_p_high within 1e-9 for 5 diverse dramas."""
    for drama in DIVERSE_5_DRAMAS:
        res = explain_drama(drama, artifacts, seed=42, max_evals=500)
        base = res["base_value"]
        total_attrib = sum(res["group_attributions"].values())
        raw_p = res["raw_p_high"]
        diff = abs((base + total_attrib) - raw_p)
        assert diff < 1e-9, f"SHAP additivity failed for {drama['title']}: diff = {diff:.2e}"
        assert 0.02 <= res["calibrated_p_high"] <= 0.98, f"Calibrated P(High) out of [0.02, 0.98] bounds: {res['calibrated_p_high']}"


def test_json_serializable_and_expected_keys(artifacts):
    """(b) JSON-serializability: json.dumps(result) succeeds and produces all expected keys."""
    sample_drama = DIVERSE_5_DRAMAS[0]
    res = explain_drama(sample_drama, artifacts, seed=42, max_evals=500)
    
    # Must serialize and deserialize cleanly
    dumped = json.dumps(res)
    loaded = json.loads(dumped)
    assert isinstance(loaded, dict)
    
    expected_keys = [
        "title",
        "predicted_tier",
        "raw_p_high",
        "calibrated_p_high",
        "calibrated_p_high_rounded_5pct",
        "likelihood_band",
        "likelihood_band_definition",
        "band_observed_rate",
        "base_value",
        "group_attributions",
        "top_positive_groups",
        "top_negative_groups",
        "warnings",
        "caveats",
    ]
    for key in expected_keys:
        assert key in loaded, f"Expected key '{key}' missing from explain_drama output!"

    assert loaded["likelihood_band"] in ["Unlikely", "Possible", "Likely", "Very likely"]
    from src.explain import band_for
    assert loaded["likelihood_band"] == band_for(loaded["calibrated_p_high"])
    assert "observed on test dramas released 2022-2025, relative to dramas of the same country and year" in loaded["band_observed_rate"]
    assert 0.02 <= loaded["calibrated_p_high"] <= 0.98
    assert len(loaded["group_attributions"]) == len(GROUP_DEFS)


def test_no_talent_history_returns_nan_reach_and_warning(artifacts):
    """(c) No talent history: debut talent gets NaN reach and standard warning."""
    inputs_debut = {
        "title": "Debut Feature Pitch",
        "country": "South Korea",
        "director": "Totally New Debut Director 2099",
        "screenwriter": "Debut Screenwriter 2099",
        "cast": ["Brand New Actor A 2099", "Brand New Actor B 2099"],
        "planned_start_date": "2024-05-01",
    }
    df_built, warnings = build_feature_row(inputs_debut, artifacts)
    
    # Prior reach features must be NaN
    assert np.isnan(df_built["director_prior_reach_max"].iloc[0])
    assert np.isnan(df_built["director_prior_reach_mean"].iloc[0])
    assert np.isnan(df_built["screenwriter_prior_reach_max"].iloc[0])
    assert np.isnan(df_built["screenwriter_prior_reach_mean"].iloc[0])
    assert np.isnan(df_built["cast_prior_reach_max"].iloc[0])
    assert np.isnan(df_built["cast_prior_reach_mean"].iloc[0])

    # Warning text check
    assert any("No qualifying prior credits found within the 12-month rule; feature set to missing" in w for w in warnings)


def test_blank_network_returns_warning(artifacts):
    """(d) Blank network: blank network produces warning and missing encoding."""
    inputs_blank = {
        "title": "Unattached Indie Drama",
        "country": "China",
        "network": None,
    }
    df_built, warnings = build_feature_row(inputs_blank, artifacts)
    net_val = df_built["original_network"].iloc[0]
    
    assert np.isnan(net_val) or net_val == -1.0
    assert any("Network is blank; encoded as missing" in w for w in warnings)


def test_two_seeds_agree_on_sign_and_top_group_for_3_dramas(artifacts):
    """(e) Two seeds agree: two runs with different seeds produce the same sign and top-1 group for 3 dramas."""
    test_subset = [DIVERSE_5_DRAMAS[0], DIVERSE_5_DRAMAS[1], DIVERSE_5_DRAMAS[2]]
    for drama in test_subset:
        r1 = explain_drama(drama, artifacts, seed=42, max_evals=1000)
        r2 = explain_drama(drama, artifacts, seed=123, max_evals=1000)
        
        top1 = list(r1["group_attributions"].keys())[0]
        top2 = list(r2["group_attributions"].keys())[0]
        sign1 = np.sign(r1["group_attributions"][top1])
        sign2 = np.sign(r2["group_attributions"][top2])
        
        assert top1 == top2, f"Top group mismatch for '{drama['title']}': {top1} vs {top2}"
        assert sign1 == sign2, f"Sign mismatch for '{drama['title']}': {sign1} vs {sign2}"


def test_feature_builder_reproduces_training_table_values_for_20_random_test_dramas(train_only_artifacts):
    """(f) Feature builder test: reconstruct inputs for 20 random test dramas and verify all 47 features match test_v2."""
    artifacts = train_only_artifacts
    features = artifacts["metadata"]["features"]
    cat_cols = artifacts["metadata"]["cat_cols"]
    enc = artifacts["metadata"]["ordinal_encoder"]

    _, test_df = load_modeling_data("v2")
    labeled_df = load_labeled_data("data/processed/dramas_labeled.csv")

    X_test_expected = test_df[features].copy()
    X_test_expected[cat_cols] = enc.transform(X_test_expected[cat_cols].astype(str))
    X_test_expected = X_test_expected.astype(np.float64)

    # Identify rows where test index collided with creator's train row index in Milestone 4 construction
    _, test_raw = temporal_train_test_split(labeled_df)
    test_prep = prepare_talent_columns(test_raw)
    d_idx = train_only_artifacts["talent_lookup"]["_d_idx"] if "_d_idx" in train_only_artifacts["talent_lookup"] else {}
    s_idx = train_only_artifacts["talent_lookup"]["_s_idx"] if "_s_idx" in train_only_artifacts["talent_lookup"] else {}
    c_idx = train_only_artifacts["talent_lookup"]["_c_idx"] if "_c_idx" in train_only_artifacts["talent_lookup"] else {}
    
    collision_indices = set()
    for c_i, r_i in test_prep.iterrows():
        c_dt = r_i["eff_cur_date"] - pd.Timedelta(days=365)
        for p in r_i["clean_dirs"] or []:
            for x in d_idx.get(p, []):
                if x[0] <= c_dt and x[2] == c_i:
                    collision_indices.add(c_i)
        for p in r_i["clean_scrs"] or []:
            for x in s_idx.get(p, []):
                if x[0] <= c_dt and x[2] == c_i:
                    collision_indices.add(c_i)
        for p in r_i["clean_cast"] or []:
            for x in c_idx.get(p, []):
                if x[0] <= c_dt and x[2] == c_i:
                    collision_indices.add(c_i)

    clean_pool = test_df.index.difference(collision_indices)
    sample_indices = test_df.loc[clean_pool].sample(n=20, random_state=42).index

    for idx in sample_indices:
        t_name = test_df.loc[idx, "name"]
        raw_matches = labeled_df[labeled_df["name"] == t_name]
        assert len(raw_matches) > 0, f"Raw drama not found for title '{t_name}'"
        raw_row = raw_matches.iloc[0]

        # Parse genres whether separated by pipe or comma
        g_raw = raw_row["genres"]
        if pd.notna(g_raw):
            g_list = [g.strip() for g in str(g_raw).replace("|", ",").split(",") if g.strip()]
        else:
            g_list = []

        inputs = {
            "country": raw_row["country"],
            "network": raw_row["original_network"],
            "content_rating": raw_row["content_rating"],
            "genres": g_list,
            "weekdays": raw_row["aired_on"],
            "episodes": raw_row["episodes"],
            "director": raw_row["director"],
            "screenwriter": raw_row["screenwriter"],
            "cast": raw_row["cast_summary"],
            "planned_start_date": raw_row["start_date"],
        }

        df_built, _ = build_feature_row(inputs, artifacts)
        expected_vec = X_test_expected.loc[idx].values
        built_vec = df_built.iloc[0].values

        assert np.allclose(built_vec, expected_vec, equal_nan=True, atol=1e-5), (
            f"Feature mismatch on drama '{t_name}' (idx={idx})"
        )


def test_no_prior_used_starts_after_planned_start_minus_365_days(artifacts):
    """(g) 12-month rule: prior starting within 365 days of planned_start_date is excluded."""
    # Synthetic lookup with one credit on 2022-01-01 and one on 2023-08-01
    synth_artifacts = artifacts.copy()
    synth_lookup = {
        "director": {},
        "screenwriter": {},
        "cast": {
            "test actor": [
                (pd.to_datetime("2022-01-01"), 0.8),
                (pd.to_datetime("2023-08-01"), 0.2),  # Less than 365 days before 2024-01-01
            ]
        },
        "global_mean": 0.5022,
        "smoothing_m": 3.0,
    }
    synth_artifacts["talent_lookup"] = synth_lookup
    
    inputs = {
        "country": "South Korea",
        "cast": ["test actor"],
        "planned_start_date": "2024-01-01",
    }
    df_built, _ = build_feature_row(inputs, synth_artifacts)
    
    # 2023-08-01 drama must be excluded. Only 2022-01-01 drama qualifies (n_p=1, raw=0.8)
    expected_smoothed = (1 * 0.8 + 3.0 * 0.5022) / (1 + 3.0)
    actual_smoothed = df_built["cast_prior_reach_mean"].iloc[0]
    assert np.isclose(expected_smoothed, actual_smoothed), (
        f"12-month rule failed: expected {expected_smoothed}, got {actual_smoothed}"
    )


def test_saved_model_reproduces_test_metrics(artifacts):
    """
    (8) Saved model reproduces headline test metrics:
    Re-scores test_df (N=2,525) with headline model & isotonic calibrator in models/
    and confirms ROC-AUC, raw Brier, and calibrated Brier match README headline values.
    """
    from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score

    train_df, test_df = load_modeling_data("v2")
    assert len(train_df) == 5010, f"Expected 5,010 train rows, found {len(train_df)}"
    assert len(test_df) == 2525, f"Expected 2,525 test rows, found {len(test_df)}"

    clf = artifacts["headline_model"]
    iso = artifacts["isotonic_calibrator"]
    meta = artifacts["metadata"]

    features = meta["features"]
    cat_cols = meta["cat_cols"]
    enc = meta["ordinal_encoder"]

    X_test = test_df[features].copy()
    X_test[cat_cols] = enc.transform(X_test[cat_cols].astype(str))
    X_test_num = X_test.astype(np.float64)

    assert list(X_test_num.columns) == list(features), "Feature column order mismatch"
    high_idx = meta["high_class_idx"]
    raw_p = clf.predict_proba(X_test_num)[:, high_idx]
    cal_p = iso.predict(raw_p)
    preds = clf.predict(X_test_num)

    y_bin = (test_df["target_tier"].values == "High").astype(int)
    y_tier = test_df["target_tier"].values

    roc = roc_auc_score(y_bin, raw_p)
    brier_raw = brier_score_loss(y_bin, raw_p)
    brier_cal = brier_score_loss(y_bin, cal_p)
    f1 = f1_score(y_tier, preds, average="macro")

    assert np.isclose(roc, 0.8627, atol=0.002), f"ROC-AUC mismatch: {roc}"
    assert np.isclose(brier_raw, 0.1309, atol=0.0005), f"Raw Brier mismatch: {brier_raw}"
    assert np.isclose(brier_cal, 0.1273, atol=0.0005), f"Calibrated Brier mismatch: {brier_cal}"
    assert np.isclose(f1, 0.6557, atol=0.005), f"Macro-F1 mismatch: {f1}"


def test_likelihood_band_boundary_thresholds():
    """Verify band_for(p) strictly follows 0.20, 0.50, 0.75 thresholds at boundaries."""
    from src.explain import band_for
    boundary_cases = [
        (0.0, "Unlikely"),
        (0.199, "Unlikely"),
        (0.20, "Possible"),
        (0.499, "Possible"),
        (0.50, "Likely"),
        (0.749, "Likely"),
        (0.75, "Very likely"),
        (1.0, "Very likely"),
    ]
    for p, expected in boundary_cases:
        assert band_for(p) == expected, f"band_for({p}) returned {band_for(p)}, expected {expected}"


