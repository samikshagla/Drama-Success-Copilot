"""
Explainability Module for Drama Success Copilot (Milestone 5)
--------------------------------------------------------------
Provides:
1. `explain_drama(inputs, artifacts=None)`:
   - Builds 47-feature vector from plain dictionary inputs.
   - Computes Bayesian smoothed talent priors using the 12-month rule.
   - Scores predicted tier, raw P(High), calibrated P(High), and likelihood band.
   - Generates net group SHAP attributions via PermutationExplainer (max_evals=1000).
   - Generates top positive/negative drivers, actionable warnings, and standing caveats.
2. `build_feature_row(inputs, artifacts)`: Standalone feature builder reproducing training data.
3. `load_artifacts(models_dir="models")`: Loads serialized model artifacts.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import unicodedata
import re
import numpy as np
import pandas as pd
import joblib
import shap

_CACHED_ARTIFACTS: Optional[Dict[str, Any]] = None

# 9 standard feature groups
GROUP_DEFS = {
    "talent_cast": ["cast_prior_reach_max", "cast_prior_reach_mean"],
    "network": ["original_network"],
    "genres": [
        "genre_action", "genre_adventure", "genre_business", "genre_comedy",
        "genre_crime", "genre_drama", "genre_family", "genre_fantasy",
        "genre_food", "genre_historical", "genre_horror", "genre_law",
        "genre_life", "genre_medical", "genre_melodrama", "genre_military",
        "genre_music", "genre_mystery", "genre_political", "genre_psychological",
        "genre_romance", "genre_sci_fi", "genre_sitcom", "genre_sports",
        "genre_supernatural", "genre_thriller", "genre_tokusatsu", "genre_war",
        "genre_wuxia", "genre_youth"
    ],
    "talent_director": ["director_prior_reach_max", "director_prior_reach_mean"],
    "talent_screenwriter": ["screenwriter_prior_reach_max", "screenwriter_prior_reach_mean"],
    "content_rating": ["content_rating"],
    "country": ["country"],
    "weekdays": [
        "aired_on_monday", "aired_on_tuesday", "aired_on_wednesday",
        "aired_on_thursday", "aired_on_friday", "aired_on_saturday",
        "aired_on_sunday"
    ],
    "episodes": ["episodes"],
}


# Observed High rate of the likelihood band on test dramas (released 2022-2025, N=2,525)
TEST_BAND_OBSERVED_RATES = {
    "Unlikely": "7.0% [95% CI: 5.8%, 8.4%] (N=1,436; observed on test dramas released 2022-2025, relative to dramas of the same country and year)",
    "Possible": "31.7% [95% CI: 27.9%, 35.8%] (N=527; observed on test dramas released 2022-2025, relative to dramas of the same country and year)",
    "Likely": "57.8% [95% CI: 52.6%, 62.9%] (N=351; observed on test dramas released 2022-2025, relative to dramas of the same country and year)",
    "Very likely": "79.6% [95% CI: 73.7%, 84.5%] (N=211; observed on test dramas released 2022-2025, relative to dramas of the same country and year)",
}


def band_for(cal_p_high: float, tol: float = 1e-9) -> str:
    """
    Return likelihood band for a calibrated P(High) probability using standard 0.20, 0.50, 0.75 cutoffs.
    Robust to floating-point noise with tolerance 1e-9; values within tol below an edge or equal/above
    belong to the upper band:
    - [0.00, 0.20): Unlikely
    - [0.20, 0.50): Possible
    - [0.50, 0.75): Likely
    - [0.75, 1.00]: Very likely
    """
    if cal_p_high < 0.20 - tol:
        return "Unlikely"
    elif cal_p_high < 0.50 - tol:
        return "Possible"
    elif cal_p_high < 0.75 - tol:
        return "Likely"
    else:
        return "Very likely"


def load_artifacts(models_dir: Union[str, Path] = None) -> Dict[str, Any]:
    """Load serialized model artifacts from models directory, anchored at repo root."""
    global _CACHED_ARTIFACTS
    if _CACHED_ARTIFACTS is not None:
        return _CACHED_ARTIFACTS

    repo_root = Path(__file__).resolve().parent.parent
    if models_dir is None:
        p = repo_root / "models"
    else:
        p = Path(models_dir)
        if not p.is_absolute() and not p.exists():
            if (repo_root / p).exists():
                p = repo_root / p

    clf = joblib.load(p / "headline_model.joblib")
    iso_cal = joblib.load(p / "isotonic_calibrator.joblib")
    bg = joblib.load(p / "background_sample.joblib")
    talent = joblib.load(p / "talent_lookup.joblib")
    meta = joblib.load(p / "metadata.joblib")

    _CACHED_ARTIFACTS = {
        "headline_model": clf,
        "isotonic_calibrator": iso_cal,
        "background_sample": bg,
        "talent_lookup": talent,
        "metadata": meta,
    }
    return _CACHED_ARTIFACTS


def normalize_person_name(s: Any) -> str:
    """Standardize person names: lowercase, collapse whitespace, strip parentheses, remove accents."""
    if not s or pd.isna(s) or str(s).strip() in ["", "Unknown"]:
        return ""
    s = re.sub(r"\(.*?\)", "", str(s))
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("utf-8")
    s = s.lower()
    return re.sub(r"\s+", " ", s).strip()


def parse_person_names(val: Any) -> List[str]:
    """Parse string or list of names into a clean list of normalized names."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, (list, tuple)):
        raw_list = val
    elif isinstance(val, str):
        if "[Main Role]" in val:
            raw_list = [p for p in val.split(";") if "[Main Role]" in p]
        elif ";" in val:
            raw_list = val.split(";")
        else:
            raw_list = val.split(",")
    else:
        raw_list = [str(val)]

    cleaned = []
    for item in raw_list:
        c = re.sub(r"\[.*?\]", "", str(item))
        n = normalize_person_name(c)
        if n and n != "unknown":
            cleaned.append(n)
    return cleaned


def compute_talent_reach_from_lookup(
    names: List[str],
    role_lookup: Dict[str, List[Tuple[pd.Timestamp, float]]],
    cutoff_date: pd.Timestamp,
    m: float = 3.0,
    global_mean: float = 0.5022
) -> Tuple[Optional[float], Optional[float], int, List[str]]:
    """
    Compute Bayesian smoothed reach (max and mean) for a given creator list.
    Returns: (reach_max, reach_mean, n_qualifying_titles, missing_history_names)
    """
    if not names:
        return None, None, 0, []

    reach_values = []
    no_history = []
    total_titles = 0

    for person in names:
        history = role_lookup.get(person, [])
        qualifying = [entry for entry in history if entry[0] <= cutoff_date]
        n_p = len(qualifying)
        total_titles += n_p

        if n_p == 0:
            no_history.append(person)
        else:
            raw_mean = float(np.mean([entry[1] for entry in qualifying]))
            smoothed = (n_p * raw_mean + m * global_mean) / (n_p + m)
            reach_values.append(smoothed)

    if reach_values:
        return float(max(reach_values)), float(np.mean(reach_values)), total_titles, no_history
    else:
        return None, None, total_titles, no_history


def build_feature_row(inputs: Dict[str, Any], artifacts: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the exact 47-feature numeric row from plain inputs.
    Returns: (feature_df, warnings_list)
    """
    meta = artifacts["metadata"]
    features = meta["features"]
    talent_lookup = artifacts["talent_lookup"]
    warnings: List[str] = []

    # 1. Planned start date & 12-month rule cutoff
    raw_date = inputs.get("planned_start_date")
    if raw_date is not None and not pd.isna(raw_date):
        planned_dt = pd.to_datetime(raw_date)
    else:
        year = inputs.get("release_year") or inputs.get("year") or 2024
        planned_dt = pd.to_datetime(f"{year}-01-01")
    cutoff_dt = planned_dt - pd.Timedelta(days=365)

    row: Dict[str, Any] = {}

    # 2. Episodes
    ep = inputs.get("episodes")
    ep_min, ep_max = meta.get("train_episode_range", (1, 150))
    if ep is None or pd.isna(ep):
        row["episodes"] = np.nan
        warnings.append("Episode count is blank; filling it in provides a more reliable forecast.")
    else:
        ep_val = float(ep)
        row["episodes"] = ep_val
        if ep_val < ep_min or ep_val > ep_max:
            warnings.append(
                f"Episode count ({int(ep_val)}) is outside the training range ({ep_min}-{ep_max}); predictions may be less reliable."
            )

    # 3. Country (Categorical)
    c_raw = inputs.get("country", "")
    c_str = str(c_raw).strip() if c_raw and not pd.isna(c_raw) else "Unknown"
    row["country"] = c_str
    if c_str not in meta.get("valid_countries", []):
        warnings.append(f"Country '{c_str}' is outside standard training markets ({', '.join(meta.get('valid_countries', []))}).")

    # 4. Broadcaster Network (Categorical)
    net_raw = inputs.get("network") or inputs.get("original_network")
    if net_raw is None or pd.isna(net_raw) or str(net_raw).strip() in ["", "Unknown"]:
        row["original_network"] = np.nan
        warnings.append("Network is blank; encoded as missing. The model predicts lower P(High) when this field is blank than when it holds a typical value; fill it in with a typical broadcaster for an accurate forecast.")
    else:
        net_str = str(net_raw).strip()
        if net_str in meta["valid_networks"]:
            row["original_network"] = net_str
        else:
            row["original_network"] = "Other"

    # 5. Content Rating (Categorical)
    cr_raw = inputs.get("content_rating")
    if cr_raw is None or pd.isna(cr_raw) or str(cr_raw).strip() in ["", "Unknown", "Not Yet Rated"]:
        row["content_rating"] = np.nan
        warnings.append("Content rating is blank. The model predicts lower P(High) when this field is blank than when it holds a typical value; fill it in if known.")
    else:
        cr_str = str(cr_raw).strip()
        mapped_cr = meta["rating_map"].get(cr_str, cr_str)
        if mapped_cr in ["G", "13+", "15+", "18+"]:
            row["content_rating"] = mapped_cr
        else:
            row["content_rating"] = np.nan
            warnings.append(f"Content rating '{cr_str}' is unrecognized; treated as missing (NaN).")

    # 6. Weekdays (7 columns)
    wk_raw = inputs.get("weekdays") or inputs.get("aired_on")
    if (
        wk_raw is None
        or (isinstance(wk_raw, float) and np.isnan(wk_raw))
        or str(wk_raw).strip() in ["", "Unknown", "nan"]
        or (
            isinstance(wk_raw, (list, tuple))
            and (len(wk_raw) == 0 or all(str(x).strip() in ["", "Unknown", "nan"] for x in wk_raw))
        )
    ):
        for w in meta["weekdays"]:
            row[f"aired_on_{w.lower()}"] = np.nan
        warnings.append("Broadcast weekday schedule is blank or empty. The model predicts lower P(High) when this field is blank than when it holds a typical value; fill it in.")
    else:
        if isinstance(wk_raw, str):
            wk_list = [d.strip() for d in wk_raw.split(",")]
        else:
            wk_list = [str(d).strip() for d in wk_raw]
        for w in meta["weekdays"]:
            row[f"aired_on_{w.lower()}"] = 1.0 if any(w.lower() == d.lower() for d in wk_list) else 0.0

    # 7. Genres (30 columns)
    g_raw = inputs.get("genres")
    if (
        g_raw is None
        or (isinstance(g_raw, float) and np.isnan(g_raw))
        or str(g_raw).strip() in ["", "Unknown", "nan"]
        or (
            isinstance(g_raw, (list, tuple))
            and (len(g_raw) == 0 or all(str(x).strip() in ["", "Unknown", "nan"] for x in g_raw))
        )
    ):
        for g in meta["valid_genres"]:
            col_g = "genre_" + g.lower().replace("-", "_").replace(" ", "_")
            row[col_g] = np.nan
        warnings.append("Genres list is blank or empty. The model predicts lower P(High) when this field is blank than when it holds a typical value; fill it in.")
    else:
        if isinstance(g_raw, str):
            g_list = [item.strip() for item in g_raw.replace("|", ",").split(",")]
        else:
            g_list = [str(item).strip() for item in g_raw]
        for g in meta["valid_genres"]:
            col_g = "genre_" + g.lower().replace("-", "_").replace(" ", "_")
            row[col_g] = 1.0 if any(g.lower() == item.lower() for item in g_list) else 0.0

    # 8. Talent Track Record Priors (6 columns)
    dirs = parse_person_names(inputs.get("director") or inputs.get("directors"))
    scrs = parse_person_names(inputs.get("screenwriter") or inputs.get("screenwriters") or inputs.get("writers") or inputs.get("writer"))
    cast = parse_person_names(inputs.get("cast") or inputs.get("lead_cast") or inputs.get("actors"))

    d_max, d_mean, _, d_no_hist = compute_talent_reach_from_lookup(
        dirs, talent_lookup["director"], cutoff_dt,
        m=meta.get("smoothing_m", 3.0), global_mean=meta.get("global_mean", 0.5022)
    )
    s_max, s_mean, _, s_no_hist = compute_talent_reach_from_lookup(
        scrs, talent_lookup["screenwriter"], cutoff_dt,
        m=meta.get("smoothing_m", 3.0), global_mean=meta.get("global_mean", 0.5022)
    )
    c_max, c_mean, _, c_no_hist = compute_talent_reach_from_lookup(
        cast, talent_lookup["cast"], cutoff_dt,
        m=meta.get("smoothing_m", 3.0), global_mean=meta.get("global_mean", 0.5022)
    )

    row["director_prior_reach_max"] = d_max if d_max is not None else np.nan
    row["director_prior_reach_mean"] = d_mean if d_mean is not None else np.nan
    row["screenwriter_prior_reach_max"] = s_max if s_max is not None else np.nan
    row["screenwriter_prior_reach_mean"] = s_mean if s_mean is not None else np.nan
    row["cast_prior_reach_max"] = c_max if c_max is not None else np.nan
    row["cast_prior_reach_mean"] = c_mean if c_mean is not None else np.nan

    # Talent warnings
    if (dirs and d_no_hist) or (scrs and s_no_hist) or (cast and c_no_hist):
        warnings.append("No qualifying prior credits found within the 12-month rule; feature set to missing. The model cannot distinguish a debut from an unlisted credit; prior reach is treated as unknown (NaN).")

    # Construct single-row DataFrame aligned with features
    df_row = pd.DataFrame([row])[features].copy()

    # Apply OrdinalEncoder to categorical columns
    cat_cols = meta["cat_cols"]
    enc = meta["ordinal_encoder"]
    df_row[cat_cols] = enc.transform(df_row[cat_cols].astype(str))

    assert list(df_row.columns) == list(features), (
        f"Feature order mismatch: {list(df_row.columns)} vs {features}"
    )
    return df_row.astype(np.float64), warnings


def ordinal_suffix(n: int) -> str:
    """Format integer with correct English ordinal suffix (1st, 2nd, 3rd, 4th, 21st, 22nd, etc.)."""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    last = n % 10
    if last == 1:
        return f"{n}st"
    elif last == 2:
        return f"{n}nd"
    elif last == 3:
        return f"{n}rd"
    else:
        return f"{n}th"


def describe_group_reason(group_name: str, signed_val: float, inputs: Dict[str, Any], feature_row: pd.DataFrame) -> str:
    """Generate a concise, human-readable reason for why a feature group pushed probability up or down."""
    direction = "raised" if signed_val > 0 else "reduced"
    impact_str = f"{signed_val:+.3f}"

    if group_name == "talent_cast":
        cast_val = feature_row["cast_prior_reach_mean"].iloc[0]
        if pd.isna(cast_val):
            return f"Lead cast prior reach is missing (NaN), which {direction} predicted High likelihood ({impact_str})."
        ord_str = ordinal_suffix(round(cast_val * 100))
        return f"Lead cast prior track record averaged around the {ord_str} percentile within country-year, which {direction} High likelihood ({impact_str})."

    elif group_name == "network":
        net = inputs.get("network") or inputs.get("original_network") or "Blank / Unknown"
        return f"Broadcaster network '{net}' {direction} predicted High likelihood ({impact_str})."

    elif group_name == "content_rating":
        cr = inputs.get("content_rating") or "Blank / Unknown"
        return f"Content rating '{cr}' {direction} predicted High likelihood ({impact_str})."

    elif group_name == "genres":
        g_list = inputs.get("genres", [])
        g_str = ", ".join(g_list) if isinstance(g_list, list) and g_list else str(g_list)
        return f"Genre combination ({g_str}) {direction} predicted High likelihood ({impact_str})."

    elif group_name == "talent_director":
        d_val = feature_row["director_prior_reach_mean"].iloc[0]
        if pd.isna(d_val):
            return f"Director prior track record is unlisted/missing (NaN) ({impact_str})."
        ord_str = ordinal_suffix(round(d_val * 100))
        return f"Director earlier works averaged around the {ord_str} percentile within country-year ({impact_str})."

    elif group_name == "talent_screenwriter":
        s_val = feature_row["screenwriter_prior_reach_mean"].iloc[0]
        if pd.isna(s_val):
            return f"Screenwriter prior track record is unlisted/missing (NaN) ({impact_str})."
        ord_str = ordinal_suffix(round(s_val * 100))
        return f"Screenwriter earlier works averaged around the {ord_str} percentile within country-year ({impact_str})."

    elif group_name == "country":
        c = inputs.get("country", "Unknown")
        return f"Country cohort ({c}) baseline {direction} predicted High likelihood ({impact_str})."

    elif group_name == "weekdays":
        wk = inputs.get("weekdays") or inputs.get("aired_on") or "Blank"
        return f"Broadcast schedule ({wk}) {direction} predicted High likelihood ({impact_str})."

    elif group_name == "episodes":
        ep = inputs.get("episodes", "Unknown")
        return f"Episode count of {ep} episodes {direction} predicted High likelihood ({impact_str})."

    return f"{group_name} {direction} predicted High likelihood ({impact_str})."


def explain_drama(
    inputs: Dict[str, Any],
    artifacts: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    max_evals: int = 1000
) -> Dict[str, Any]:
    """
    Produce a complete prediction and explainability report for an Asian drama pitch.
    
    Parameters:
        inputs: Dictionary of plain drama attributes (country, network, genres, content_rating,
                weekdays, episodes, director, screenwriter, cast, planned_start_date).
        artifacts: Optional dictionary of pre-loaded artifacts (from load_artifacts).
        seed: Random seed for PermutationExplainer (default: 42).
        max_evals: Evaluation budget for PermutationExplainer (default: 1000).
        
    Returns:
        Compact JSON-serializable dictionary with predictions, likelihood band,
        group attributions, top drivers, warnings, and caveats.
    """
    if artifacts is None:
        artifacts = load_artifacts()

    clf = artifacts["headline_model"]
    iso_cal = artifacts["isotonic_calibrator"]
    bg_sample = artifacts["background_sample"]
    meta = artifacts["metadata"]
    features = meta["features"]
    high_idx = meta["high_class_idx"]

    # 1. Build numeric feature row
    X_num, warnings = build_feature_row(inputs, artifacts)
    assert list(X_num.columns) == list(features), "Feature order mismatch with fit-time features"

    # 2. Model predictions (always pass DataFrame with matching feature names and order)
    raw_p_high = float(clf.predict_proba(X_num)[:, high_idx][0])
    cal_p_raw = float(iso_cal.predict([raw_p_high])[0])
    cal_p_high = float(np.clip(cal_p_raw, 0.02, 0.98))
    pred_tier = str(clf.predict(X_num)[0])
    cal_p_rounded_5pct = float(round(cal_p_high * 20.0) / 20.0)

    # 3. Likelihood band (4 tiers based on calibrated P(High))
    likelihood_band = band_for(cal_p_high)

    # 4. SHAP Permutation Explainer
    def predict_high_proba(arr):
        if isinstance(arr, pd.DataFrame):
            df_arr = arr
        else:
            df_arr = pd.DataFrame(arr, columns=features)
        assert list(df_arr.columns) == list(features), "Feature columns do not match fit-time features"
        return clf.predict_proba(df_arr)[:, high_idx]

    explainer = shap.explainers.PermutationExplainer(predict_high_proba, bg_sample, seed=seed)
    sv = explainer(X_num, max_evals=max_evals)

    base_val = float(sv.base_values[0])
    sv_row = sv.values[0]

    # 5. Net group attributions across 9 groups
    group_signed: Dict[str, float] = {}
    for g, cols in GROUP_DEFS.items():
        col_indices = [features.index(c) for c in cols if c in features]
        group_signed[g] = float(sv_row[col_indices].sum())

    # Sort positive and negative drivers
    pos_groups = sorted(
        [(g, val) for g, val in group_signed.items() if val > 0],
        key=lambda x: x[1], reverse=True
    )
    neg_groups = sorted(
        [(g, val) for g, val in group_signed.items() if val < 0],
        key=lambda x: x[1]
    )

    top_pos_list = [
        {
            "group": g,
            "attribution": round(val, 4),
            "reason": describe_group_reason(g, val, inputs, X_num)
        }
        for g, val in pos_groups[:3]
    ]

    top_neg_list = [
        {
            "group": g,
            "attribution": round(val, 4),
            "reason": describe_group_reason(g, val, inputs, X_num)
        }
        for g, val in neg_groups[:3]
    ]

    # Standing caveats
    caveats = [
        "Predictions and likelihood bands are relative to Asian dramas released in the same country and year cohort, measuring relative audience reach tier.",
        "Feature attributions represent associational model explanations, not causal effects.",
        "SHAP attributions describe the raw model probability output, not the isotonic-calibrated probability.",
        "PermutationExplainer values carry minor Monte Carlo sampling variance across random seeds."
    ]

    return {
        "title": str(inputs.get("title") or inputs.get("name") or "Untitled Pitch"),
        "predicted_tier": pred_tier,
        "raw_p_high": float(raw_p_high),
        "calibrated_p_high": round(cal_p_high, 4),
        "calibrated_p_high_rounded_5pct": round(cal_p_rounded_5pct, 2),
        "likelihood_band": likelihood_band,
        "likelihood_band_definition": "Likelihood to reach the High audience-reach tier relative to dramas of the same country and year",
        "band_observed_rate": TEST_BAND_OBSERVED_RATES[likelihood_band],
        "base_value": float(base_val),
        "group_attributions": {g: float(val) for g, val in sorted(group_signed.items(), key=lambda x: abs(x[1]), reverse=True)},
        "top_positive_groups": top_pos_list,
        "top_negative_groups": top_neg_list,
        "warnings": warnings,
        "caveats": caveats,
    }
