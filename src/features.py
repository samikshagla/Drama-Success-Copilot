"""
Feature Engineering Module for Drama Success Copilot (Milestone 3)
-----------------------------------------------------------------
Leakage-Free Feature Engineering Principles:
1. Split First: Train (release_year <= 2021) and Test (release_year >= 2022).
2. Fit on Train Only: All vocabularies (genres, networks), scalers, frequency encodings,
   and imputation rules are fitted strictly on the training set and applied to test.
3. Strict Expanding-Window Priors: Any historical feature (actor track record, director rating)
   must only aggregate titles released at least 12 months prior to the drama's premiere date.
"""

from typing import Tuple, Dict, Any, Set, List, Optional
from collections import defaultdict
import unicodedata
import re
import pandas as pd
import numpy as np


def load_labeled_data(filepath: str = "data/processed/dramas_labeled.csv") -> pd.DataFrame:
    """Load the scoped, labeled dataset produced in Step 2E."""
    df = pd.read_csv(filepath)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    return df


def temporal_train_test_split(
    df: pd.DataFrame, cutoff_year: int = 2021
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset strictly along time boundary into train (<= cutoff_year)
    and test (> cutoff_year).
    """
    train_df = df[df["release_year"] <= cutoff_year].copy().reset_index(drop=True)
    test_df = df[df["release_year"] > cutoff_year].copy().reset_index(drop=True)
    
    assert train_df["release_year"].max() <= cutoff_year, "Train set contains post-cutoff data!"
    assert test_df["release_year"].min() > cutoff_year, "Test set contains pre-cutoff data!"
    assert len(train_df) + len(test_df) == len(df), "Row count mismatch during temporal split!"
    
    return train_df, test_df


def fit_basic_features(train_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Fit vocabularies and mappings strictly on the training set.
    
    Rules:
    - Networks: Keep networks with >= 10 train titles (excluding 'Unknown').
    - Genres: Keep atomic genres appearing in >= 30 train titles.
    - Weekdays: Standard 7-day calendar.
    - Content Rating: Map legitimate target ratings; 'Not Yet Rated' -> missing.
    """
    # 1. Networks with >= 10 train dramas (excluding Unknown)
    net_counts = train_df["original_network"].value_counts()
    valid_networks = set(
        net_counts[(net_counts >= 10) & (net_counts.index != "Unknown")].index
    )
    
    # 2. Atomic genres with >= 30 train dramas
    all_genres = [
        g.strip()
        for sub in train_df["genres"].dropna()
        if sub != "Unknown"
        for g in sub.split("|")
        if g.strip()
    ]
    g_counts = pd.Series(all_genres).value_counts()
    valid_genres = sorted(list(g_counts[g_counts >= 30].index))
    
    # 3. Standard weekdays
    weekdays = [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ]
    
    # 4. Content rating mapping
    real_ratings = {
        "15+ - Teens 15 or older": "15+",
        "13+ - Teens 13 or older": "13+",
        "G - All Ages": "G",
        "18+ Restricted (violence & profanity)": "18+",
        "R - Restricted Screening (nudity & violence)": "18+"
    }
    
    return {
        "valid_networks": valid_networks,
        "valid_genres": valid_genres,
        "weekdays": weekdays,
        "rating_map": real_ratings,
        "countries": ["China", "Japan", "South Korea", "Thailand"]
    }


def transform_basic_features(
    df: pd.DataFrame, params: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transform raw attributes into Core and Missingness feature groups
    using parameters fitted on the training set.
    
    Returns:
        core_df: Features representing valid pitch parameters (Unknowns treated as missing).
        miss_df: Binary indicators capturing page-completeness signals (ablation group).
    """
    core_df = pd.DataFrame(index=df.index)
    miss_df = pd.DataFrame(index=df.index)
    
    # --- 1. Scope / Numeric ---
    core_df["episodes"] = df["episodes"].astype(float)
    
    # --- 2. Country (Categorical) ---
    # Single clean string categorical representation; one-hot encoding deferred to tabular preprocessor
    core_df["country"] = df["country"].astype(str)
    
    # --- 3. Aired On Weekdays ---
    # In core group: 1.0 if weekday matches, 0.0 if not. Unknown maps strictly to NaN.
    for w in params["weekdays"]:
        col_w = f"aired_on_{w.lower()}"
        core_df[col_w] = df["aired_on"].apply(
            lambda x: np.nan
            if (pd.isna(x) or x == "Unknown")
            else (1.0 if w in [d.strip() for d in str(x).split(",")] else 0.0)
        )
    miss_df["aired_on_is_missing"] = (
        (df["aired_on"] == "Unknown") | df["aired_on"].isna()
    ).astype(int)
    
    # --- 4. Content Rating ---
    # Map real ratings; placeholders ('Not Yet Rated' or null) map to NaN in core
    core_df["content_rating"] = df["content_rating"].map(params["rating_map"])
    miss_df["content_rating_is_missing"] = (
        (df["content_rating"] == "Not Yet Rated") | df["content_rating"].isna()
    ).astype(int)
    
    # --- 5. Original Network ---
    # In core group: Keep network name if in train vocabulary, else 'Other'; 'Unknown' -> NaN
    def clean_network(n):
        if pd.isna(n) or n == "Unknown":
            return np.nan
        return n if n in params["valid_networks"] else "Other"
        
    core_df["original_network"] = df["original_network"].apply(clean_network)
    miss_df["network_is_missing"] = (
        (df["original_network"] == "Unknown") | df["original_network"].isna()
    ).astype(int)
    
    # --- 6. Genres (Multi-Hot Indicators) ---
    # In core group: 1.0 if genre present, 0.0 if absent. Unknown maps strictly to NaN.
    for g in params["valid_genres"]:
        col_g = "genre_" + g.lower().replace("-", "_").replace(" ", "_")
        core_df[col_g] = df["genres"].apply(
            lambda x: np.nan
            if (pd.isna(x) or x == "Unknown")
            else (1.0 if g in [item.strip() for item in str(x).split("|")] else 0.0)
        )
    miss_df["genre_is_missing"] = (
        (df["genres"] == "Unknown") | df["genres"].isna()
    ).astype(int)
    
    return core_df, miss_df


def normalize_talent_name(s: str) -> str:
    """
    Standardize person names: lowercase, collapse whitespace, strip parentheses, remove accents.
    """
    if not s or pd.isna(s) or s == "Unknown":
        return ""
    s = re.sub(r"\(.*?\)", "", str(s))
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("utf-8")
    s = s.lower()
    return re.sub(r"\s+", " ", s).strip()


def parse_talent_crew(val: Any) -> Optional[List[str]]:
    """Parse comma-separated director or screenwriter strings. Return None if Unknown or empty."""
    if pd.isna(val) or val == "Unknown":
        return None
    names = [normalize_talent_name(p) for p in str(val).split(",")]
    names = [n for n in names if n]
    return names if names else None


def parse_talent_cast(val: Any) -> Optional[List[str]]:
    """
    Parse semicolon-separated cast strings, keeping [Main Role] entries only.
    Return None if Unknown, empty, or if no [Main Role] entries exist.
    """
    if pd.isna(val) or val == "Unknown":
        return None
    names = []
    for p in str(val).split(";"):
        if "[Main Role]" in p:
            c = re.sub(r"\[.*?\]", "", p)
            n = normalize_talent_name(c)
            if n:
                names.append(n)
    return names if names else None


def prepare_talent_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add effective release dates and parsed talent lists to a dataframe.
    - eff_cur_date: For the current drama being evaluated, if start_date is null, use Jan 1 of release_year.
    - eff_pri_date: For a prior candidate drama, if start_date is null, use Dec 31 of release_year.
    """
    df = df.copy()
    start_dt = pd.to_datetime(df["start_date"])
    df["eff_cur_date"] = [
        pd.to_datetime(f"{r.release_year}-01-01") if pd.isna(s) else s
        for s, r in zip(start_dt, df.itertuples())
    ]
    df["eff_pri_date"] = [
        pd.to_datetime(f"{r.release_year}-12-31") if pd.isna(s) else s
        for s, r in zip(start_dt, df.itertuples())
    ]
    df["clean_dirs"] = df["director"].apply(parse_talent_crew)
    df["clean_scrs"] = df["screenwriter"].apply(parse_talent_crew)
    df["clean_cast"] = df["cast_summary"].apply(parse_talent_cast)
    return df


def build_talent_inverted_index(pool_df: pd.DataFrame) -> Tuple[Dict[str, list], Dict[str, list], Dict[str, list]]:
    """
    Build lookup indices mapping normalized person name -> [(eff_pri_date, target_pct, drama_idx, drama_name), ...]
    Only labeled, scoped dramas in pool_df are indexed.
    """
    dir_idx = defaultdict(list)
    scr_idx = defaultdict(list)
    cast_idx = defaultdict(list)
    
    for idx, row in pool_df.iterrows():
        pri_date = row["eff_pri_date"]
        target = row["target_pct"]
        d_name = row["name"]
        
        if row["clean_dirs"]:
            for person in row["clean_dirs"]:
                dir_idx[person].append((pri_date, target, idx, d_name))
        if row["clean_scrs"]:
            for person in row["clean_scrs"]:
                scr_idx[person].append((pri_date, target, idx, d_name))
        if row["clean_cast"]:
            for person in row["clean_cast"]:
                cast_idx[person].append((pri_date, target, idx, d_name))
                
    return dir_idx, scr_idx, cast_idx


def compute_talent_priors(
    target_df: pd.DataFrame,
    dir_idx: Dict[str, list],
    scr_idx: Dict[str, list],
    cast_idx: Dict[str, list],
    m: float = 3.0,
    global_mean: float = 0.5022
) -> pd.DataFrame:
    """
    Compute 10 expanding-window talent track-record features for target_df:
    - 12-month rule: eff_pri_date <= eff_cur_date - 365 days
    - No self-priors: prior drama_idx != target drama_idx
    - Unknown credit: all features for that role are NaN (including n_prior)
    - Debut talent (known credits but 0 prior qualifying titles): n_prior = 0, reach = NaN
    - Empirical Bayes smoothing: (n * mean_prior + m * global_mean) / (n + m)
    """
    rows = []
    leakage_failures = 0
    
    for idx, row in target_df.iterrows():
        cur_date = row["eff_cur_date"]
        cutoff = cur_date - pd.Timedelta(days=365)
        
        # 1. Director Track Record
        if row["clean_dirs"] is None:
            dir_n_max = np.nan
            dir_n_5yr_max = np.nan
            dir_r_max = np.nan
            dir_r_mean = np.nan
        else:
            n_list = []
            n_5yr_list = []
            reach_list = []
            cutoff_5yr = cutoff - pd.Timedelta(days=5 * 365)
            for p in row["clean_dirs"]:
                priors = [x for x in dir_idx.get(p, []) if x[0] <= cutoff and x[2] != idx]
                for x in priors:
                    if x[0] > cutoff or x[2] == idx:
                        leakage_failures += 1
                n_p = len(priors)
                n_p_5yr = len([x for x in priors if x[0] > cutoff_5yr])
                n_list.append(n_p)
                n_5yr_list.append(n_p_5yr)
                if n_p > 0:
                    raw_mean = np.mean([x[1] for x in priors])
                    reach_list.append((n_p * raw_mean + m * global_mean) / (n_p + m))
            dir_n_max = float(max(n_list)) if n_list else 0.0
            dir_n_5yr_max = float(max(n_5yr_list)) if n_5yr_list else 0.0
            dir_r_max = float(max(reach_list)) if reach_list else np.nan
            dir_r_mean = float(np.mean(reach_list)) if reach_list else np.nan
            
        # 2. Screenwriter Track Record
        if row["clean_scrs"] is None:
            scr_n_max = np.nan
            scr_n_5yr_max = np.nan
            scr_r_max = np.nan
            scr_r_mean = np.nan
        else:
            n_list = []
            n_5yr_list = []
            reach_list = []
            cutoff_5yr = cutoff - pd.Timedelta(days=5 * 365)
            for p in row["clean_scrs"]:
                priors = [x for x in scr_idx.get(p, []) if x[0] <= cutoff and x[2] != idx]
                for x in priors:
                    if x[0] > cutoff or x[2] == idx:
                        leakage_failures += 1
                n_p = len(priors)
                n_p_5yr = len([x for x in priors if x[0] > cutoff_5yr])
                n_list.append(n_p)
                n_5yr_list.append(n_p_5yr)
                if n_p > 0:
                    raw_mean = np.mean([x[1] for x in priors])
                    reach_list.append((n_p * raw_mean + m * global_mean) / (n_p + m))
            scr_n_max = float(max(n_list)) if n_list else 0.0
            scr_n_5yr_max = float(max(n_5yr_list)) if n_5yr_list else 0.0
            scr_r_max = float(max(reach_list)) if reach_list else np.nan
            scr_r_mean = float(np.mean(reach_list)) if reach_list else np.nan
            
        # 3. Lead Cast [Main Role] Track Record
        if row["clean_cast"] is None:
            cast_n_max = np.nan
            cast_n_5yr_max = np.nan
            cast_n_mean = np.nan
            cast_n_5yr_mean = np.nan
            cast_r_max = np.nan
            cast_r_mean = np.nan
        else:
            n_list = []
            n_5yr_list = []
            reach_list = []
            cutoff_5yr = cutoff - pd.Timedelta(days=5 * 365)
            for p in row["clean_cast"]:
                priors = [x for x in cast_idx.get(p, []) if x[0] <= cutoff and x[2] != idx]
                for x in priors:
                    if x[0] > cutoff or x[2] == idx:
                        leakage_failures += 1
                n_p = len(priors)
                n_p_5yr = len([x for x in priors if x[0] > cutoff_5yr])
                n_list.append(n_p)
                n_5yr_list.append(n_p_5yr)
                if n_p > 0:
                    raw_mean = np.mean([x[1] for x in priors])
                    reach_list.append((n_p * raw_mean + m * global_mean) / (n_p + m))
            cast_n_max = float(max(n_list)) if n_list else 0.0
            cast_n_5yr_max = float(max(n_5yr_list)) if n_5yr_list else 0.0
            cast_n_mean = float(np.mean(n_list)) if n_list else 0.0
            cast_n_5yr_mean = float(np.mean(n_5yr_list)) if n_5yr_list else 0.0
            cast_r_max = float(max(reach_list)) if reach_list else np.nan
            cast_r_mean = float(np.mean(reach_list)) if reach_list else np.nan
            
        rows.append({
            "director_n_prior_max": dir_n_max,
            "director_n_prior_5yr_max": dir_n_5yr_max,
            "director_prior_reach_max": dir_r_max,
            "director_prior_reach_mean": dir_r_mean,
            "screenwriter_n_prior_max": scr_n_max,
            "screenwriter_n_prior_5yr_max": scr_n_5yr_max,
            "screenwriter_prior_reach_max": scr_r_max,
            "screenwriter_prior_reach_mean": scr_r_mean,
            "cast_n_prior_max": cast_n_max,
            "cast_n_prior_5yr_max": cast_n_5yr_max,
            "cast_n_prior_mean": cast_n_mean,
            "cast_n_prior_5yr_mean": cast_n_5yr_mean,
            "cast_prior_reach_max": cast_r_max,
            "cast_prior_reach_mean": cast_r_mean,
        })
        
    assert leakage_failures == 0, f"Future leakage detected: {leakage_failures} invalid priors accessed!"
    return pd.DataFrame(rows, index=target_df.index)


# =====================================================================
# Feature Groups Configuration (Switchable Feature Subsets)
# =====================================================================

FEATURE_GROUPS = {
    "core": [
        "episodes",
        "country",
        "original_network",
        "content_rating",
        "aired_on_monday",
        "aired_on_tuesday",
        "aired_on_wednesday",
        "aired_on_thursday",
        "aired_on_friday",
        "aired_on_saturday",
        "aired_on_sunday",
        "genre_action",
        "genre_adventure",
        "genre_business",
        "genre_comedy",
        "genre_crime",
        "genre_drama",
        "genre_family",
        "genre_fantasy",
        "genre_food",
        "genre_historical",
        "genre_horror",
        "genre_law",
        "genre_life",
        "genre_medical",
        "genre_melodrama",
        "genre_military",
        "genre_music",
        "genre_mystery",
        "genre_political",
        "genre_psychological",
        "genre_romance",
        "genre_sci_fi",
        "genre_sitcom",
        "genre_sports",
        "genre_supernatural",
        "genre_thriller",
        "genre_tokusatsu",
        "genre_war",
        "genre_wuxia",
        "genre_youth",
    ],
    "talent_reach": [
        "director_prior_reach_max",
        "director_prior_reach_mean",
        "screenwriter_prior_reach_max",
        "screenwriter_prior_reach_mean",
        "cast_prior_reach_max",
        "cast_prior_reach_mean",
    ],
    "talent_experience": [
        "director_n_prior_max",
        "director_n_prior_5yr_max",
        "screenwriter_n_prior_max",
        "screenwriter_n_prior_5yr_max",
        "cast_n_prior_max",
        "cast_n_prior_5yr_max",
        "cast_n_prior_mean",
        "cast_n_prior_5yr_mean",
    ],
    "missingness": [
        "aired_on_is_missing",
        "content_rating_is_missing",
        "network_is_missing",
        "genre_is_missing",
    ],
}

# Default active feature groups for modeling:
# talent_experience is OFF by default because n_prior tracks premiere year (confounded with corpus start).
# missingness is OFF by default (reserved for page-completeness bias ablation benchmarking).
DEFAULT_FEATURE_GROUPS = ["core", "talent_reach"]


def get_feature_columns(
    groups: Optional[List[str]] = None,
    include_experience: bool = False,
    include_missingness: bool = False,
) -> List[str]:
    """
    Return list of active feature column names according to enabled groups.
    
    Parameters:
        groups: Explicit list of group names to include (e.g. ['core', 'talent_reach']).
                If None, uses DEFAULT_FEATURE_GROUPS modified by convenience flags.
        include_experience: If True and groups is None, adds 'talent_experience' group.
        include_missingness: If True and groups is None, adds 'missingness' group.
    
    Returns:
        List of feature column names.
    """
    if groups is None:
        active_groups = list(DEFAULT_FEATURE_GROUPS)
        if include_experience and "talent_experience" not in active_groups:
            active_groups.append("talent_experience")
        if include_missingness and "missingness" not in active_groups:
            active_groups.append("missingness")
    else:
        active_groups = groups

    cols: List[str] = []
    for g in active_groups:
        if g not in FEATURE_GROUPS:
            raise ValueError(f"Unknown feature group: '{g}'. Valid groups: {list(FEATURE_GROUPS.keys())}")
        cols.extend(FEATURE_GROUPS[g])
        
    return cols



