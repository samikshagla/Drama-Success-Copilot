"""
Agent Tools Module for Drama Success Copilot (Milestone 7 Part A)
================================================================
Provides three agent tools with explicit typed schemas and runtime validators:
1. explain_pitch(pitch): Wraps explain_drama for feature attributions and predictions.
2. find_comparables(pitch): Wraps find_similar for semantic comparables with fixed caveat.
3. what_if(pitch, changes): Evaluates what-if attachment levers with associational caveat.

Design Choice: Standard library TypedDict + explicit runtime validators
- Zero external dependencies (pydantic is not installed; no LLM SDKs).
- Native JSON serializability via standard dict structures.
- Strict control character stripping, bounds enforcement, and prompt injection defense.
"""

from __future__ import annotations

import copy
import difflib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

import numpy as np
import pandas as pd

from src.explain import (
    GROUP_DEFS,
    TEST_BAND_OBSERVED_RATES,
    band_for,
    build_feature_row,
    explain_drama,
    load_artifacts as load_explain_artifacts,
)
from src.retriever import (
    find_similar,
    load_retriever_artifacts,
    parse_query_start_date,
)
from src.shared_constants import (
    BAND_BOUNDARY_WARNING,
    COMPARABLES_CAVEAT,
    WHAT_IF_CAVEAT,
)

# Allowed lever fields in what_if
ALLOWED_WHAT_IF_LEVERS = {
    "network",
    "genres",
    "content_rating",
    "directors",
    "writers",
    "lead_cast",
}

DISALLOWED_LEVERS = {
    "episodes",
    "weekdays",
    "aired_on",
    "country",
    "planned_start",
    "start_date",
    "release_year",
    "synopsis",
    "synopsis_text",
}

VALID_COUNTRIES = {"South Korea", "China", "Japan", "Thailand"}

# 35 unique step-function plateaus output by the isotonic calibrator
CALIBRATOR_PLATEAUS = [
    0.0,
    0.01515152,
    0.01923077,
    0.02024291,
    0.02290076,
    0.03191489,
    0.03333333,
    0.04545455,
    0.0952381,
    0.11711712,
    0.125,
    0.12844037,
    0.16666667,
    0.2,
    0.25,
    0.2712766,
    0.31428571,
    0.33333333,
    0.36363636,
    0.41509434,
    0.5,
    0.53333333,
    0.58333333,
    0.61538462,
    0.66666667,
    0.69090909,
    0.75,
    0.75925926,
    0.78571429,
    0.8,
    0.86666667,
    0.875,
    0.91489362,
    0.93333333,
    1.0,
]

# Most common non-missing training values by country (train cohorts <= 2021)
COUNTRY_TYPICAL_VALUES: Dict[str, Dict[str, str]] = {
    "South Korea": {
        "network": "Naver TV Cast",
        "content_rating": "15+",
    },
    "China": {
        "network": "iQiyi",
        "content_rating": "13+",
    },
    "Japan": {
        "network": "NHK",
        "content_rating": "13+",
    },
    "Thailand": {
        "network": "Channel 7",
        "content_rating": "G",
    },
}

# Empirical training frequency distributions across 57 networks and content ratings (train cohorts <= 2021)
COUNTRY_TRAIN_DISTRIBUTIONS: Dict[str, Dict[str, Dict[str, int]]] = {
    "South Korea": {
        "networks": {
            "Naver TV Cast": 166, "Other": 142, "tvN": 112, "KBS2": 107, "MBC": 102,
            "SBS": 100, "jTBC": 58, "Naver TV Cast  vLive": 39, "OCN": 36, "KBS1": 22,
            "vLive": 17, "Netflix": 17, "Daum Kakao TV": 13, "Tencent Video": 2,
        },
        "content_ratings": {"15+": 639, "13+": 93, "G": 33, "18+": 17},
    },
    "China": {
        "networks": {
            "Other": 229, "iQiyi": 217, "Youku": 215, "Tencent Video": 195, "Hunan TV": 99,
            "Mango TV": 85, "CCTV": 82, "Kuaishou": 67, "Sohu TV": 54, "JSTV": 42,
            "Anhui TV": 33, "ZJTV": 32, "Dragon TV": 27, "BTV": 23, "LeTV": 17,
            "Shenzhen TV": 16, "iQiyi  Tencent Video": 15, "Hunan TV  Mango TV": 13,
            "iQiyi  Tencent Video  Youku": 12, "Douyin": 10, "PPTV": 3,
        },
        "content_ratings": {"13+": 257, "G": 100, "15+": 73, "18+": 9},
    },
    "Japan": {
        "networks": {
            "Other": 119, "NHK": 107, "TV Tokyo": 88, "Fuji TV": 87, "TV Asahi": 82,
            "NTV": 71, "TBS": 66, "WOWOW": 34, "Amazon Prime": 23, "Tokyo MX": 18,
            "Hulu": 17, "Fuji TV  Tokai TV": 12, "MBS": 12, "Paravi": 12, "Netflix": 9,
        },
        "content_ratings": {"13+": 46, "G": 46, "15+": 40, "18+": 16},
    },
    "Thailand": {
        "networks": {
            "Channel 7": 243, "Channel 3": 240, "One 31": 162, "GMM 25": 140, "Other": 65,
            "Channel 8": 54, "LINE TV": 33, "PPTV": 27, "True4U": 22, "Channel 9": 16,
            "LINE TV  One 31": 14, "Workpoint TV": 14, "True4U  TrueID": 10, "Thai PBS": 10,
            "Netflix": 2, "iQiyi": 1,
        },
        "content_ratings": {"G": 259, "13+": 140, "15+": 68, "18+": 40},
    },
}

# Empirical joint training distribution of (original_network, content_rating) for dramas with both observed (train <= 2021)
COUNTRY_JOINT_TRAIN_DISTRIBUTIONS: Dict[str, Dict[tuple[str, str], int]] = {
    "South Korea": {
        ("tvN", "15+"): 104, ("MBC", "15+"): 95, ("KBS2", "15+"): 86, ("SBS", "15+"): 84,
        ("Other", "15+"): 74, ("jTBC", "15+"): 54, ("Naver TV Cast", "13+"): 33, ("OCN", "15+"): 29,
        ("Other", "13+"): 22, ("Naver TV Cast", "15+"): 20, ("Naver TV Cast  vLive", "15+"): 17,
        ("KBS1", "15+"): 14, ("Daum Kakao TV", "15+"): 10, ("Netflix", "15+"): 9, ("vLive", "15+"): 6,
        ("Naver TV Cast", "G"): 6, ("Other", "18+"): 6, ("Other", "G"): 5, ("Netflix", "18+"): 5,
        ("Naver TV Cast  vLive", "13+"): 4, ("KBS2", "13+"): 4, ("Netflix", "13+"): 3, ("SBS", "13+"): 3,
        ("vLive", "G"): 2, ("vLive", "13+"): 2, ("MBC", "13+"): 2, ("Naver TV Cast  vLive", "G"): 2,
        ("OCN", "18+"): 2, ("Tencent Video", "15+"): 2, ("jTBC", "G"): 1, ("jTBC", "13+"): 1,
        ("MBC", "18+"): 1, ("KBS2", "G"): 1, ("SBS", "G"): 1, ("MBC", "G"): 1, ("tvN", "13+"): 1,
        ("Daum Kakao TV", "13+"): 1, ("Daum Kakao TV", "G"): 1, ("SBS", "18+"): 1, ("jTBC", "18+"): 1,
        ("tvN", "18+"): 1,
    },
    "China": {
        ("Youku", "13+"): 50, ("iQiyi", "13+"): 46, ("Other", "13+"): 40, ("Tencent Video", "13+"): 34,
        ("Youku", "G"): 20, ("Mango TV", "13+"): 20, ("Other", "G"): 14, ("Tencent Video", "15+"): 14,
        ("Other", "15+"): 13, ("Youku", "15+"): 13, ("Hunan TV", "13+"): 12, ("Mango TV", "G"): 9,
        ("Tencent Video", "G"): 9, ("Sohu TV", "G"): 7, ("Sohu TV", "13+"): 7, ("iQiyi", "15+"): 6,
        ("Hunan TV", "G"): 6, ("iQiyi  Tencent Video", "13+"): 6, ("Hunan TV", "15+"): 5, ("CCTV", "G"): 5,
        ("Mango TV", "15+"): 5, ("iQiyi", "G"): 4, ("iQiyi  Tencent Video  Youku", "15+"): 4,
        ("iQiyi  Tencent Video  Youku", "13+"): 4, ("Hunan TV  Mango TV", "G"): 4, ("CCTV", "15+"): 3,
        ("Youku", "18+"): 2, ("Anhui TV", "13+"): 2, ("ZJTV", "13+"): 2, ("LeTV", "G"): 2,
        ("JSTV", "13+"): 2, ("Dragon TV", "13+"): 2, ("JSTV", "G"): 2, ("Kuaishou", "13+"): 2,
        ("Hunan TV  Mango TV", "13+"): 2, ("Other", "18+"): 1, ("BTV", "13+"): 1, ("Anhui TV", "G"): 1,
        ("LeTV", "13+"): 1, ("Anhui TV", "15+"): 1, ("JSTV", "15+"): 1, ("Hunan TV", "18+"): 1,
        ("Shenzhen TV", "13+"): 1, ("iQiyi", "18+"): 1, ("ZJTV", "15+"): 1, ("Tencent Video", "18+"): 1,
        ("Kuaishou", "15+"): 1, ("iQiyi  Tencent Video", "G"): 1, ("Sohu TV", "15+"): 1,
    },
    "Japan": {
        ("NHK", "G"): 10, ("Other", "G"): 10, ("NTV", "13+"): 9, ("Other", "13+"): 7,
        ("TBS", "15+"): 7, ("Other", "15+"): 6, ("NTV", "15+"): 5, ("Fuji TV", "13+"): 5,
        ("NHK", "13+"): 5, ("Fuji TV", "15+"): 5, ("WOWOW", "15+"): 4, ("TV Asahi", "13+"): 4,
        ("TV Tokyo", "G"): 4, ("Fuji TV", "18+"): 3, ("TV Tokyo", "15+"): 3, ("MBS", "13+"): 3,
        ("NTV", "G"): 3, ("TV Asahi", "15+"): 3, ("Netflix", "G"): 3, ("Amazon Prime", "13+"): 3,
        ("Other", "18+"): 3, ("Fuji TV", "G"): 3, ("TBS", "13+"): 3, ("Netflix", "15+"): 2,
        ("Netflix", "13+"): 2, ("TV Asahi", "G"): 2, ("Hulu", "18+"): 2, ("Hulu", "13+"): 2,
        ("TBS", "18+"): 2, ("Amazon Prime", "18+"): 2, ("TBS", "G"): 2, ("MBS", "15+"): 2,
        ("WOWOW", "13+"): 1, ("Amazon Prime", "15+"): 1, ("TV Tokyo", "18+"): 1, ("NHK", "18+"): 1,
        ("Hulu", "G"): 1, ("Amazon Prime", "G"): 1, ("Netflix", "18+"): 1, ("Tokyo MX", "15+"): 1,
        ("TV Tokyo", "13+"): 1, ("NHK", "15+"): 1,
    },
    "Thailand": {
        ("One 31", "G"): 84, ("GMM 25", "G"): 70, ("Channel 3", "G"): 42, ("GMM 25", "13+"): 36,
        ("One 31", "13+"): 33, ("Channel 7", "G"): 31, ("Other", "13+"): 14, ("Channel 7", "13+"): 13,
        ("Channel 3", "13+"): 13, ("Channel 3", "15+"): 13, ("GMM 25", "15+"): 11, ("One 31", "15+"): 9,
        ("Channel 7", "15+"): 8, ("Channel 8", "G"): 8, ("Channel 9", "13+"): 7, ("Other", "15+"): 7,
        ("Other", "18+"): 7, ("GMM 25", "18+"): 6, ("One 31", "18+"): 6, ("Other", "G"): 6,
        ("LINE TV", "15+"): 5, ("LINE TV", "18+"): 5, ("LINE TV  One 31", "13+"): 5, ("LINE TV", "13+"): 5,
        ("PPTV", "G"): 4, ("LINE TV  One 31", "18+"): 4, ("Channel 8", "13+"): 3, ("Channel 9", "15+"): 3,
        ("LINE TV  One 31", "G"): 3, ("LINE TV", "G"): 2, ("Channel 8", "18+"): 2, ("True4U", "15+"): 2,
        ("Channel 7", "18+"): 2, ("Channel 8", "15+"): 2, ("PPTV", "13+"): 2, ("PPTV", "15+"): 1,
        ("True4U  TrueID", "G"): 1, ("True4U  TrueID", "13+"): 1, ("Workpoint TV", "G"): 1,
        ("True4U", "G"): 1, ("Netflix", "15+"): 1, ("Workpoint TV", "18+"): 1,
        ("LINE TV  One 31", "15+"): 1, ("Workpoint TV", "15+"): 1, ("Channel 9", "G"): 1,
        ("True4U", "13+"): 1, ("Channel 3", "18+"): 1, ("iQiyi", "13+"): 1,
        ("True4U  TrueID", "15+"): 1, ("Thai PBS", "G"): 1,
    },
}


def is_blank_value(val: Any) -> bool:
    """Helper to detect if a network or content rating value is blank/missing."""
    if val is None:
        return True
    s = str(val).strip()
    return s in ["", "Unknown", "Not Yet Rated", "nan", "None"]


def is_near_calibrator_band_edge(cal_p: float) -> bool:
    """
    Returns True if calibrated P(High) sits on or within one plateau of a band edge (0.20, 0.50, 0.75).
    In the 35 sorted plateaus, edge indices are:
    - 0.20: index 13 (neighbors: idx 12 [0.1667], idx 14 [0.2500])
    - 0.50: index 20 (neighbors: idx 19 [0.4151], idx 21 [0.5333])
    - 0.75: index 26 (neighbors: idx 25 [0.6909], idx 27 [0.7593])
    """
    diffs = [abs(p - cal_p) for p in CALIBRATOR_PLATEAUS]
    closest_idx = int(np.argmin(diffs))
    edge_indices = {12, 13, 14, 19, 20, 21, 25, 26, 27}
    return closest_idx in edge_indices


def match_network(typed_net: Optional[str], valid_networks: List[str]) -> tuple[Optional[str], Optional[str]]:
    """
    Matches a typed network name to the 57 valid training networks case- and whitespace-insensitively.
    If no match, finds close matches using difflib with cutoff=0.8.
    When nothing is close (similarity < 0.8), emits warning that network is not recognized
    and falls back to 'Other', with NO suggestions.
    Returns: (matched_network, warning_if_unrecognized)
    """
    if is_blank_value(typed_net):
        return None, None
    s = " ".join(str(typed_net).strip().split())
    norm_map = {" ".join(n.strip().split()).lower(): n for n in valid_networks}
    if s.lower() in norm_map:
        return norm_map[s.lower()], None

    lower_candidates = {n.lower(): n for n in valid_networks}
    close_lowers = difflib.get_close_matches(s.lower(), list(lower_candidates.keys()), n=3, cutoff=0.8)
    if close_lowers:
        suggestions = [lower_candidates[cl] for cl in close_lowers]
    else:
        suggestions = difflib.get_close_matches(s, valid_networks, n=3, cutoff=0.8)

    sugg_str = f" Did you mean: {', '.join(repr(x) for x in suggestions)}?" if suggestions else ""
    warning = f"Broadcaster network '{typed_net}' is not one of the 57 recognized training networks; falls back to 'Other'.{sugg_str}"
    return s, warning


# -----------------------------------------------------------------------------
# Input & Output TypedDict Schemas
# -----------------------------------------------------------------------------

class PitchInput(TypedDict, total=False):
    """Input pitch schema."""
    country: str
    network: Optional[str]
    genres: List[str]
    content_rating: Optional[str]
    weekdays: List[str]
    episodes: Union[int, float]
    directors: List[str]
    writers: List[str]
    lead_cast: List[str]
    planned_start: Union[str, int]
    synopsis_text: Optional[str]


class QuotableBlock(TypedDict):
    """Guaranteed quotable block for downstream memo text and agent quoting."""
    likelihood_band: str
    observed_band_rate: str


class ComparableQuotableRecord(TypedDict):
    """Clean quotable record for a retrieved comparable drama."""
    name: str
    release_year: int
    country: str
    network: Optional[str]
    tier: str
    url: str


class ComparablesQuotable(TypedDict):
    """Quotable block for comparables tool."""
    comparables: List[ComparableQuotableRecord]
    caveat: str
    scoring_note: str


class ComparableNeighbor(TypedDict):
    """Metadata dictionary for a single retrieved comparable drama."""
    name: str
    release_year: int
    country: str
    network: Optional[str]
    target_tier: str
    target_pct: float
    similarity_score: float
    url: str


class FindComparablesOutput(TypedDict):
    """Output schema for find_comparables tool."""
    tool: Literal["find_comparables"]
    k_requested: int
    k_returned: int
    candidate_pool_size: int
    planned_start: str
    maturity_cutoff: str
    country_filter: Optional[str]
    comparables: List[ComparableNeighbor]
    quotable: ComparablesQuotable
    caveat: str


class PredictionSummary(TypedDict):
    """Prediction summary sub-schema."""
    raw_p_high: float
    calibrated_p_high: float
    predicted_tier: str
    tier_note: str
    likelihood_band: str
    observed_band_rate: str


class ExplainPitchOutput(TypedDict):
    """Output schema for explain_pitch tool."""
    tool: Literal["explain_pitch"]
    pitch_summary: Dict[str, Any]
    prediction: PredictionSummary
    quotable: QuotableBlock
    attributions: Dict[str, float]
    top_positive_drivers: List[Dict[str, Any]]
    top_negative_drivers: List[Dict[str, Any]]
    warnings: List[str]
    caveats: List[str]


class WhatIfState(TypedDict):
    """State before or after changes in what_if tool."""
    raw_p_high: float
    calibrated_p_high: float
    predicted_tier: str
    likelihood_band: str


class WhatIfDeltas(TypedDict):
    """Probability deltas in what_if tool."""
    raw_p_high_delta: float
    calibrated_p_high_delta: float


class RawDecompositionShares(TypedDict):
    """Shares of raw P(High) delta attributable to baseline control vs specific choice."""
    marginal_control_share_pct: float
    specific_choice_share_pct: float
    total_share_pct: float


class BlankFillDecomposition(TypedDict):
    """Decomposition of what_if delta when filling blank fields."""
    filled_fields: List[str]
    marginal_control_distribution: str
    raw_shares: RawDecompositionShares
    marginal_control_delta: WhatIfDeltas
    specific_choice_delta: WhatIfDeltas
    calibrated_delta_note: str
    mode_values_used: Dict[str, str]
    mode_control_delta: WhatIfDeltas
    mode_specific_choice_delta: WhatIfDeltas


class WhatIfQuotable(TypedDict):
    """Quotable block for what_if tool."""
    before: QuotableBlock
    after: QuotableBlock
    band_changed: bool
    delta_band_summary: str


class WhatIfOutput(TypedDict):
    """Output schema for what_if tool."""
    tool: Literal["what_if"]
    before: WhatIfState
    after: WhatIfState
    delta: WhatIfDeltas
    decomposition: Optional[BlankFillDecomposition]
    changes_applied: Dict[str, Any]
    quotable: WhatIfQuotable
    band_boundary_warning: Optional[str]
    caveat: str
    warnings: List[str]


# -----------------------------------------------------------------------------
# Runtime Sanitization & Validators
# -----------------------------------------------------------------------------

def sanitize_string(val: Any, max_len: int = 1000) -> str:
    """Strips ASCII/Unicode control characters (except newline/tab) and truncates length."""
    if val is None:
        return ""
    s = str(val)
    # Remove non-printable control chars (preserve standard \n, \r, \t)
    cleaned = "".join(ch for ch in s if ch in "\n\r\t" or ord(ch) >= 32)
    cleaned = cleaned.strip()
    return cleaned[:max_len]


def sanitize_string_list(items: Any, max_items: int = 20, max_item_len: int = 100) -> List[str]:
    """Sanitizes and caps a list of strings."""
    if items is None:
        return []
    if isinstance(items, str):
        items = [x.strip() for x in items.replace("|", ",").split(",") if x.strip()]
    if not isinstance(items, (list, tuple, set)):
        return []
    result = []
    for x in list(items)[:max_items]:
        s = sanitize_string(x, max_len=max_item_len)
        if s:
            result.append(s)
    return result


def validate_pitch(pitch: PitchInput) -> Dict[str, Any]:
    """
    Validates and sanitizes a pitch dictionary.
    Enforces maximum lengths, strips control characters, and treats all text as pure data.
    """
    if not isinstance(pitch, dict):
        raise TypeError(f"Pitch must be a dictionary, got {type(pitch).__name__}")

    country_raw = sanitize_string(pitch.get("country"), max_len=50)
    if country_raw not in VALID_COUNTRIES:
        raise ValueError(
            f"Invalid country '{country_raw}'. Must be one of: {', '.join(sorted(VALID_COUNTRIES))}"
        )

    # Planned start
    start_raw = pitch.get("planned_start")
    if not start_raw:
        raise ValueError("planned_start is required (YYYY-MM-DD date or 4-digit YYYY release year).")
    start_str = sanitize_string(start_raw, max_len=20)
    # Test date parsing
    try:
        parse_query_start_date(start_str)
    except Exception as e:
        raise ValueError(f"Invalid planned_start '{start_str}': {e}")

    # Network
    net_raw = pitch.get("network")
    network = sanitize_string(net_raw, max_len=50) if net_raw else None

    # Content rating
    cr_raw = pitch.get("content_rating")
    content_rating = sanitize_string(cr_raw, max_len=30) if cr_raw else None

    # Episodes
    ep_raw = pitch.get("episodes", 16)
    try:
        episodes = float(ep_raw)
        if episodes <= 0 or np.isnan(episodes):
            episodes = 16.0
    except (ValueError, TypeError):
        episodes = 16.0

    # Genres, Weekdays, Talent
    genres = sanitize_string_list(pitch.get("genres"), max_items=20, max_item_len=50)
    weekdays = sanitize_string_list(pitch.get("weekdays"), max_items=7, max_item_len=20)
    directors = sanitize_string_list(pitch.get("directors"), max_items=10, max_item_len=100)
    writers = sanitize_string_list(pitch.get("writers"), max_items=10, max_item_len=100)
    lead_cast = sanitize_string_list(pitch.get("lead_cast"), max_items=20, max_item_len=100)

    # Synopsis text (optional; capped at 2,000 chars, treated as literal data)
    syn_raw = pitch.get("synopsis_text")
    synopsis_text = sanitize_string(syn_raw, max_len=2000) if syn_raw else None

    return {
        "country": country_raw,
        "network": network,
        "genres": genres,
        "content_rating": content_rating,
        "weekdays": weekdays,
        "episodes": episodes,
        "directors": directors,
        "writers": writers,
        "lead_cast": lead_cast,
        "planned_start": start_str,
        "synopsis_text": synopsis_text,
    }


def validate_what_if_changes(changes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates changes dictionary for what_if tool:
    - Allowed levers ONLY: network, genres, content_rating, directors, writers, lead_cast.
    - Rejects disallowed levers (episodes, weekdays, country, planned_start) with a clear message.
    - At most 3 changes per call.
    """
    if not isinstance(changes, dict):
        raise TypeError(f"Changes must be a dictionary, got {type(changes).__name__}")

    if len(changes) > 3:
        raise ValueError(
            f"what_if accepts at most 3 changes per call; received {len(changes)} changes: {list(changes.keys())}"
        )

    validated_changes: Dict[str, Any] = {}
    for key, val in changes.items():
        k = str(key).strip().lower()
        if k in DISALLOWED_LEVERS:
            raise ValueError(
                f"Disallowed lever '{key}' in what_if. Only network, genres, content_rating, "
                "directors, writers, and lead_cast may be modified."
            )
        if k not in ALLOWED_WHAT_IF_LEVERS:
            raise ValueError(
                f"Unrecognized lever '{key}'. Allowed levers are: {', '.join(sorted(ALLOWED_WHAT_IF_LEVERS))}."
            )

        if k == "network":
            validated_changes["network"] = sanitize_string(val, max_len=50) if val else None
        elif k == "content_rating":
            validated_changes["content_rating"] = sanitize_string(val, max_len=30) if val else None
        elif k in ["genres", "directors", "writers", "lead_cast"]:
            max_n = 20 if k in ["genres", "lead_cast"] else 10
            validated_changes[k] = sanitize_string_list(val, max_items=max_n, max_item_len=100)

    return validated_changes


# -----------------------------------------------------------------------------
# Tool 1: explain_pitch
# -----------------------------------------------------------------------------

def explain_pitch(
    pitch: PitchInput,
    artifacts: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    max_evals: int = 1000,
) -> ExplainPitchOutput:
    """
    Wraps explain_drama to evaluate a pitch with tabular features, SHAP attributions,
    and calibrated likelihood bands.
    """
    validated = validate_pitch(pitch)
    if artifacts is None:
        artifacts = load_explain_artifacts()

    # Format dictionary for explain_drama with case-insensitive network matching
    meta = artifacts.get("metadata", {})
    valid_networks = meta.get("valid_networks", [])
    matched_network, net_warning = match_network(validated.get("network"), valid_networks)

    inputs = {
        "country": validated["country"],
        "network": matched_network,
        "genres": validated["genres"],
        "content_rating": validated["content_rating"],
        "weekdays": validated["weekdays"],
        "episodes": validated["episodes"],
        "directors": validated["directors"],
        "writers": validated["writers"],
        "lead_cast": validated["lead_cast"],
        "planned_start": validated["planned_start"],
    }

    # Run production explain_drama
    raw_explain = explain_drama(inputs=inputs, artifacts=artifacts, seed=seed, max_evals=max_evals)

    # Summary of pitch fields (strictly omitting any synopsis text)
    pitch_summary = {
        "country": validated["country"],
        "network": matched_network,
        "genres": validated["genres"],
        "content_rating": validated["content_rating"],
        "weekdays": validated["weekdays"],
        "episodes": validated["episodes"],
        "directors": validated["directors"],
        "writers": validated["writers"],
        "lead_cast": validated["lead_cast"],
        "planned_start": validated["planned_start"],
    }

    prediction: PredictionSummary = {
        "raw_p_high": raw_explain["raw_p_high"],
        "calibrated_p_high": raw_explain["calibrated_p_high"],
        "predicted_tier": raw_explain["predicted_tier"],
        "tier_note": "most likely tier among three (model argmax)",
        "likelihood_band": raw_explain["likelihood_band"],
        "observed_band_rate": raw_explain["band_observed_rate"],
    }

    raw_warnings = list(raw_explain.get("warnings", []))
    if net_warning:
        raw_warnings.append(net_warning)

    quotable: QuotableBlock = {
        "likelihood_band": raw_explain["likelihood_band"],
        "observed_band_rate": raw_explain["band_observed_rate"],
    }

    output: ExplainPitchOutput = {
        "tool": "explain_pitch",
        "pitch_summary": pitch_summary,
        "prediction": prediction,
        "quotable": quotable,
        "attributions": raw_explain["group_attributions"],
        "top_positive_drivers": raw_explain["top_positive_groups"],
        "top_negative_drivers": raw_explain["top_negative_groups"],
        "warnings": raw_warnings,
        "caveats": raw_explain["caveats"],
    }

    # Verify JSON serializability
    json.dumps(output)
    return output


# -----------------------------------------------------------------------------
# Tool 2: find_comparables
# -----------------------------------------------------------------------------

def find_comparables(
    pitch: PitchInput,
    k: int = 5,
    artifacts: Optional[Dict[str, Any]] = None,
    model: Any = None,
) -> FindComparablesOutput:
    """
    Wraps find_similar using synopsis_text, planned_start, and country.
    Returns metadata only (no synopsis text) with fixed caveat.
    """
    validated = validate_pitch(pitch)
    synopsis = validated.get("synopsis_text")
    if not synopsis or not synopsis.strip():
        empty_output: FindComparablesOutput = {
            "tool": "find_comparables",
            "k_requested": int(k),
            "k_returned": 0,
            "candidate_pool_size": 0,
            "planned_start": str(validated["planned_start"]),
            "maturity_cutoff": "N/A",
            "country_filter": validated["country"],
            "comparables": [],
            "quotable": {
                "comparables": [],
                "caveat": COMPARABLES_CAVEAT,
                "scoring_note": "No synopsis text provided in pitch; textual premise comparables cannot be retrieved.",
            },
            "caveat": COMPARABLES_CAVEAT,
        }
        json.dumps(empty_output)
        return empty_output

    if artifacts is None:
        artifacts = load_retriever_artifacts()

    # Call find_similar
    sim_result = find_similar(
        query_text=synopsis,
        planned_start=validated["planned_start"],
        country=validated["country"],
        k=k,
        artifacts=artifacts,
        model=model,
    )

    # Sanitize comparables metadata: ensure NO synopsis text
    comparables: List[ComparableNeighbor] = []
    for n in sim_result.get("neighbors", []):
        comparables.append(
            {
                "name": str(n["name"]),
                "release_year": int(n["release_year"]),
                "country": str(n["country"]),
                "network": n["network"],
                "target_tier": str(n["target_tier"]),
                "target_pct": float(n["target_pct"]),
                "similarity_score": float(n["similarity_score"]),
                "url": str(n["url"]),
            }
        )

    quotable_comparables: List[ComparableQuotableRecord] = [
        {
            "name": str(c["name"]),
            "release_year": int(c["release_year"]),
            "country": str(c["country"]),
            "network": c["network"],
            "tier": str(c["target_tier"]),
            "url": str(c["url"]),
        }
        for c in comparables
    ]

    comparables_quotable: ComparablesQuotable = {
        "comparables": quotable_comparables,
        "caveat": COMPARABLES_CAVEAT,
        "scoring_note": "Similarity scores measure text premise proximity and must not be presented as probabilities.",
    }

    output: FindComparablesOutput = {
        "tool": "find_comparables",
        "k_requested": int(k),
        "k_returned": len(comparables),
        "candidate_pool_size": int(sim_result["candidate_pool_size"]),
        "planned_start": sim_result["planned_start"],
        "maturity_cutoff": sim_result["maturity_cutoff"],
        "country_filter": sim_result["country_filter"],
        "comparables": comparables,
        "quotable": comparables_quotable,
        "caveat": COMPARABLES_CAVEAT,
    }

    # Verify JSON serializability
    json.dumps(output)
    return output


# -----------------------------------------------------------------------------
# Tool 3: what_if
# -----------------------------------------------------------------------------

def _score_pitch(pitch_dict: Dict[str, Any], artifacts: Dict[str, Any]) -> tuple[WhatIfState, List[str]]:
    """Helper to compute raw and calibrated P(High) for a pitch without full SHAP."""
    clf = artifacts["headline_model"]
    iso_cal = artifacts["isotonic_calibrator"]
    meta = artifacts["metadata"]
    high_idx = meta["high_class_idx"]

    X_num, warnings = build_feature_row(pitch_dict, artifacts)
    raw_p = float(clf.predict_proba(X_num)[:, high_idx][0])
    cal_p_raw = float(iso_cal.predict([raw_p])[0])
    cal_p = float(np.clip(cal_p_raw, 0.02, 0.98))
    pred_tier = str(clf.predict(X_num)[0])
    band = band_for(cal_p)

    state: WhatIfState = {
        "raw_p_high": float(round(raw_p, 4)),
        "calibrated_p_high": float(round(cal_p, 4)),
        "predicted_tier": pred_tier,
        "likelihood_band": band,
    }
    return state, warnings


def what_if(
    pitch: PitchInput,
    changes: Dict[str, Any],
    artifacts: Optional[Dict[str, Any]] = None,
) -> WhatIfOutput:
    """
    Evaluates what-if attachment levers on a pitch:
    - Allowed levers ONLY: network, genres, content_rating, directors, writers, lead_cast.
    - Rejects disallowed levers (episodes, weekdays, country, planned_start).
    - At most 3 changes per call.
    - Recomputes talent priors with the 12-month rule.
    - Returns before/after raw and calibrated P(High), band, deltas, and associational caveat.
    """
    validated_pitch = validate_pitch(pitch)
    validated_changes = validate_what_if_changes(changes)

    if artifacts is None:
        artifacts = load_explain_artifacts()

    meta = artifacts.get("metadata", {})
    valid_networks = meta.get("valid_networks", [])

    # Case- and whitespace-insensitive network matching for baseline
    matched_base_net, base_net_warn = match_network(validated_pitch.get("network"), valid_networks)
    base_pitch_to_score = copy.deepcopy(validated_pitch)
    if matched_base_net is not None:
        base_pitch_to_score["network"] = matched_base_net

    # 1. Evaluate baseline ('before')
    before_state, before_warnings = _score_pitch(base_pitch_to_score, artifacts)
    if base_net_warn:
        before_warnings.append(base_net_warn)

    # 2. If no changes, return delta 0
    if not validated_changes:
        band_edge_triggered = is_near_calibrator_band_edge(before_state["calibrated_p_high"])
        band_boundary_warning = (
            "band edges are approximate: the calibrator is a step function"
            if band_edge_triggered
            else None
        )
        if band_boundary_warning:
            before_warnings.append(band_boundary_warning)
        empty_quotable: WhatIfQuotable = {
            "before": {
                "likelihood_band": before_state["likelihood_band"],
                "observed_band_rate": TEST_BAND_OBSERVED_RATES[before_state["likelihood_band"]],
            },
            "after": {
                "likelihood_band": before_state["likelihood_band"],
                "observed_band_rate": TEST_BAND_OBSERVED_RATES[before_state["likelihood_band"]],
            },
            "band_changed": False,
            "delta_band_summary": f"Likelihood band remains '{before_state['likelihood_band']}' ({TEST_BAND_OBSERVED_RATES[before_state['likelihood_band']]})",
        }
        output: WhatIfOutput = {
            "tool": "what_if",
            "before": before_state,
            "after": copy.deepcopy(before_state),
            "delta": {
                "raw_p_high_delta": 0.0,
                "calibrated_p_high_delta": 0.0,
            },
            "decomposition": None,
            "changes_applied": {},
            "quotable": empty_quotable,
            "band_boundary_warning": band_boundary_warning,
            "caveat": WHAT_IF_CAVEAT,
            "warnings": list(dict.fromkeys(before_warnings)),
        }
        json.dumps(output)
        return output

    # 3. Apply changes onto modified pitch with network matching
    modified_pitch = copy.deepcopy(base_pitch_to_score)
    applied_changes = copy.deepcopy(validated_changes)
    change_net_warn = None
    if "network" in applied_changes:
        matched_change_net, change_net_warn = match_network(applied_changes["network"], valid_networks)
        modified_pitch["network"] = matched_change_net
        applied_changes["network"] = matched_change_net
        for k, v in applied_changes.items():
            if k != "network":
                modified_pitch[k] = v
    else:
        for k, v in applied_changes.items():
            modified_pitch[k] = v

    # 4. Evaluate modified pitch ('after') - talent priors recomputed automatically
    after_state, after_warnings = _score_pitch(modified_pitch, artifacts)

    # Combine unique warnings
    combined_warnings = list(dict.fromkeys(before_warnings + after_warnings))
    if change_net_warn:
        combined_warnings.append(change_net_warn)

    # 5. Detect blank-filling changes and compute decomposition
    blank_filled_levers = []
    for lever in ["network", "content_rating"]:
        if (
            lever in applied_changes
            and is_blank_value(validated_pitch.get(lever))
            and not is_blank_value(applied_changes[lever])
        ):
            blank_filled_levers.append(lever)

    decomposition: Optional[BlankFillDecomposition] = None
    if blank_filled_levers:
        country = validated_pitch["country"]
        country_dist = COUNTRY_TRAIN_DISTRIBUTIONS.get(country, COUNTRY_TRAIN_DISTRIBUTIONS["South Korea"])
        net_dist = country_dist["networks"]
        cr_dist = country_dist["content_ratings"]
        total_net = sum(net_dist.values())
        total_cr = sum(cr_dist.values())

        clf = artifacts["headline_model"]
        iso_cal = artifacts["isotonic_calibrator"]
        meta = artifacts["metadata"]
        high_idx = meta["high_class_idx"]

        # 1. Marginal distribution control: expectation over country training distribution
        cand_items: List[Dict[str, str]] = []
        weights: List[float] = []

        if "network" in blank_filled_levers and "content_rating" in blank_filled_levers:
            dist_name = "empirical joint distribution of (network, content_rating) in country training cohorts (years <= 2021)"
            joint_pairs = COUNTRY_JOINT_TRAIN_DISTRIBUTIONS.get(country, COUNTRY_JOINT_TRAIN_DISTRIBUTIONS["South Korea"])
            total_joint = sum(joint_pairs.values())
            for (net, cr), count in joint_pairs.items():
                cand_items.append({"network": net, "content_rating": cr})
                weights.append(count / total_joint)
        elif "network" in blank_filled_levers:
            dist_name = "empirical univariate distribution of network in country training cohorts (years <= 2021)"
            for net, n_count in net_dist.items():
                cand_items.append({"network": net})
                weights.append(n_count / total_net)
        else:  # "content_rating" in blank_filled_levers
            dist_name = "empirical univariate distribution of content_rating in country training cohorts (years <= 2021)"
            for cr, cr_count in cr_dist.items():
                cand_items.append({"content_rating": cr})
                weights.append(cr_count / total_cr)

        batch_rows = []
        for cand in cand_items:
            cand_p = copy.deepcopy(base_pitch_to_score)
            for k, v in cand.items():
                cand_p[k] = v
            row, _ = build_feature_row(cand_p, artifacts)
            batch_rows.append(row)

        batch_df = pd.concat(batch_rows, ignore_index=True)
        raw_probs = clf.predict_proba(batch_df)[:, high_idx]
        cal_probs = np.clip(iso_cal.predict(raw_probs), 0.02, 0.98)

        exp_raw = float(np.sum(raw_probs * np.array(weights)))
        exp_cal = float(np.sum(cal_probs * np.array(weights)))

        total_raw_delta = round(after_state["raw_p_high"] - before_state["raw_p_high"], 4)
        m_raw_delta = round(exp_raw - before_state["raw_p_high"], 4)
        s_raw_delta = round(after_state["raw_p_high"] - exp_raw, 4)

        if abs(total_raw_delta) > 1e-6:
            marginal_share = round((m_raw_delta / total_raw_delta) * 100.0, 1)
            specific_share = round((s_raw_delta / total_raw_delta) * 100.0, 1)
            total_share = 100.0
        else:
            marginal_share = 0.0
            specific_share = 0.0
            total_share = 0.0

        raw_shares: RawDecompositionShares = {
            "marginal_control_share_pct": marginal_share,
            "specific_choice_share_pct": specific_share,
            "total_share_pct": total_share,
        }

        marginal_ctrl_delta: WhatIfDeltas = {
            "raw_p_high_delta": float(m_raw_delta),
            "calibrated_p_high_delta": float(round(exp_cal - before_state["calibrated_p_high"], 4)),
        }
        specific_choice_delta: WhatIfDeltas = {
            "raw_p_high_delta": float(s_raw_delta),
            "calibrated_p_high_delta": float(round(after_state["calibrated_p_high"] - exp_cal, 4)),
        }

        # 2. Secondary mode-based control
        typical_dict = COUNTRY_TYPICAL_VALUES.get(country, {"network": "Other", "content_rating": "15+"})
        mode_pitch = copy.deepcopy(base_pitch_to_score)
        mode_used = {}
        for lever in blank_filled_levers:
            typ_val = typical_dict.get(lever, "Other")
            mode_pitch[lever] = typ_val
            mode_used[lever] = typ_val

        mode_state, _ = _score_pitch(mode_pitch, artifacts)

        mode_ctrl_delta: WhatIfDeltas = {
            "raw_p_high_delta": float(round(mode_state["raw_p_high"] - before_state["raw_p_high"], 4)),
            "calibrated_p_high_delta": float(round(mode_state["calibrated_p_high"] - before_state["calibrated_p_high"], 4)),
        }
        mode_specific_delta: WhatIfDeltas = {
            "raw_p_high_delta": float(round(after_state["raw_p_high"] - mode_state["raw_p_high"], 4)),
            "calibrated_p_high_delta": float(round(after_state["calibrated_p_high"] - mode_state["calibrated_p_high"], 4)),
        }

        decomposition = {
            "filled_fields": blank_filled_levers,
            "marginal_control_distribution": dist_name,
            "raw_shares": raw_shares,
            "marginal_control_delta": marginal_ctrl_delta,
            "specific_choice_delta": specific_choice_delta,
            "calibrated_delta_note": "step-function values, not measurements",
            "mode_values_used": mode_used,
            "mode_control_delta": mode_ctrl_delta,
            "mode_specific_choice_delta": mode_specific_delta,
        }
        combined_warnings.append(
            "part of this change reflects the model's lower likelihood of the High tier when a field is blank; "
            "marginal and mode controls decompose this baseline lift."
        )

    # 6. Check band boundary proximity and band transitions
    band_edge_triggered = (
        is_near_calibrator_band_edge(before_state["calibrated_p_high"])
        or is_near_calibrator_band_edge(after_state["calibrated_p_high"])
        or (before_state["likelihood_band"] != after_state["likelihood_band"])
    )
    band_boundary_warning = (
        BAND_BOUNDARY_WARNING
        if band_edge_triggered
        else None
    )
    if band_boundary_warning:
        combined_warnings.append(band_boundary_warning)

    quotable: WhatIfQuotable = {
        "before": {
            "likelihood_band": before_state["likelihood_band"],
            "observed_band_rate": TEST_BAND_OBSERVED_RATES[before_state["likelihood_band"]],
        },
        "after": {
            "likelihood_band": after_state["likelihood_band"],
            "observed_band_rate": TEST_BAND_OBSERVED_RATES[after_state["likelihood_band"]],
        },
        "band_changed": before_state["likelihood_band"] != after_state["likelihood_band"],
        "delta_band_summary": (
            f"Likelihood band moves from '{before_state['likelihood_band']}' ({TEST_BAND_OBSERVED_RATES[before_state['likelihood_band']]}) to '{after_state['likelihood_band']}' ({TEST_BAND_OBSERVED_RATES[after_state['likelihood_band']]})"
            if before_state["likelihood_band"] != after_state["likelihood_band"]
            else f"Likelihood band remains '{before_state['likelihood_band']}' ({TEST_BAND_OBSERVED_RATES[before_state['likelihood_band']]})"
        ),
    }

    output: WhatIfOutput = {
        "tool": "what_if",
        "before": before_state,
        "after": after_state,
        "delta": {
            "raw_p_high_delta": float(round(after_state["raw_p_high"] - before_state["raw_p_high"], 4)),
            "calibrated_p_high_delta": float(round(after_state["calibrated_p_high"] - before_state["calibrated_p_high"], 4)),
        },
        "decomposition": decomposition,
        "changes_applied": applied_changes,
        "quotable": quotable,
        "band_boundary_warning": band_boundary_warning,
        "caveat": WHAT_IF_CAVEAT,
        "warnings": list(dict.fromkeys(combined_warnings)),
    }

    json.dumps(output)
    return output
