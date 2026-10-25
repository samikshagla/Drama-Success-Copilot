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


def is_blank_value(val: Any) -> bool:
    """Helper to detect if a network or content rating value is blank/missing."""
    if val is None:
        return True
    s = str(val).strip()
    return s in ["", "Unknown", "Not Yet Rated", "nan", "None"]


def format_probability_display(cal_p: float) -> str:
    """
    Format calibrated probability for executive display:
    - Extremes (at or near 0.02 / 0.98 clip bounds): "under 5%" or "over 95%"
    - Intermediate values: rounded to nearest 5% increment (e.g., "10%", "25%", "60%")
    - Never prints "0%" / "0.00" or "100%" / "1.00"
    """
    if cal_p <= 0.05:
        return "under 5%"
    if cal_p >= 0.95:
        return "over 95%"
    rounded_pct = int(round(round(cal_p * 20.0) / 20.0 * 100))
    if rounded_pct <= 0:
        return "under 5%"
    if rounded_pct >= 100:
        return "over 95%"
    return f"{rounded_pct}%"


def match_network(typed_net: Optional[str], valid_networks: List[str]) -> tuple[Optional[str], Optional[str]]:
    """
    Matches a typed network name to the 57 valid training networks case- and whitespace-insensitively.
    If no match, finds up to 3 close matches using difflib.
    Returns: (matched_network, warning_if_unrecognized)
    """
    if is_blank_value(typed_net):
        return None, None
    s = " ".join(str(typed_net).strip().split())
    norm_map = {" ".join(n.strip().split()).lower(): n for n in valid_networks}
    if s.lower() in norm_map:
        return norm_map[s.lower()], None

    lower_candidates = {n.lower(): n for n in valid_networks}
    close_lowers = difflib.get_close_matches(s.lower(), list(lower_candidates.keys()), n=3, cutoff=0.4)
    if close_lowers:
        suggestions = [lower_candidates[cl] for cl in close_lowers]
    else:
        suggestions = difflib.get_close_matches(s, valid_networks, n=3, cutoff=0.3)

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
    probability_display: str


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


class BlankFillDecomposition(TypedDict):
    """Decomposition of what_if delta when filling blank fields."""
    filled_fields: List[str]
    typical_values_used: Dict[str, str]
    typical_value_control_delta: WhatIfDeltas
    specific_choice_delta: WhatIfDeltas


class WhatIfQuotable(TypedDict):
    """Quotable block for what_if tool."""
    before: QuotableBlock
    after: QuotableBlock


class WhatIfOutput(TypedDict):
    """Output schema for what_if tool."""
    tool: Literal["what_if"]
    before: WhatIfState
    after: WhatIfState
    delta: WhatIfDeltas
    decomposition: Optional[BlankFillDecomposition]
    changes_applied: Dict[str, Any]
    quotable: WhatIfQuotable
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
        "probability_display": format_probability_display(raw_explain["calibrated_p_high"]),
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
        raise ValueError("synopsis_text is required in pitch to find comparable dramas.")

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
        "caveat": "comparables show similar premises; their tiers are not a probability for this pitch",
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
        "caveat": "comparables show similar premises; their tiers are not a probability for this pitch",
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
        empty_quotable: WhatIfQuotable = {
            "before": {
                "likelihood_band": before_state["likelihood_band"],
                "observed_band_rate": TEST_BAND_OBSERVED_RATES[before_state["likelihood_band"]],
                "probability_display": format_probability_display(before_state["calibrated_p_high"]),
            },
            "after": {
                "likelihood_band": before_state["likelihood_band"],
                "observed_band_rate": TEST_BAND_OBSERVED_RATES[before_state["likelihood_band"]],
                "probability_display": format_probability_display(before_state["calibrated_p_high"]),
            },
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
            "caveat": "Results are associational, not causal; changing attachments does not guarantee audience reach shifts.",
            "warnings": before_warnings,
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
        typical_dict = COUNTRY_TYPICAL_VALUES.get(country, {"network": "Other", "content_rating": "15+"})

        control_pitch = copy.deepcopy(base_pitch_to_score)
        typical_used = {}
        for lever in blank_filled_levers:
            typ_val = typical_dict.get(lever, "Other")
            control_pitch[lever] = typ_val
            typical_used[lever] = typ_val

        control_state, _ = _score_pitch(control_pitch, artifacts)

        typ_delta: WhatIfDeltas = {
            "raw_p_high_delta": float(round(control_state["raw_p_high"] - before_state["raw_p_high"], 4)),
            "calibrated_p_high_delta": float(round(control_state["calibrated_p_high"] - before_state["calibrated_p_high"], 4)),
        }
        spec_delta: WhatIfDeltas = {
            "raw_p_high_delta": float(round(after_state["raw_p_high"] - control_state["raw_p_high"], 4)),
            "calibrated_p_high_delta": float(round(after_state["calibrated_p_high"] - control_state["calibrated_p_high"], 4)),
        }

        decomposition = {
            "filled_fields": blank_filled_levers,
            "typical_values_used": typical_used,
            "typical_value_control_delta": typ_delta,
            "specific_choice_delta": spec_delta,
        }
        combined_warnings.append("part of this change reflects the model's lower P(high) when a field is blank")

    quotable: WhatIfQuotable = {
        "before": {
            "likelihood_band": before_state["likelihood_band"],
            "observed_band_rate": TEST_BAND_OBSERVED_RATES[before_state["likelihood_band"]],
            "probability_display": format_probability_display(before_state["calibrated_p_high"]),
        },
        "after": {
            "likelihood_band": after_state["likelihood_band"],
            "observed_band_rate": TEST_BAND_OBSERVED_RATES[after_state["likelihood_band"]],
            "probability_display": format_probability_display(after_state["calibrated_p_high"]),
        },
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
        "caveat": "Results are associational, not causal; changing attachments does not guarantee audience reach shifts.",
        "warnings": list(dict.fromkeys(combined_warnings)),
    }

    json.dumps(output)
    return output
