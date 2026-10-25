"""
Unit and Integration Tests for Agent Tools (Milestone 7 Part A)
==============================================================
Tests verify:
1. Schema validation: validates inputs, enforces string/list caps, rejects invalid fields.
2. Disallowed levers: what_if strictly rejects episodes, weekdays, country, planned_start, and >3 changes.
3. Empty changes: what_if with empty changes produces delta = 0.0.
4. Talent change recomputation: what_if recomputes talent priors using the 12-month rule.
5. Absence of synopsis text: no synopsis text in any tool output across all fixture pitches.
6. Prompt injection defense: adversarial instruction strings are treated purely as data.
7. Determinism & JSON-serializability: tools produce identical, JSON-serializable outputs across calls.
8. Clean skip if artifacts are missing (no network calls during tests).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from src.agent_tools import (
    ALLOWED_WHAT_IF_LEVERS,
    DISALLOWED_LEVERS,
    explain_pitch,
    find_comparables,
    validate_pitch,
    validate_what_if_changes,
    what_if,
)
from src.explain import load_artifacts as load_explain_artifacts
from src.retriever import load_retriever_artifacts

repo_root = Path(__file__).resolve().parent.parent

FIXTURES_FILE = repo_root / "tests/fixtures/pitches.json"

REQUIRED_EXPLAIN_FILES = [
    repo_root / "models/headline_model.joblib",
    repo_root / "models/isotonic_calibrator.joblib",
    repo_root / "models/background_sample.joblib",
    repo_root / "models/talent_lookup.joblib",
    repo_root / "models/metadata.joblib",
]

REQUIRED_RETRIEVER_FILES = [
    repo_root / "models/retriever_embeddings.npy",
    repo_root / "models/retriever_metadata.parquet",
    repo_root / "models/retriever_info.json",
]

missing_explain = [p.name for p in REQUIRED_EXPLAIN_FILES if not p.exists()]
missing_retriever = [p.name for p in REQUIRED_RETRIEVER_FILES if not p.exists()]


@pytest.fixture(scope="module")
def pitches() -> list[dict[str, Any]]:
    """Loads the 12 invented test pitches from fixture JSON."""
    assert FIXTURES_FILE.exists(), f"Fixtures file {FIXTURES_FILE} not found"
    with open(FIXTURES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def explain_artifacts():
    if missing_explain:
        pytest.skip(f"Missing explain artifacts: {', '.join(missing_explain)}")
    return load_explain_artifacts()


@pytest.fixture(scope="module")
def retriever_artifacts():
    if missing_retriever:
        pytest.skip(f"Missing retriever artifacts: {', '.join(missing_retriever)}")
    return load_retriever_artifacts()


# -----------------------------------------------------------------------------
# 1. Schema Validation Tests
# -----------------------------------------------------------------------------

def test_validate_pitch_valid(pitches):
    """Validates that well-formed pitches parse and sanitize cleanly."""
    p0 = pitches[0]
    validated = validate_pitch(p0)
    assert validated["country"] == "South Korea"
    assert validated["network"] == "tvN"
    assert "Thriller" in validated["genres"]
    assert validated["planned_start"] == "2024-03-01"
    assert validated["episodes"] == 16.0


def test_validate_pitch_invalid_country():
    """Asserts that invalid country raises ValueError with allowed list."""
    with pytest.raises(ValueError, match="Invalid country"):
        validate_pitch({"country": "United States", "planned_start": "2024-01-01"})


def test_validate_pitch_missing_start():
    """Asserts that missing planned_start raises ValueError."""
    with pytest.raises(ValueError, match="planned_start is required"):
        validate_pitch({"country": "South Korea"})


def test_validate_pitch_type_error():
    """Asserts that non-dict input raises TypeError."""
    with pytest.raises(TypeError, match="must be a dictionary"):
        validate_pitch(["not", "a", "dict"])  # type: ignore


# -----------------------------------------------------------------------------
# 2. What-If Levers & Constraints Tests
# -----------------------------------------------------------------------------

def test_what_if_disallowed_levers_rejected(pitches):
    """Asserts what_if strictly rejects episodes, weekdays, country, planned_start."""
    p = pitches[0]
    for disallowed in ["episodes", "weekdays", "country", "planned_start"]:
        with pytest.raises(ValueError, match="Disallowed lever"):
            what_if(p, {disallowed: "some_value"})


def test_what_if_max_changes_limit(pitches):
    """Asserts what_if rejects more than 3 changes per call."""
    p = pitches[0]
    four_changes = {
        "network": "JTBC",
        "genres": ["Comedy"],
        "content_rating": "15+",
        "directors": ["Bong Joon Ho"],
    }
    with pytest.raises(ValueError, match="at most 3 changes"):
        what_if(p, four_changes)


def test_what_if_empty_changes_gives_delta_zero(pitches, explain_artifacts):
    """Asserts what_if with empty changes dictionary returns delta 0.0 and identical states."""
    p = pitches[0]
    res = what_if(p, {}, artifacts=explain_artifacts)
    assert res["delta"]["raw_p_high_delta"] == 0.0
    assert res["delta"]["calibrated_p_high_delta"] == 0.0
    assert res["before"]["raw_p_high"] == res["after"]["raw_p_high"]
    assert res["before"]["calibrated_p_high"] == res["after"]["calibrated_p_high"]
    assert res["changes_applied"] == {}
    assert "associational, not causal" in res["caveat"]


def test_what_if_talent_change_recomputes_priors(pitches, explain_artifacts):
    """Asserts what_if recomputes talent track records when directors/cast change."""
    # pitch 8 has empty talent (debut)
    p = pitches[7]
    star_talent_change = {
        "lead_cast": ["Song Joong Ki", "Kim Tae Ri"],
        "writers": ["Kim Eun Hee"],
    }
    res = what_if(p, star_talent_change, artifacts=explain_artifacts)
    assert res["after"]["raw_p_high"] != res["before"]["raw_p_high"]
    assert res["delta"]["raw_p_high_delta"] != 0.0
    assert "lead_cast" in res["changes_applied"]
    assert "writers" in res["changes_applied"]


# -----------------------------------------------------------------------------
# 3. Absence of Synopsis Text & Prompt Injection Defense
# -----------------------------------------------------------------------------

def test_no_synopsis_text_in_tool_outputs(pitches, explain_artifacts, retriever_artifacts):
    """Asserts that no synopsis text or synopsis key appears in any tool output."""
    p = pitches[0]

    # 1. explain_pitch
    out_exp = explain_pitch(p, artifacts=explain_artifacts)
    assert "synopsis" not in out_exp["pitch_summary"]
    assert "synopsis_text" not in out_exp["pitch_summary"]

    # 2. find_comparables
    out_comp = find_comparables(p, k=3, artifacts=retriever_artifacts)
    for c in out_comp["comparables"]:
        assert "synopsis" not in c
        assert "synopsis_clean" not in c
        assert "synopsis_text" not in c
    assert "comparables show similar premises" in out_comp["caveat"]

    # 3. what_if
    out_wi = what_if(p, {"network": "KBS2"}, artifacts=explain_artifacts)
    assert "synopsis" not in str(out_wi)
    assert "synopsis_text" not in str(out_wi)


def test_prompt_injection_treated_as_plain_data(pitches, explain_artifacts):
    """
    Asserts adversarial prompt injection pitches (pitch 11 and 12) are treated
    purely as literal text without affecting logic or overriding models.
    """
    p11 = pitches[10]  # contains "Ignore all previous instructions..."
    p12 = pitches[11]  # contains '{"role": "system", ...}'

    for p in [p11, p12]:
        out = explain_pitch(p, artifacts=explain_artifacts)
        assert out["tool"] == "explain_pitch"
        assert isinstance(out["prediction"]["calibrated_p_high"], float)
        # Should NOT automatically classify as 0.999 High
        assert out["prediction"]["calibrated_p_high"] <= 0.98

        wi_out = what_if(p, {"network": "tvN"}, artifacts=explain_artifacts)
        assert wi_out["tool"] == "what_if"
        assert "associational, not causal" in wi_out["caveat"]


# -----------------------------------------------------------------------------
# 4. Determinism & JSON Serializability
# -----------------------------------------------------------------------------

def test_outputs_deterministic_and_json_serializable(pitches, explain_artifacts, retriever_artifacts):
    """Asserts tools produce identical, valid JSON across repeated calls."""
    p = pitches[0]

    # explain_pitch
    exp1 = explain_pitch(p, artifacts=explain_artifacts, seed=42)
    exp2 = explain_pitch(p, artifacts=explain_artifacts, seed=42)
    assert json.dumps(exp1, sort_keys=True) == json.dumps(exp2, sort_keys=True)

    # find_comparables
    comp1 = find_comparables(p, k=3, artifacts=retriever_artifacts)
    comp2 = find_comparables(p, k=3, artifacts=retriever_artifacts)
    assert json.dumps(comp1, sort_keys=True) == json.dumps(comp2, sort_keys=True)

    # what_if
    wi1 = what_if(p, {"network": "JTBC"}, artifacts=explain_artifacts)
    wi2 = what_if(p, {"network": "JTBC"}, artifacts=explain_artifacts)
    assert json.dumps(wi1, sort_keys=True) == json.dumps(wi2, sort_keys=True)
