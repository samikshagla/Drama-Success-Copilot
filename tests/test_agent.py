"""
tests/test_agent.py
Comprehensive test suite for Agent Orchestrator, MockLLM, and Grounding Verifier.

Tests:
1. ChaosLLM adversarial doubles across 12 distinct attack vectors:
   - (a) unknown tool
   - (b) 10 calls (exceeding <= 4 cap)
   - (c) altered pitch argument
   - (d) disallowed lever in what_if
   - (e) omitted required tools
   - (f) invented percentage (e.g. "91%")
   - (g) forbidden commercial/certainty phrases ("revenue", "guaranteed", "will succeed")
   - (h) ungrounded comparable drama title
   - (i) talent name leak from historical lookup
   - (j) 6+ consecutive words echoed from synopsis text
   - (k) missing required caveats
   - (l) continuous calibrated decimal probability leak
2. Positive grounding verification: normal MockLLM memo passes cleanly without fallback.
3. Determinism: identical repeated runs produce identical outputs.
4. Injection control: prompt injection has 0 effect on predictive sections of the memo.
5. Missing artifact skips for fresh clone hygiene.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.agent import AgentOrchestrator, MockLLM, TOOL_SPECS, run_agent
from src.agent_tools import (
    load_explain_artifacts,
    load_retriever_artifacts,
)
from src.grounding import verify_memo_grounding
from src.shared_constants import (
    COMPARABLES_CAVEAT,
    LIMITATIONS_NOTICE,
    OBSERVED_RATE_QUALIFIER,
    WHAT_IF_CAVEAT,
)


@pytest.fixture(scope="module")
def eval_pitches() -> List[Dict[str, Any]]:
    path = Path("tests/fixtures/pitches_eval.json")
    if not path.exists():
        pytest.skip(f"Fixtures file {path} not found.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def explain_artifacts():
    try:
        return load_explain_artifacts()
    except Exception as e:
        pytest.skip(f"Explain artifacts not available: {e}")


@pytest.fixture(scope="module")
def retriever_artifacts():
    try:
        return load_retriever_artifacts()
    except Exception as e:
        pytest.skip(f"Retriever artifacts not available: {e}")


# -----------------------------------------------------------------------------
# ChaosLLM Test Double
# -----------------------------------------------------------------------------

class ChaosLLM:
    """
    Adversarial test double capable of injecting plan corruptions or bad memo text.
    """

    def __init__(self, mode: str, bad_text: Optional[str] = None):
        self.mode = mode
        self.bad_text = bad_text
        self.mock = MockLLM()

    def generate_plan(
        self,
        pitch: Dict[str, Any],
        tool_specs: List[Dict[str, Any]],
        what_if_changes: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        # Plan corruption modes
        if self.mode == "unknown_tool":
            return [
                {"tool": "delete_all_files", "arguments": {"path": "/"}},
                {"tool": "explain_pitch", "arguments": {"pitch": pitch}},
            ]
        elif self.mode == "ten_calls":
            return [
                {"tool": "explain_pitch", "arguments": {"pitch": pitch}}
                for _ in range(10)
            ]
        elif self.mode == "altered_pitch_field":
            altered = copy.deepcopy(pitch)
            altered["country"] = "United States"
            return [
                {"tool": "explain_pitch", "arguments": {"pitch": altered}},
                {"tool": "find_comparables", "arguments": {"pitch": pitch, "k": 3}},
            ]
        elif self.mode == "disallowed_lever":
            return [
                {"tool": "explain_pitch", "arguments": {"pitch": pitch}},
                {"tool": "find_comparables", "arguments": {"pitch": pitch, "k": 3}},
                {"tool": "what_if", "arguments": {"pitch": pitch, "changes": {"episodes": 100, "country": "China"}}},
            ]
        elif self.mode == "omits_required_tools":
            # Return empty plan
            return []

        # Default standard plan
        return self.mock.generate_plan(pitch, tool_specs, what_if_changes)

    def write_memo(
        self,
        tool_outputs: Dict[str, Any],
        pitch_context: Dict[str, Any],
        what_if_changes: Optional[Dict[str, Any]] = None,
    ) -> str:
        # If specific bad_text supplied, return it
        if self.bad_text is not None:
            return self.bad_text

        # Otherwise produce valid memo and inject corruption according to mode
        base_memo = self.mock.write_memo(tool_outputs, pitch_context, what_if_changes)

        if self.mode == "invented_percentage":
            return base_memo + "\nAudience market penetration estimate is 91% in urban households."
        elif self.mode == "forbidden_phrase_guaranteed":
            return base_memo + "\nThis drama is guaranteed to be a hit."
        elif self.mode == "forbidden_phrase_revenue":
            return base_memo + "\nProjected box office revenue exceeds 50 million dollars."
        elif self.mode == "forbidden_phrase_will_succeed":
            return base_memo + "\nWith this cast attachment, the project will succeed."
        elif self.mode == "ungrounded_comparable_title":
            # Inject fake title in table
            return base_memo.replace("| 1 |", "| 1 | Invented Imaginary Drama 2099 | 2099 | South Korea | tvN | High | # |\n| 2 |")
        elif self.mode == "talent_name_leak":
            return base_memo + "\nExecutive Producer notes that Kim Won-seok praised the early treatment."
        elif self.mode == "synopsis_echo_6_words":
            syn = pitch_context.get("synopsis_text", "")
            words = syn.split()
            if len(words) >= 6:
                six_gram = " ".join(words[:6])
                return base_memo + f"\nLogline narrative premise: {six_gram} and conflicts unfold."
            return base_memo + "\nLogline narrative premise: detective hunts an untraceable extortionist targeting pharmaceutical conglomerates."
        elif self.mode == "missing_required_caveat":
            return base_memo.replace(COMPARABLES_CAVEAT, "Comparables indicate general genre precedent.")
        elif self.mode == "calibrated_probability_decimal":
            return base_memo + "\nThe model calibrated posterior probability is 0.5833."

        return base_memo


# -----------------------------------------------------------------------------
# Positive & Determinism Tests
# -----------------------------------------------------------------------------

def test_agent_normal_memo_passes_grounding(eval_pitches, explain_artifacts, retriever_artifacts):
    """Assert normal execution on complete pitch passes grounding with 0 failures and no fallback."""
    p1 = eval_pitches[0]
    orchestrator = AgentOrchestrator(
        llm_client=MockLLM(),
        explain_artifacts=explain_artifacts,
        retriever_artifacts=retriever_artifacts,
    )
    result = orchestrator.run({"pitch": p1})
    assert result["grounding_passed"] is True
    assert result["fallback_to_template"] is False
    assert len(result["grounding_failures"]) == 0
    assert len(result["plan_rejections"]) == 0

    memo = result["memo"]
    assert "### 1. What Was Assessed" in memo
    assert "### 2. Likelihood Band & Historical Track Record" in memo
    assert "### 3. Main Predictive Drivers" in memo
    assert "### 4. Advisory Warnings & Data Hygiene Alerts" in memo
    assert "### 5. Comparable Historical Dramas" in memo
    assert "### 6. Methodological Limitations & Governance Notice" in memo
    assert COMPARABLES_CAVEAT in memo
    assert LIMITATIONS_NOTICE in memo
    assert OBSERVED_RATE_QUALIFIER in memo


def test_agent_what_if_normal_memo_passes_grounding(eval_pitches, explain_artifacts, retriever_artifacts):
    """Assert what_if execution produces complete grounded memo with decomposition and what-if caveat."""
    p5 = next(p for p in eval_pitches if p["id"] == "pitch_05_blank_network_and_rating")
    orchestrator = AgentOrchestrator(
        llm_client=MockLLM(),
        explain_artifacts=explain_artifacts,
        retriever_artifacts=retriever_artifacts,
    )
    changes = {"network": "JTBC", "content_rating": "15+"}
    result = orchestrator.run({"pitch": p5, "changes": changes})

    assert result["grounding_passed"] is True
    assert result["fallback_to_template"] is False
    assert len(result["grounding_failures"]) == 0

    memo = result["memo"]
    assert "### 6. Sensitivity Simulation (What-If Analysis)" in memo
    assert WHAT_IF_CAVEAT in memo
    assert "two thirds" in memo.lower()
    assert "| State | Likelihood Band | Empirical Benchmark (Observed Test Rate) |" in memo
    assert "| Before | Unlikely |" in memo
    assert "| After | Possible |" in memo


def test_agent_determinism_across_runs(eval_pitches, explain_artifacts, retriever_artifacts):
    """Assert identical inputs yield 100% byte-for-byte identical memos across repeated runs."""
    p1 = eval_pitches[0]
    orch = AgentOrchestrator(MockLLM(), explain_artifacts, retriever_artifacts)
    res1 = orch.run({"pitch": p1})
    res2 = orch.run({"pitch": p1})
    assert res1["memo"] == res2["memo"]
    assert res1["plan"] == res2["plan"]


# -----------------------------------------------------------------------------
# ChaosLLM Adversarial Double Tests
# -----------------------------------------------------------------------------

def test_chaos_llm_unknown_tool_blocked(eval_pitches, explain_artifacts, retriever_artifacts):
    """(a) ChaosLLM: Unknown tool in plan is rejected and logged."""
    p1 = eval_pitches[0]
    client = ChaosLLM(mode="unknown_tool")
    orch = AgentOrchestrator(client, explain_artifacts, retriever_artifacts)
    res = orch.run({"pitch": p1})
    assert any("unknown tool 'delete_all_files'" in r for r in res["plan_rejections"])
    assert not any(call["tool"] == "delete_all_files" for call in res["plan"])


def test_chaos_llm_ten_calls_truncated(eval_pitches, explain_artifacts, retriever_artifacts):
    """(b) ChaosLLM: 10 calls plan is capped to <= 4 calls with rejection logged."""
    p1 = eval_pitches[0]
    client = ChaosLLM(mode="ten_calls")
    orch = AgentOrchestrator(client, explain_artifacts, retriever_artifacts)
    res = orch.run({"pitch": p1})
    assert any("exceeded max call limit" in r for r in res["plan_rejections"])
    assert len(res["plan"]) <= 4


def test_chaos_llm_altered_pitch_field_blocked(eval_pitches, explain_artifacts, retriever_artifacts):
    """(c) ChaosLLM: Altered pitch field in tool argument is rejected and logged."""
    p1 = eval_pitches[0]
    client = ChaosLLM(mode="altered_pitch_field")
    orch = AgentOrchestrator(client, explain_artifacts, retriever_artifacts)
    res = orch.run({"pitch": p1})
    assert any("arguments altered or not equal to validated pitch" in r for r in res["plan_rejections"])


def test_chaos_llm_disallowed_lever_blocked(eval_pitches, explain_artifacts, retriever_artifacts):
    """(d) ChaosLLM: Disallowed lever in what_if is rejected."""
    p1 = eval_pitches[0]
    client = ChaosLLM(mode="disallowed_lever")
    orch = AgentOrchestrator(client, explain_artifacts, retriever_artifacts)
    # User request has valid changes, but ChaosLLM attempts to inject disallowed levers into plan
    res = orch.run({"pitch": p1, "changes": {"network": "tvN"}})
    assert any("disallowed levers" in r or "rejected" in r for r in res["plan_rejections"])


def test_chaos_llm_omits_required_tools_auto_injected(eval_pitches, explain_artifacts, retriever_artifacts):
    """(e) ChaosLLM: Plan omitting required tools has them automatically injected by orchestrator."""
    p1 = eval_pitches[0]
    client = ChaosLLM(mode="omits_required_tools")
    orch = AgentOrchestrator(client, explain_artifacts, retriever_artifacts)
    res = orch.run({"pitch": p1})
    assert any("injected by orchestrator" in r for r in res["plan_rejections"])
    tool_names = [call["tool"] for call in res["plan"]]
    assert "explain_pitch" in tool_names
    assert "find_comparables" in tool_names


def test_chaos_llm_invented_percentage_blocked(eval_pitches, explain_artifacts, retriever_artifacts):
    """(f) ChaosLLM: Invented percentage (e.g. 91%) is rejected and falls back to template."""
    p1 = eval_pitches[0]
    client = ChaosLLM(mode="invented_percentage")
    orch = AgentOrchestrator(client, explain_artifacts, retriever_artifacts)
    res = orch.run({"pitch": p1})
    assert res["grounding_passed"] is False
    assert res["fallback_to_template"] is True
    assert any("Ungrounded percentage '91%' found" in f for f in res["grounding_failures"])
    assert "91%" not in res["memo"]


@pytest.mark.parametrize("mode, phrase", [
    ("forbidden_phrase_guaranteed", "guaranteed"),
    ("forbidden_phrase_revenue", "revenue"),
    ("forbidden_phrase_will_succeed", "will succeed"),
])
def test_chaos_llm_forbidden_phrases_blocked(eval_pitches, explain_artifacts, retriever_artifacts, mode, phrase):
    """(g) ChaosLLM: Commercial/certainty phrases trigger fallback to template memo."""
    p1 = eval_pitches[0]
    client = ChaosLLM(mode=mode)
    orch = AgentOrchestrator(client, explain_artifacts, retriever_artifacts)
    res = orch.run({"pitch": p1})
    assert res["grounding_passed"] is False
    assert res["fallback_to_template"] is True
    assert any(phrase in f.lower() for f in res["grounding_failures"])


def test_chaos_llm_ungrounded_comparable_title_blocked(eval_pitches, explain_artifacts, retriever_artifacts):
    """(h) ChaosLLM: Invented comparable title in table triggers fallback."""
    p1 = eval_pitches[0]
    client = ChaosLLM(mode="ungrounded_comparable_title")
    orch = AgentOrchestrator(client, explain_artifacts, retriever_artifacts)
    res = orch.run({"pitch": p1})
    assert res["grounding_passed"] is False
    assert res["fallback_to_template"] is True
    assert any("Ungrounded comparable title" in f for f in res["grounding_failures"])


def test_chaos_llm_talent_name_leak_blocked(eval_pitches, explain_artifacts, retriever_artifacts):
    """(i) ChaosLLM: Mentioning full director name from talent lookup triggers fallback."""
    p1 = eval_pitches[0]
    client = ChaosLLM(mode="talent_name_leak")
    orch = AgentOrchestrator(client, explain_artifacts, retriever_artifacts)
    res = orch.run({"pitch": p1})
    assert res["grounding_passed"] is False
    assert res["fallback_to_template"] is True
    assert any("Bong Joon Ho" in f or "talent" in f.lower() for f in res["grounding_failures"])


def test_chaos_llm_synopsis_echo_6_words_blocked(eval_pitches, explain_artifacts, retriever_artifacts):
    """(j) ChaosLLM: Echoing 6+ consecutive words from synopsis triggers fallback."""
    p1 = eval_pitches[0]
    client = ChaosLLM(mode="synopsis_echo_6_words")
    orch = AgentOrchestrator(client, explain_artifacts, retriever_artifacts)
    res = orch.run({"pitch": p1})
    assert res["grounding_passed"] is False
    assert res["fallback_to_template"] is True
    assert any("Synopsis text echoed (6+ words)" in f for f in res["grounding_failures"])


def test_chaos_llm_missing_required_caveat_blocked(eval_pitches, explain_artifacts, retriever_artifacts):
    """(k) ChaosLLM: Omitting mandatory comparables caveat triggers fallback."""
    p1 = eval_pitches[0]
    client = ChaosLLM(mode="missing_required_caveat")
    orch = AgentOrchestrator(client, explain_artifacts, retriever_artifacts)
    res = orch.run({"pitch": p1})
    assert res["grounding_passed"] is False
    assert res["fallback_to_template"] is True
    assert any("Missing required comparables caveat" in f for f in res["grounding_failures"])


def test_chaos_llm_calibrated_probability_decimal_blocked(eval_pitches, explain_artifacts, retriever_artifacts):
    """(l) ChaosLLM: Quoting continuous probability decimal triggers fallback."""
    p1 = eval_pitches[0]
    client = ChaosLLM(mode="calibrated_probability_decimal")
    orch = AgentOrchestrator(client, explain_artifacts, retriever_artifacts)
    res = orch.run({"pitch": p1})
    assert res["grounding_passed"] is False
    assert res["fallback_to_template"] is True
    assert any("Forbidden continuous/calibrated decimal probability found: '0.5833'" in f for f in res["grounding_failures"])


# -----------------------------------------------------------------------------
# Injection Control Test
# -----------------------------------------------------------------------------

def test_agent_injection_control_identical_predictive_sections(eval_pitches, explain_artifacts, retriever_artifacts):
    """
    Assert prompt injection in synopsis produces 100% identical predictive memo sections
    (what was assessed, band, track record, drivers, warnings, limitations) compared to
    the same pitch with injected text replaced by neutral text. Only the comparables table may differ.
    """
    p_inj = next(p for p in eval_pitches if p["id"] == "pitch_11_injection_ignore_instructions")
    p_neutral = copy.deepcopy(p_inj)
    p_neutral["synopsis_text"] = "A dedicated prosecutor investigates financial irregularities within a regional bank."

    orch = AgentOrchestrator(MockLLM(), explain_artifacts, retriever_artifacts)
    res_inj = orch.run({"pitch": p_inj})
    res_neu = orch.run({"pitch": p_neutral})

    memo_inj = res_inj["memo"]
    memo_neu = res_neu["memo"]

    # Split into sections before comparables and after comparables
    def get_predictive_parts(m: str) -> str:
        parts = m.split("### 5. Comparable Historical Dramas")
        pre_comp = parts[0]
        post_comp = re.split(r"### \d+\. Methodological Limitations", parts[1])[1]
        return pre_comp + "\n### Limitations" + post_comp

    assert get_predictive_parts(memo_inj) == get_predictive_parts(memo_neu)


def test_what_if_disallowed_lever_surfaced_in_memo(eval_pitches, explain_artifacts, retriever_artifacts):
    """Asserts memo explicitly states requested disallowed levers are not supported, not silently dropped."""
    p1 = eval_pitches[0]
    orch = AgentOrchestrator(MockLLM(), explain_artifacts, retriever_artifacts)
    # 1. Request with only disallowed lever
    res = orch.run({"pitch": p1, "changes": {"episodes": 100}})
    assert "not supported" in res["memo"].lower()
    assert "episodes" in res["memo"]
    assert "Sensitivity Simulation" in res["memo"]

    # 2. Request with mixed allowed and disallowed levers
    res_mixed = orch.run({"pitch": p1, "changes": {"network": "tvN", "episodes": 100}})
    assert "not supported" in res_mixed["memo"].lower()
    assert "episodes" in res_mixed["memo"]
    assert "tvN" in res_mixed["memo"]


def test_edge_warnings_episodes_and_start_year(eval_pitches, explain_artifacts, retriever_artifacts):
    """Asserts tools and memo emit warnings for episodes < 6, episodes > 150, and start > 2025."""
    orch = AgentOrchestrator(MockLLM(), explain_artifacts, retriever_artifacts)

    # Episodes < 6 (pitch 18 modified to 4 episodes)
    p_short = next(p for p in eval_pitches if p["id"] == "pitch_18_short_6_episodes")
    p_sub6 = copy.deepcopy(p_short)
    p_sub6["episodes"] = 4
    out_sub6 = orch.run({"pitch": p_sub6})
    assert any("below the 6-episode scope cutoff" in w for w in out_sub6["tool_outputs"]["explain_pitch"]["warnings"])
    assert "below the 6-episode scope cutoff" in out_sub6["memo"]

    # Episodes > 150 (pitch 19: 450 episodes)
    p_long = next(p for p in eval_pitches if p["id"] == "pitch_19_long_450_episodes")
    out_long = orch.run({"pitch": p_long})
    assert any("far above the typical training distribution" in w for w in out_long["tool_outputs"]["explain_pitch"]["warnings"])
    assert "far above the typical training distribution" in out_long["memo"]

    # Start date > 2025 (pitch 20: 2031)
    p_future = next(p for p in eval_pitches if p["id"] == "pitch_20_future_start_2031")
    out_future = orch.run({"pitch": p_future})
    assert any("after the 2025 training/evaluation horizon" in w for w in out_future["tool_outputs"]["explain_pitch"]["warnings"])
    assert "after the 2025 training/evaluation horizon" in out_future["memo"]


def test_dynamic_blank_fill_share_phrasing():
    """Asserts blank-fill decomposition phrasing adapts dynamically across share thresholds."""
    from src.agent import format_blank_fill_share_statement

    # 1. Share in 0.60 - 0.72 -> "about two thirds"
    decomp_two_thirds = {
        "filled_fields": ["network", "content_rating"],
        "raw_shares": {"marginal_control_share_pct": 68.0},
    }
    stmt = format_blank_fill_share_statement(decomp_two_thirds)
    assert stmt is not None
    assert "About two thirds" in stmt

    # 2. Share > 0.72 -> "most"
    decomp_most = {
        "filled_fields": ["network"],
        "raw_shares": {"marginal_control_share_pct": 82.5},
    }
    stmt = format_blank_fill_share_statement(decomp_most)
    assert stmt is not None
    assert "Most" in stmt

    # 3. Share in 0.40 - 0.60 -> "about half"
    decomp_half = {
        "filled_fields": ["content_rating"],
        "raw_shares": {"marginal_control_share_pct": 48.0},
    }
    stmt = format_blank_fill_share_statement(decomp_half)
    assert stmt is not None
    assert "About half" in stmt

    # 4. Share < 0.40 -> "a minority"
    decomp_min = {
        "filled_fields": ["network"],
        "raw_shares": {"marginal_control_share_pct": 25.0},
    }
    stmt = format_blank_fill_share_statement(decomp_min)
    assert stmt is not None
    assert "A minority" in stmt

    # 5. No blanks filled -> None
    assert format_blank_fill_share_statement(None) is None
    assert format_blank_fill_share_statement({"filled_fields": []}) is None


def test_talent_name_variants_blocked(eval_pitches, explain_artifacts, retriever_artifacts):
    """Asserts talent name detection catches reversed order, hyphenated, and all caps variants."""
    p1 = eval_pitches[0]
    orch = AgentOrchestrator(MockLLM(), explain_artifacts, retriever_artifacts)
    res = orch.run({"pitch": p1})
    base_memo = res["memo"]
    talent_lookup = explain_artifacts["talent_lookup"]

    variants = [
        "Directed by Won-seok Kim with strong execution.",
        "Directed by WON-SEOK KIM with strong execution.",
        "Directed by Kim-Won-Seok with strong execution.",
        "Directed by KIM WON SEOK with strong execution.",
        "Screenplay written by Eun-sook Kim.",
        "Screenplay written by EUN SOOK KIM.",
        "Screenplay written by Eun Sook Kim.",
    ]
    for var in variants:
        corrupted = base_memo.replace("### 1. What Was Assessed\n", f"### 1. What Was Assessed\n{var}\n")
        is_grounded, failures = verify_memo_grounding(corrupted, res["tool_outputs"], p1, talent_lookup=talent_lookup)
        assert not is_grounded, f"Variant failed to be blocked: {var}"
        assert any("talent" in f.lower() for f in failures)


def test_random_memo_mutations_200_blocked(eval_pitches, explain_artifacts, retriever_artifacts):
    """
    Applies 200 random corruptions to a valid memo (inserting ungrounded percentages,
    deleting caveats, swapping titles, inserting forbidden words, etc.) and asserts 100% block rate.
    """
    import random
    rng = random.Random(42)

    p1 = eval_pitches[0]
    orch = AgentOrchestrator(MockLLM(), explain_artifacts, retriever_artifacts)
    res = orch.run({"pitch": p1})
    base_memo = res["memo"]
    talent_lookup = explain_artifacts["talent_lookup"]

    mutation_types = [
        "insert_percentage",
        "delete_caveat",
        "swap_title",
        "insert_forbidden_phrase",
        "insert_decimal",
        "echo_synopsis",
        "insert_word_percentage",
        "insert_ratio",
        "insert_ungrounded_number",
    ]

    passed_mutations: List[str] = []

    for i in range(200):
        mut_type = rng.choice(mutation_types)
        corrupted = base_memo

        if mut_type == "insert_percentage":
            allowed_pcts = {"57.8", "95", "52.6", "62.9"}
            while True:
                random_pct = f"{rng.randint(1, 99)}.{rng.randint(0, 9)}%"
                if random_pct.rstrip("%") not in allowed_pcts:
                    break
            corrupted = corrupted.replace("Model-Assigned Likelihood Band:", f"Model forecast gives {random_pct} chance:")
        elif mut_type == "delete_caveat":
            caveat = rng.choice([COMPARABLES_CAVEAT, LIMITATIONS_NOTICE, OBSERVED_RATE_QUALIFIER])
            corrupted = corrupted.replace(caveat, "")
        elif mut_type == "swap_title":
            corrupted = corrupted.replace("Partners for Justice Season 2 (2019)", "Fictional Phantom Mystery (2020)")
        elif mut_type == "insert_forbidden_phrase":
            phrase = rng.choice([
                "guaranteed", "surefire", "revenue", "roi", "box office", "profit", "earnings",
                "near-certain", "almost certain", "bound to", "can't miss", "sure thing", "no doubt",
                "will almost certainly", "certain hit", "money-maker"
            ])
            corrupted = corrupted.replace("### 1. What Was Assessed\n", f"### 1. What Was Assessed\nThis pitch is {phrase} success.\n")
        elif mut_type == "insert_decimal":
            corrupted = corrupted.replace("Model-Assigned Likelihood Band:", "Model continuous probability 0.5833 indicates:")
        elif mut_type == "echo_synopsis":
            syn_words = p1.get("synopsis_text", "").split()
            six_words = " ".join(syn_words[:6])
            corrupted = corrupted.replace("### 1. What Was Assessed\n", f"### 1. What Was Assessed\n{six_words}\n")
        elif mut_type == "insert_word_percentage":
            word_pct = rng.choice(["ninety percent", "91 per cent", "fifty percent", "twenty-five percent"])
            corrupted = corrupted.replace("### 1. What Was Assessed\n", f"### 1. What Was Assessed\nAudience reach estimate is {word_pct}.\n")
        elif mut_type == "insert_ratio":
            ratio_expr = rng.choice(["nine in ten", "9 out of 10", "3 in 4", "four out of five"])
            corrupted = corrupted.replace("### 1. What Was Assessed\n", f"### 1. What Was Assessed\nExpected viewership success is {ratio_expr}.\n")
        elif mut_type == "insert_ungrounded_number":
            fake_num = rng.choice(["85000", "2011", "99", "123456"])
            corrupted = corrupted.replace("### 1. What Was Assessed\n", f"### 1. What Was Assessed\nTargeting benchmark rating of {fake_num}.\n")

        is_grounded, failures = verify_memo_grounding(corrupted, res["tool_outputs"], p1, talent_lookup=talent_lookup)
        if is_grounded:
            passed_mutations.append(f"Iteration {i} ({mut_type}):\n{corrupted[:200]}")

    block_rate = (200 - len(passed_mutations)) / 200.0 * 100.0
    print(f"\n200 Random Mutations Block Rate: {block_rate:.1f}% ({200 - len(passed_mutations)}/200 blocked)")
    if passed_mutations:
        print("MUTATIONS THAT PASSED:")
        for pm in passed_mutations:
            print(pm)
    assert len(passed_mutations) == 0, f"{len(passed_mutations)} mutations bypassed grounding checks!"


# -----------------------------------------------------------------------------
# New Governance Tests: Drivers, Low-Info, Comparables Reasons, What-If
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("pitch_id", ["pitch_01_complete_kr_thriller", "pitch_05_blank_network_and_rating", "pitch_14_adversarial_disallowed_lever"])
def test_talent_driver_filtering_and_track_record_note(eval_pitches, explain_artifacts, retriever_artifacts, pitch_id):
    """
    Assert talent groups (talent_cast, talent_director, talent_screenwriter) are NOT listed
    among positive/negative drivers when attached talent has no qualifying history (NaN reach),
    assert country and episodes are never listed as drivers, and assert baseline sentence and
    talent track record note are present. Tested on fixtures 01, 05, and 14.
    """
    p = next(x for x in eval_pitches if x["id"] == pitch_id)
    orch = AgentOrchestrator(MockLLM(), explain_artifacts, retriever_artifacts)
    res = orch.run({"pitch": p})
    memo = res["memo"]

    # Extract drivers section
    drivers_sec = memo.split("### 3. Main Predictive Drivers")[1].split("### 4.")[0]

    # None of talent groups should appear in positive or negative driver lists
    assert "`talent_cast`" not in drivers_sec
    assert "`talent_director`" not in drivers_sec
    assert "`talent_screenwriter`" not in drivers_sec

    # Country and episodes should NEVER appear as drivers
    assert "`country`" not in drivers_sec
    assert "`episodes`" not in drivers_sec

    # Mandatory baseline & episode influence sentence
    assert "Country is a cohort baseline and planned episode count had little influence on this assessment." in memo

    # Explicit talent track record note must be present
    expected_note = "Talent track record: no qualifying history found for the attached talent, so talent could not contribute to this assessment"
    assert expected_note in memo


def test_low_information_pitch_warning_and_phrasing(eval_pitches, explain_artifacts, retriever_artifacts):
    """
    Assert when >= 3 of network, content_rating, genres, weekdays are blank, memo adds:
    'Low-information pitch: this band mostly reflects missing fields, not the concept; fill them in and re-run.'
    Also assert phrasing uses 'planned premiere year: 2024', 'broadcast days not specified', 'network not specified',
    and plain English alerts with no technical jargon ('encoded as missing', 'NaN').
    Tested on fixture 08.
    """
    p8 = next(x for x in eval_pitches if x["id"] == "pitch_08_blank_everything_except_country_and_start")
    orch = AgentOrchestrator(MockLLM(), explain_artifacts, retriever_artifacts)
    res = orch.run({"pitch": p8})
    memo = res["memo"]

    # Section 2 low-info notice
    assert "Low-information pitch: this band mostly reflects missing fields, not the concept; fill them in and re-run." in memo

    # Section 1 phrasing & year-only start format
    assert "planned premiere year: 2024" in memo
    assert "broadcast days not specified" in memo
    assert "network not specified" in memo.lower()
    assert "content rating not specified" in memo
    assert "genres not specified" in memo

    # Executive alert phrasing: no technical jargon
    assert "encoded as missing" not in memo
    assert "NaN" not in memo


def test_no_comparables_reasons(eval_pitches, explain_artifacts, retriever_artifacts):
    """
    Assert memo explicitly distinguishes between missing synopsis text vs no candidates found within date cutoff.
    """
    orch = AgentOrchestrator(MockLLM(), explain_artifacts, retriever_artifacts)

    # Case A: Pitch with no synopsis text
    p_no_syn = copy.deepcopy(eval_pitches[0])
    p_no_syn["synopsis_text"] = ""
    res_a = orch.run({"pitch": p_no_syn})
    assert "No historical comparables identified: no synopsis text was provided in the pitch." in res_a["memo"]

    # Case B: Pitch with synopsis text but start date in 2000 (no eligible candidates in maturity window)
    p_early = copy.deepcopy(eval_pitches[0])
    p_early["planned_start"] = "2000-01-01"
    res_b = orch.run({"pitch": p_early})
    assert "No historical comparables identified: no eligible candidate titles found within the maturity date cutoff." in res_b["memo"]


def test_blank_field_warning_location_in_what_if(eval_pitches, explain_artifacts, retriever_artifacts):
    """
    Assert the blank-field warning ('part of this change reflects...') appears ONLY in the what-if section,
    NOT in the general assessment alerts (Section 4).
    """
    p5 = next(x for x in eval_pitches if x["id"] == "pitch_05_blank_network_and_rating")
    orch = AgentOrchestrator(MockLLM(), explain_artifacts, retriever_artifacts)
    res = orch.run({"pitch": p5, "changes": {"network": "JTBC", "content_rating": "15+"}})
    memo = res["memo"]

    # Split into sections
    sec4 = memo.split("### 4. Advisory Warnings & Data Hygiene Alerts")[1].split("### 5.")[0]
    sec6 = memo.split("### 6. Sensitivity Simulation (What-If Analysis)")[1].split("### 7.")[0]

    # Warning must NOT be in Section 4
    assert "part of this change reflects" not in sec4.lower()

    # Warning MUST be in Section 6
    assert "part of this change reflects" in sec6.lower()


@pytest.mark.parametrize("bad_expr", [
    "ninety percent",
    "91 per cent",
    "nine in ten",
    "9 out of 10",
    "3 in 4",
    "one in five",
])
def test_grounding_word_percentages_and_ratios_blocked(eval_pitches, explain_artifacts, retriever_artifacts, bad_expr):
    """Assert word percentages and ratio expressions are rejected by grounding verifier."""
    p1 = eval_pitches[0]
    orch = AgentOrchestrator(MockLLM(), explain_artifacts, retriever_artifacts)
    res = orch.run({"pitch": p1})
    base_memo = res["memo"]

    corrupted = base_memo + f"\nProjected benchmark likelihood is {bad_expr}."
    is_grounded, failures = verify_memo_grounding(corrupted, res["tool_outputs"], p1)
    assert not is_grounded
    assert any("forbidden word or mixed percentage" in f.lower() or "forbidden ratio expression" in f.lower() for f in failures)


@pytest.mark.parametrize("phrase", [
    "near-certain",
    "almost certain",
    "bound to",
    "can't miss",
    "sure thing",
    "no doubt",
    "will almost certainly",
    "certain hit",
    "money-maker",
])
def test_grounding_new_certainty_phrases_blocked(eval_pitches, explain_artifacts, retriever_artifacts, phrase):
    """Assert all 9 newly extended certainty/commercial phrases are blocked by grounding verifier."""
    p1 = eval_pitches[0]
    orch = AgentOrchestrator(MockLLM(), explain_artifacts, retriever_artifacts)
    res = orch.run({"pitch": p1})
    base_memo = res["memo"]

    corrupted = base_memo + f"\nThis production is considered a {phrase} outcome."
    is_grounded, failures = verify_memo_grounding(corrupted, res["tool_outputs"], p1)
    assert not is_grounded
    assert any("forbidden commercial/certainty phrase" in f.lower() for f in failures)


def test_grounding_general_numeric_check_blocked(eval_pitches, explain_artifacts, retriever_artifacts):
    """Assert ungrounded numbers and years outside pitch and outputs are rejected."""
    p1 = eval_pitches[0]
    orch = AgentOrchestrator(MockLLM(), explain_artifacts, retriever_artifacts)
    res = orch.run({"pitch": p1})
    base_memo = res["memo"]

    # Inject fabricated numbers
    corrupted_num = base_memo + "\nEstimated viewer impressions exceed 85000 individuals."
    is_g1, f1 = verify_memo_grounding(corrupted_num, res["tool_outputs"], p1)
    assert not is_g1
    assert any("ungrounded number '85000'" in f.lower() for f in f1)

    # Inject fabricated year
    corrupted_yr = base_memo + "\nReferencing historical performance patterns from 2011."
    is_g2, f2 = verify_memo_grounding(corrupted_yr, res["tool_outputs"], p1)
    assert not is_g2
    assert any("ungrounded year 2011" in f.lower() or "ungrounded number '2011'" in f.lower() for f in f2)
