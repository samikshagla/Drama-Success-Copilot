"""
src/agent.py
Agent Orchestrator and LLM Client Protocol for Drama Success Copilot.

Enforces:
1. LLMClient protocol decoupling agent orchestration from underlying model.
2. Deterministic MockLLM with template memo generation and quotable-only citation.
3. Stub LiveLLM adapter requiring explicit environment configuration (no keys in code).
4. Strict Orchestrator governance: <= 4 calls, allowlisted tools, anti-injection argument equality,
   mandatory explain_pitch and find_comparables, conditional what_if, and grounding verification.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from src.agent_tools import (
    ALLOWED_WHAT_IF_LEVERS,
    COUNTRY_TRAIN_DISTRIBUTIONS,
    explain_pitch,
    find_comparables,
    load_explain_artifacts,
    load_retriever_artifacts,
    validate_pitch,
    validate_what_if_changes,
    what_if,
)
from src.grounding import verify_memo_grounding
from src.shared_constants import (
    ALLOWED_TOOLS,
    BAND_BOUNDARY_WARNING,
    COMPARABLES_CAVEAT,
    LIMITATIONS_NOTICE,
    OBSERVED_RATE_QUALIFIER,
    WHAT_IF_CAVEAT,
)

logger = logging.getLogger(__name__)

# Canonical network capitalization mapping across 57 recognized training networks
CANONICAL_NETWORKS_MAP: Dict[str, str] = {}
for _c, _dist in COUNTRY_TRAIN_DISTRIBUTIONS.items():
    for _n in _dist.get("networks", {}).keys():
        CANONICAL_NETWORKS_MAP[_n.lower()] = _n


# -----------------------------------------------------------------------------
# Tool Specifications
# -----------------------------------------------------------------------------

TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": "explain_pitch",
        "description": "Evaluate pre-release pitch, compute calibrated success tier likelihood band, and extract SHAP drivers.",
        "parameters": {
            "type": "object",
            "properties": {"pitch": {"type": "object", "description": "Structured pitch metadata"}},
            "required": ["pitch"],
        },
    },
    {
        "name": "find_comparables",
        "description": "Retrieve semantic premise comparables under strict 12-month maturity rule without returning synopsis text.",
        "parameters": {
            "type": "object",
            "properties": {
                "pitch": {"type": "object", "description": "Structured pitch metadata"},
                "k": {"type": "integer", "description": "Number of comparables to retrieve (default 3)", "default": 3},
            },
            "required": ["pitch"],
        },
    },
    {
        "name": "what_if",
        "description": "Simulate attachment modifications across allowed levers and decompose baseline lift from filling blank fields.",
        "parameters": {
            "type": "object",
            "properties": {
                "pitch": {"type": "object", "description": "Structured pitch metadata"},
                "changes": {"type": "object", "description": "Modifications dictionary across allowed levers"},
            },
            "required": ["pitch", "changes"],
        },
    },
]


# -----------------------------------------------------------------------------
# LLMClient Protocol
# -----------------------------------------------------------------------------

@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM agents interacting with the drama success copilot."""

    def generate_plan(
        self,
        pitch: Dict[str, Any],
        tool_specs: List[Dict[str, Any]],
        what_if_changes: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return an ordered list of tool call dictionaries."""
        ...

    def write_memo(
        self,
        tool_outputs: Dict[str, Any],
        pitch_context: Dict[str, Any],
        what_if_changes: Optional[Dict[str, Any]] = None,
        disallowed_levers: Optional[List[str]] = None,
    ) -> str:
        """Draft executive decision memorandum from tool outputs."""
        ...


# -----------------------------------------------------------------------------
# Deterministic Template Generator
# -----------------------------------------------------------------------------

def format_blank_fill_share_statement(decomp: Optional[Dict[str, Any]]) -> Optional[str]:
    """Generate dynamic explanation of blank-field filling raw decomposition shares."""
    if not decomp or not decomp.get("filled_fields"):
        return None
    raw_shares = decomp.get("raw_shares", {})
    share_pct = raw_shares.get("marginal_control_share_pct", 0.0)
    share = share_pct / 100.0 if share_pct > 1.0 else share_pct

    if 0.60 <= share <= 0.72:
        qualifier = "about two thirds"
    elif share > 0.72:
        qualifier = "most"
    elif 0.40 <= share < 0.60:
        qualifier = "about half"
    else:
        qualifier = "a minority"

    filled = ", ".join(decomp["filled_fields"])
    return (
        f"Modifications filled unlisted fields ({filled}). "
        f"{qualifier.capitalize()} of the raw gain comes from the fields being filled in rather than specific choices, "
        f"reflecting the baseline statistical penalty for unlisted metadata in the training corpus."
    )


def format_driver_entry(d: Dict[str, Any], direction: str) -> str:
    """Format driver group name and direction/context according to governance guidelines."""
    group = d.get("group", "")
    reason = d.get("reason", "")
    if group == "country":
        return "`country` (country cohort baseline)"
    return f"`{group}` ({direction})"


def format_executive_alert(w: str) -> str:
    """
    Rewrite technical tool warnings into clear, executive-facing alerts:
    - Removes technical jargon ('encoded as missing', 'NaN', 'feature set to missing').
    - Preserves all underlying facts (12-month rule, debut vs unlisted, missing field penalty,
      57 known networks, 6-episode scope cutoff, 150-episode horizon, 2025 horizon).
    """
    w_clean = w.strip()

    # 1. Talent prior credits / 12-month rule
    if "no qualifying prior credits found" in w_clean.lower():
        return (
            "No qualifying prior credits found within the 12-month rule. "
            "The model cannot distinguish a debut creator from an unlisted track record; "
            "prior reach is treated as unknown."
        )

    # 2. Blank network
    if "network is blank" in w_clean.lower():
        return (
            "Broadcaster network is not specified. "
            "The model predicts lower likelihood of the High tier when this field is left blank than when a typical broadcaster is attached; "
            "specify a network for an accurate assessment."
        )

    # 3. Blank content rating
    if "content rating is blank" in w_clean.lower():
        return (
            "Content rating is not specified. "
            "The model predicts lower likelihood of the High tier when this field is left blank than when a typical rating is attached; "
            "specify a content rating if known."
        )

    # 4. Blank weekdays
    if "broadcast weekday schedule is blank" in w_clean.lower():
        return (
            "Broadcast weekday schedule is not specified. "
            "The model predicts lower likelihood of the High tier when this schedule is left blank than when typical broadcast days are attached; "
            "specify broadcast days if known."
        )

    # 5. Blank genres
    if "genres list is blank" in w_clean.lower():
        return (
            "Genre combination is not specified. "
            "The model predicts lower likelihood of the High tier when genres are left blank than when typical genres are attached; "
            "specify planned genres if known."
        )

    # 6. Unrecognized network
    if "is not among the 57 known networks" in w_clean.lower():
        m = re.search(
            r"broadcaster\s+'([^']+)'\s+is not among the 57 known networks.*?(?:close matches:\s*([^.]+))?",
            w_clean,
            re.IGNORECASE,
        )
        if m:
            net = m.group(1)
            matches = m.group(2)
            if matches:
                return f"Broadcaster '{net}' is not among the 57 known networks in historical data; treated as 'Other'. Closest recognized names: {matches.strip()}."
            return f"Broadcaster '{net}' is not among the 57 known networks in historical data; treated as 'Other'."

    # 7. Planned episode count omitted / defaulted
    if "episode count is not specified" in w_clean.lower() or "planned episode count is not specified" in w_clean.lower():
        return "Planned episode count is not specified; defaulted to typical length of 16 episodes for assessment."

    # 8. Disallowed lever notice (if already formatted)
    if "disallowed lever" in w_clean.lower():
        return w_clean

    # Fallback cleaning: strip "(NaN)" and "encoded as missing" if any other warning appears
    w_clean = re.sub(r"\(NaN\)", "(unknown)", w_clean)
    w_clean = re.sub(r";\s*encoded as missing", "", w_clean)
    w_clean = re.sub(r";\s*feature set to missing", "", w_clean)
    return w_clean


def generate_template_memo(
    tool_outputs: Dict[str, Any],
    pitch: Dict[str, Any],
    what_if_changes: Optional[Dict[str, Any]] = None,
    disallowed_levers: Optional[List[str]] = None,
) -> str:
    """
    Generate a deterministic, 100% grounded executive decision memorandum.
    Adheres strictly to all governance guidelines:
    - Numbered dynamically to prevent missing section numbers.
    - Low-information warning when >= 3 metadata fields are missing.
    - Filter talent groups with NaN reach from drivers and surface track record note.
    - Verbatim observed_band_rate with unified qualification gloss.
    - Small comparison table for Before / After what-if sensitivity.
    - Table of comparables with explicit absence explanation and retrieval pool caveat.
    - Zero continuous decimals, zero commercial certainty language, zero talent names.
    """
    exp_out = tool_outputs.get("explain_pitch", {})
    comp_out = tool_outputs.get("find_comparables", {})
    wi_out = tool_outputs.get("what_if")

    exp_quotable = exp_out.get("quotable", {})
    comp_quotable = comp_out.get("quotable", {})
    wi_quotable = wi_out.get("quotable", {}) if wi_out else None

    lines: List[str] = []
    sec_num = 1

    def is_blank(val: Any) -> bool:
        if val is None:
            return True
        if isinstance(val, (list, tuple)):
            return len(val) == 0 or all(is_blank(x) for x in val)
        s = str(val).strip()
        return s in ["", "Unknown", "Not Yet Rated", "nan", "None", "Unspecified / blank", "None specified"]

    # Section 1: What Was Assessed
    country = pitch.get("country", "Unknown")
    planned_start_raw = str(pitch.get("planned_start") or "").strip()
    is_year_only = bool(re.fullmatch(r"\d{4}", planned_start_raw))
    is_ep_blank = pitch.get("episodes") is None or is_blank(pitch.get("episodes"))

    if is_ep_blank:
        ep_phrase = "planned episode count not specified (the assessment assumed the typical length of 16 episodes)"
    else:
        try:
            episodes_int = int(round(float(pitch.get("episodes"))))
            ep_phrase = f"{episodes_int} planned episodes"
        except (ValueError, TypeError):
            ep_phrase = "planned episode count not specified (the assessment assumed the typical length of 16 episodes)"

    if is_year_only:
        schedule_part = f"Planned premiere year: {planned_start_raw}"
        if is_ep_blank:
            sched_ep_part = f"{schedule_part}; {ep_phrase}"
        else:
            sched_ep_part = f"{schedule_part} with {ep_phrase}"
    elif planned_start_raw and planned_start_raw.lower() not in ["unspecified", "none", "unknown", "nan"]:
        schedule_part = f"Planned release schedule targets premiere on {planned_start_raw}"
        if is_ep_blank:
            sched_ep_part = f"{schedule_part}; {ep_phrase}"
        else:
            sched_ep_part = f"{schedule_part} with {ep_phrase}"
    else:
        schedule_part = "Planned premiere date not specified"
        sched_ep_part = f"{schedule_part}; {ep_phrase}"

    canonical_network = exp_out.get("pitch_summary", {}).get("network")
    if is_blank(canonical_network):
        canonical_network = pitch.get("network")
    if not is_blank(canonical_network):
        c_str = str(canonical_network).strip()
        canonical_name = CANONICAL_NETWORKS_MAP.get(c_str.lower(), c_str)
        net_phrase = f"Broadcaster listed as '{canonical_name}'"
    else:
        net_phrase = "Network not specified"

    cr_val = pitch.get("content_rating")
    if not is_blank(cr_val):
        cr_phrase = f"content rating '{cr_val}'"
    else:
        cr_phrase = "content rating not specified"

    wk_val = pitch.get("weekdays")
    if not is_blank(wk_val):
        wk_list = [str(x).strip() for x in wk_val if not is_blank(x)] if isinstance(wk_val, (list, tuple)) else [str(wk_val).strip()]
        wk_phrase = f"scheduled for broadcast on {', '.join(wk_list)}"
    else:
        wk_phrase = "broadcast days not specified"

    g_val = pitch.get("genres")
    if not is_blank(g_val):
        g_list = [str(x).strip() for x in g_val if not is_blank(x)] if isinstance(g_val, (list, tuple)) else [str(g_val).strip()]
        g_phrase = f"planned genres: {', '.join(g_list)}"
    else:
        g_phrase = "genres not specified"

    lines.append(f"### {sec_num}. What Was Assessed")
    sec_num += 1
    lines.append(
        f"Production assessment for proposed drama originating in {country}. "
        f"{sched_ep_part}; {wk_phrase}. {net_phrase}, {cr_phrase}, and {g_phrase}. "
        f"Individual creator names and narrative synopsis text are excluded from assessment."
    )
    lines.append("")

    # Section 2: Likelihood Band & Observed Test Rate
    band = exp_quotable.get("likelihood_band", "Unknown")
    obs_rate = exp_quotable.get("observed_band_rate", "N/A")

    # Check for low-information pitch (>= 3 of network, content_rating, genres, weekdays blank)
    blank_metadata_count = sum([
        is_blank(pitch.get("network")),
        is_blank(pitch.get("content_rating")),
        is_blank(pitch.get("genres")),
        is_blank(pitch.get("weekdays")),
    ])
    is_low_info = blank_metadata_count >= 3

    lines.append(f"### {sec_num}. Likelihood Band & Historical Track Record")
    sec_num += 1
    if is_low_info:
        lines.append("Low-information pitch: this band mostly reflects missing fields, not the concept; fill them in and re-run.")
    lines.append(f"Model-Assigned Likelihood Band: **{band}**")
    lines.append(f"Empirical benchmark: **{obs_rate}** {OBSERVED_RATE_QUALIFIER}")
    lines.append("")

    # Section 3: Main Predictive Drivers (Group names and direction/context only)
    lines.append(f"### {sec_num}. Main Predictive Drivers")
    sec_num += 1
    pos_drivers = exp_out.get("top_positive_drivers", [])
    neg_drivers = exp_out.get("top_negative_drivers", [])

    # Threshold for material executive drivers (2.0 percentage points of High-tier probability)
    DRIVER_ATTRIBUTION_THRESHOLD = 0.020
    EXCLUDED_DRIVER_GROUPS = {"country", "episodes"}

    def is_talent_no_history(d: Dict[str, Any]) -> bool:
        group = d.get("group", "")
        reason = d.get("reason", "").lower()
        return group.startswith("talent_") and (
            "missing" in reason or "unlisted" in reason or "no qualifying" in reason or "debut" in reason or "unknown" in reason
        )

    # Filter out excluded baseline groups, sub-threshold drivers, and unlisted talent
    filtered_pos = [
        d for d in pos_drivers
        if d.get("group") not in EXCLUDED_DRIVER_GROUPS
        and abs(float(d.get("attribution", 0.0))) >= DRIVER_ATTRIBUTION_THRESHOLD
        and not is_talent_no_history(d)
    ]
    filtered_neg = [
        d for d in neg_drivers
        if d.get("group") not in EXCLUDED_DRIVER_GROUPS
        and abs(float(d.get("attribution", 0.0))) >= DRIVER_ATTRIBUTION_THRESHOLD
        and not is_talent_no_history(d)
    ]

    # Check if any attached talent had no qualifying history
    has_attached_talent = bool(pitch.get("directors") or pitch.get("writers") or pitch.get("lead_cast"))
    has_talent_no_hist = any(is_talent_no_history(d) for d in (pos_drivers + neg_drivers)) or any(
        "No qualifying prior credits found" in w for w in exp_out.get("warnings", [])
    )

    if filtered_pos:
        pos_names = [format_driver_entry(d, "positive") for d in filtered_pos[:3]]
        lines.append(f"- **Top Positive Feature Groups:** {', '.join(pos_names)}.")
    else:
        lines.append("- **Top Positive Feature Groups:** None identified.")

    if filtered_neg:
        neg_names = [format_driver_entry(d, "negative") for d in filtered_neg[:3]]
        lines.append(f"- **Top Negative Feature Groups:** {', '.join(neg_names)}.")
    else:
        lines.append("- **Top Negative Feature Groups:** None identified.")

    lines.append("- Country is a cohort baseline and planned episode count had little influence on this assessment.")

    if has_attached_talent and has_talent_no_hist:
        lines.append("- Talent track record: no qualifying history found for the attached talent, so talent could not contribute to this assessment")
    lines.append("")

    # Section 4: Advisory Warnings & Governance Alerts
    lines.append(f"### {sec_num}. Advisory Warnings & Data Hygiene Alerts")
    sec_num += 1
    raw_warnings = exp_out.get("warnings", []) + (wi_out.get("warnings", []) if wi_out else [])
    has_what_if_section = (wi_out is not None) or (disallowed_levers and len(disallowed_levers) > 0)

    clean_warnings: List[str] = []
    for w in raw_warnings:
        # Normalize P(High) -> the model's likelihood of the High tier
        w_norm = re.sub(r"\bP\(High\)\b", "the model's likelihood of the High tier", w, flags=re.IGNORECASE)
        # De-duplicate step-function warning if What-If section will present it as Calibration Step Alert
        if has_what_if_section and BAND_BOUNDARY_WARNING in w_norm.lower():
            continue
        # Show blank-field warning only in What-If section, not in assessment alerts
        if "part of this change reflects" in w_norm.lower():
            continue
        exec_alert = format_executive_alert(w_norm)
        if exec_alert not in clean_warnings:
            clean_warnings.append(exec_alert)

    if disallowed_levers:
        dl_str = ", ".join(sorted(disallowed_levers))
        clean_warnings.append(
            f"The requested modification to disallowed lever(s) '{dl_str}' is not supported (allowed levers: network, genres, content_rating, directors, writers, lead_cast)."
        )

    if clean_warnings:
        for w in clean_warnings:
            lines.append(f"- *Alert:* {w}")
    else:
        lines.append("- No data hygiene or missingness warnings triggered for this pitch.")
    lines.append("")

    # Section 5: Comparable Historical Dramas
    lines.append(f"### {sec_num}. Comparable Historical Dramas")
    sec_num += 1
    comparables = comp_quotable.get("comparables", [])
    if comparables:
        lines.append("| Rank | Title | Release Year | Country | Broadcaster | Reach Tier | URL |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for idx, rec in enumerate(comparables, 1):
            t = rec.get("name") or rec.get("title") or "Unknown"
            y = rec.get("release_year", "N/A")
            c = rec.get("country", "Unknown")
            n = rec.get("network") or "Unlisted"
            tier = rec.get("tier", "Unknown")
            url = rec.get("url", "#")
            lines.append(f"| {idx} | {t} | {y} | {c} | {n} | {tier} | {url} |")
    else:
        syn = pitch.get("synopsis_text")
        if not syn or not str(syn).strip():
            lines.append("No historical comparables identified: no synopsis text was provided in the pitch.")
        else:
            lines.append("No historical comparables identified: no eligible candidate titles found within the maturity date cutoff.")

    lines.append(f"\n> [!NOTE]\n> **Retrieval Scope Caveat:** {COMPARABLES_CAVEAT}.")
    lines.append("")

    # Section 6: Sensitivity Simulation (What-If Analysis) - dynamically included
    if wi_out or disallowed_levers:
        lines.append(f"### {sec_num}. Sensitivity Simulation (What-If Analysis)")
        sec_num += 1

        if disallowed_levers:
            dl_str = ", ".join(sorted(disallowed_levers))
            lines.append(f"- **Unsupported Levers:** The requested modification to '{dl_str}' is not supported (allowed levers: network, genres, content_rating, directors, writers, lead_cast).")

        if wi_out and wi_quotable:
            b_band = wi_quotable["before"]["likelihood_band"]
            b_rate = wi_quotable["before"]["observed_band_rate"]
            a_band = wi_quotable["after"]["likelihood_band"]
            a_rate = wi_quotable["after"]["observed_band_rate"]

            changes_applied = wi_out.get("changes_applied", {})
            if changes_applied:
                items = []
                for k, v in changes_applied.items():
                    label = k.replace("_", " ").capitalize()
                    if k == "network" and isinstance(v, str):
                        val_str = CANONICAL_NETWORKS_MAP.get(v.lower().strip(), v)
                    elif isinstance(v, list):
                        val_str = ", ".join(str(x) for x in v)
                    else:
                        val_str = str(v)
                    items.append(f"{label}: {val_str}")
                changes_str = "; ".join(items)
            else:
                changes_str = "None"

            lines.append(f"- **Attachment Changes Evaluated:** {changes_str}")
            lines.append("")
            lines.append("| State | Likelihood Band | Empirical Benchmark (Observed Test Rate) |")
            lines.append("| :--- | :--- | :--- |")
            lines.append(f"| Before | {b_band} | {b_rate} |")
            lines.append(f"| After | {a_band} | {a_rate} |")
            lines.append("")

            decomp = wi_out.get("decomposition")
            share_stmt = format_blank_fill_share_statement(decomp)
            if share_stmt:
                lines.append(
                    f"- **Blank-Field Effect:** Part of this change reflects the model's lower likelihood of the High tier "
                    f"when a field is blank; most of the lift comes from filling in the blank fields, not from the specific choices. {share_stmt}"
                )

            if wi_out.get("band_boundary_warning"):
                lines.append(f"- **Calibration Step Alert:** {BAND_BOUNDARY_WARNING}.")

            lines.append(f"\n> [!IMPORTANT]\n> **Sensitivity Caveat:** {WHAT_IF_CAVEAT}")
        lines.append("")

    # Final Section: Methodological Limitations & Governance Notice
    lines.append(f"### {sec_num}. Methodological Limitations & Governance Notice")
    lines.append(f"{LIMITATIONS_NOTICE}")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# MockLLM Implementation
# -----------------------------------------------------------------------------

class MockLLM:
    """
    Deterministic Mock LLM client implementing LLMClient protocol.
    Follows fixed policy and generates strictly grounded template memos.
    """

    def generate_plan(
        self,
        pitch: Dict[str, Any],
        tool_specs: List[Dict[str, Any]],
        what_if_changes: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate deterministic execution plan."""
        plan: List[Dict[str, Any]] = [
            {"tool": "explain_pitch", "arguments": {"pitch": copy.deepcopy(pitch)}},
            {"tool": "find_comparables", "arguments": {"pitch": copy.deepcopy(pitch), "k": 3}},
        ]
        if what_if_changes:
            plan.append({
                "tool": "what_if",
                "arguments": {
                    "pitch": copy.deepcopy(pitch),
                    "changes": copy.deepcopy(what_if_changes),
                },
            })
        return plan

    def write_memo(
        self,
        tool_outputs: Dict[str, Any],
        pitch_context: Dict[str, Any],
        what_if_changes: Optional[Dict[str, Any]] = None,
        disallowed_levers: Optional[List[str]] = None,
    ) -> str:
        """Draft memo using deterministic template generator."""
        return generate_template_memo(tool_outputs, pitch_context, what_if_changes, disallowed_levers=disallowed_levers)


# -----------------------------------------------------------------------------
# LiveLLM Stub Implementation
# -----------------------------------------------------------------------------

class LiveLLM:
    """
    Stub adapter for external LLM providers.
    Disabled by default; requires explicit environment variable configuration.
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or os.environ.get("LLM_PROVIDER", "mock")
        if self.provider == "mock":
            self._backend: LLMClient = MockLLM()
        else:
            raise NotImplementedError(
                f"Live LLM provider '{self.provider}' is configured in environment, but external LLM "
                f"SDKs are deliberately not installed in this repository. Set LLM_PROVIDER=mock."
            )

    def generate_plan(
        self,
        pitch: Dict[str, Any],
        tool_specs: List[Dict[str, Any]],
        what_if_changes: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        return self._backend.generate_plan(pitch, tool_specs, what_if_changes)

    def write_memo(
        self,
        tool_outputs: Dict[str, Any],
        pitch_context: Dict[str, Any],
        what_if_changes: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self._backend.write_memo(tool_outputs, pitch_context, what_if_changes)


# -----------------------------------------------------------------------------
# Agent Orchestrator
# -----------------------------------------------------------------------------

class AgentOrchestrator:
    """
    Governance orchestrator coordinating pitch validation, plan execution,
    tool sandboxing, and memo grounding verification.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        explain_artifacts: Optional[Dict[str, Any]] = None,
        retriever_artifacts: Optional[Dict[str, Any]] = None,
    ):
        self.client = llm_client or MockLLM()
        self.explain_artifacts = explain_artifacts
        self.retriever_artifacts = retriever_artifacts

    def _ensure_artifacts(self) -> None:
        if self.explain_artifacts is None:
            self.explain_artifacts = load_explain_artifacts()
        if self.retriever_artifacts is None:
            self.retriever_artifacts = load_retriever_artifacts()

    def run(
        self,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute full agent pipeline on structured request:
        request format: {"pitch": {...}, "changes": Optional[{...}]}
        """
        self._ensure_artifacts()

        raw_pitch = request.get("pitch", {})
        raw_changes = request.get("changes")

        # 1. Validate pitch
        val_pitch = validate_pitch(raw_pitch)

        # 2. Track plan rejections & validate changes if supplied
        plan_rejections: List[str] = []
        val_changes: Optional[Dict[str, Any]] = None
        disallowed_levers_requested: List[str] = []
        if raw_changes:
            if not isinstance(raw_changes, dict):
                raise TypeError(f"Changes must be a dictionary, got {type(raw_changes).__name__}")
            allowed_part = {}
            for k, v in raw_changes.items():
                if k in ALLOWED_WHAT_IF_LEVERS:
                    allowed_part[k] = v
                else:
                    disallowed_levers_requested.append(k)
            if allowed_part:
                val_changes = validate_what_if_changes(allowed_part)
            if disallowed_levers_requested:
                plan_rejections.append(
                    f"Requested change uses disallowed lever(s): {sorted(disallowed_levers_requested)}. "
                    f"Allowed levers: {sorted(ALLOWED_WHAT_IF_LEVERS)}"
                )

        # 3. Request execution plan from LLM layer
        try:
            raw_plan = self.client.generate_plan(val_pitch, TOOL_SPECS, val_changes)
        except Exception as e:
            plan_rejections.append(f"Plan generation failed with error: {str(e)}")
            raw_plan = []

        # 4. Plan verification and filtering
        if not isinstance(raw_plan, list):
            plan_rejections.append(f"LLM plan must be a list of tool calls, got {type(raw_plan).__name__}")
            raw_plan = []

        # Enforce <= 4 calls limit
        if len(raw_plan) > 4:
            plan_rejections.append(f"LLM plan exceeded max call limit (requested {len(raw_plan)} calls > 4 max allowed); truncated.")
            raw_plan = raw_plan[:4]

        # Check each call in plan
        verified_plan: List[Dict[str, Any]] = []
        has_explain = False
        has_comparables = False
        has_what_if = False

        for idx, call in enumerate(raw_plan):
            if not isinstance(call, dict) or "tool" not in call:
                plan_rejections.append(f"Call {idx} invalid: missing 'tool' key")
                continue

            tool_name = call["tool"]
            args = call.get("arguments", {})

            # Check allowlist
            if tool_name not in ALLOWED_TOOLS:
                plan_rejections.append(f"Call {idx} rejected: unknown tool '{tool_name}' not in allowlist {ALLOWED_TOOLS}")
                continue

            # Check argument equality with validated pitch
            if tool_name == "explain_pitch":
                if args.get("pitch") != val_pitch:
                    plan_rejections.append(f"Call {idx} ('explain_pitch') rejected: arguments altered or not equal to validated pitch")
                    continue
                if not has_explain:
                    has_explain = True
                    verified_plan.append({"tool": tool_name, "arguments": {"pitch": val_pitch}})
                else:
                    plan_rejections.append(f"Call {idx} ('explain_pitch') redundant; already scheduled")

            elif tool_name == "find_comparables":
                if args.get("pitch") != val_pitch:
                    plan_rejections.append(f"Call {idx} ('find_comparables') rejected: arguments altered or not equal to validated pitch")
                    continue
                k_val = args.get("k", 3)
                if not isinstance(k_val, int) or k_val <= 0:
                    k_val = 3
                if not has_comparables:
                    has_comparables = True
                    verified_plan.append({"tool": tool_name, "arguments": {"pitch": val_pitch, "k": k_val}})
                else:
                    plan_rejections.append(f"Call {idx} ('find_comparables') redundant; already scheduled")

            elif tool_name == "what_if":
                call_changes = args.get("changes")
                if not val_changes:
                    plan_rejections.append(f"Call {idx} ('what_if') rejected: user request supplied no structured changes")
                    continue
                if not isinstance(call_changes, dict) or not set(call_changes.keys()).issubset(ALLOWED_WHAT_IF_LEVERS):
                    plan_rejections.append(f"Call {idx} ('what_if') rejected: contains disallowed levers")
                    continue
                if args.get("pitch") != val_pitch or call_changes != val_changes:
                    plan_rejections.append(f"Call {idx} ('what_if') rejected: arguments altered or not equal to validated changes")
                    continue
                if not has_what_if:
                    has_what_if = True
                    verified_plan.append({"tool": tool_name, "arguments": {"pitch": val_pitch, "changes": val_changes}})
                else:
                    plan_rejections.append(f"Call {idx} ('what_if') redundant; already scheduled")

        # Enforce mandatory execution of explain_pitch and find_comparables
        if not has_explain:
            plan_rejections.append("Required tool 'explain_pitch' omitted by LLM plan; automatically injected by orchestrator")
            verified_plan.insert(0, {"tool": "explain_pitch", "arguments": {"pitch": val_pitch}})

        if not has_comparables:
            plan_rejections.append("Required tool 'find_comparables' omitted by LLM plan; automatically injected by orchestrator")
            verified_plan.append({"tool": "find_comparables", "arguments": {"pitch": val_pitch, "k": 3}})

        if val_changes and not has_what_if:
            # User provided changes but plan omitted what_if
            verified_plan.append({"tool": "what_if", "arguments": {"pitch": val_pitch, "changes": val_changes}})

        # Strict cap at <= 4 calls under all circumstances
        verified_plan = verified_plan[:4]

        # 5. Execute verified plan
        tool_outputs: Dict[str, Any] = {}
        for call in verified_plan:
            t_name = call["tool"]
            args = call["arguments"]
            if t_name == "explain_pitch":
                tool_outputs["explain_pitch"] = explain_pitch(args["pitch"], artifacts=self.explain_artifacts)
            elif t_name == "find_comparables":
                tool_outputs["find_comparables"] = find_comparables(
                    args["pitch"],
                    k=args.get("k", 3),
                    artifacts=self.retriever_artifacts,
                )
            elif t_name == "what_if":
                tool_outputs["what_if"] = what_if(
                    args["pitch"],
                    args["changes"],
                    artifacts=self.explain_artifacts,
                )

        # 6. Request memo from LLM client
        try:
            import inspect
            sig = inspect.signature(self.client.write_memo)
            if "disallowed_levers" in sig.parameters:
                draft_memo = self.client.write_memo(
                    tool_outputs,
                    val_pitch,
                    val_changes,
                    disallowed_levers=disallowed_levers_requested,
                )
            else:
                draft_memo = self.client.write_memo(tool_outputs, val_pitch, val_changes)
        except Exception as e:
            plan_rejections.append(f"Client memo generation raised exception: {str(e)}; falling back to template")
            draft_memo = ""

        # 7. Grounding and Governance Verification
        talent_lookup = self.explain_artifacts.get("talent_lookup") if self.explain_artifacts else None
        is_grounded, grounding_failures = verify_memo_grounding(
            draft_memo,
            tool_outputs,
            val_pitch,
            talent_lookup=talent_lookup,
        )

        final_memo = draft_memo
        fallback_used = False

        if not is_grounded:
            fallback_used = True
            logger.warning("Draft memo failed grounding verification (%d failures); falling back to template memo", len(grounding_failures))
            final_memo = generate_template_memo(
                tool_outputs,
                val_pitch,
                val_changes,
                disallowed_levers=disallowed_levers_requested,
            )

        return {
            "memo": final_memo,
            "tool_outputs": tool_outputs,
            "plan": verified_plan,
            "plan_rejections": plan_rejections,
            "grounding_passed": is_grounded,
            "grounding_failures": grounding_failures,
            "fallback_to_template": fallback_used,
        }


def run_agent(
    pitch: Dict[str, Any],
    changes: Optional[Dict[str, Any]] = None,
    llm_client: Optional[LLMClient] = None,
    explain_artifacts: Optional[Dict[str, Any]] = None,
    retriever_artifacts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience functional wrapper for AgentOrchestrator."""
    orchestrator = AgentOrchestrator(
        llm_client=llm_client,
        explain_artifacts=explain_artifacts,
        retriever_artifacts=retriever_artifacts,
    )
    request = {"pitch": pitch}
    if changes:
        request["changes"] = changes
    return orchestrator.run(request)
