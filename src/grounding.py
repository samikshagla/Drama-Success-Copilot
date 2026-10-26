"""
src/grounding.py
Comprehensive Grounding and Governance Verifier for Executive Decision Memorandums.

Verifies:
1. Citation and number grounding: all percentages, years, and counts trace directly to tool outputs or pitch inputs.
2. Anti-leakage / Continuous probability check: no raw or calibrated decimal probabilities (e.g. 0.5833, 0.20).
3. Comparable integrity: all referenced titles match returned comparables records.
4. Privacy: no real or debut talent names mentioned (multi-word name scan, excluding comparable title spans).
5. Anti-injection: 6-word synopsis echo rule and rejection of pitch free-text echo.
6. Required caveats: mandatory presence of shared caveat constants.
7. Forbidden phrases: rejection of commercial, financial, or certainty language.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib

from src.shared_constants import (
    BAND_BOUNDARY_WARNING,
    COMPARABLES_CAVEAT,
    FORBIDDEN_PHRASES,
    LIMITATIONS_NOTICE,
    OBSERVED_RATE_QUALIFIER,
    WHAT_IF_CAVEAT,
)


def extract_percentages(text: str) -> List[str]:
    """Extract all percentages from text (e.g., '57.8%', '79.6%')."""
    return re.findall(r"\b(\d+(?:\.\d+)?)\s*%", text)


def extract_years(text: str) -> List[int]:
    """Extract all 4-digit years from text in range 1900-2099."""
    matches = re.findall(r"\b(19\d\d|20\d\d)\b", text)
    return [int(m) for m in matches]


def normalize_words(text: str) -> List[str]:
    """Extract lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def check_synopsis_echo(memo: str, synopsis: Optional[str]) -> List[str]:
    """Flag if 6+ consecutive words from synopsis are echoed in the memo."""
    if not synopsis:
        return []
    syn_words = normalize_words(synopsis)
    if len(syn_words) < 6:
        return []

    memo_words = normalize_words(memo)
    memo_text_joined = " ".join(memo_words)

    failures = []
    # Check 6-word sliding windows
    for i in range(len(syn_words) - 5):
        gram = " ".join(syn_words[i : i + 6])
        if gram in memo_text_joined:
            failures.append(f"Synopsis text echoed (6+ words): '{gram}'")
            break  # Flag first violation
    return failures


def get_name_variants(name: str) -> List[str]:
    """Generate normalized order and reversed order (Western vs Eastern) variants."""
    tokens = re.sub(r"[-_]", " ", name.strip().lower()).split()
    if len(tokens) < 2:
        return []
    variants = [" ".join(tokens)]
    if len(tokens) == 2:
        variants.append(f"{tokens[1]} {tokens[0]}")
    elif len(tokens) == 3:
        # e.g., ["kim", "won", "seok"] -> ["won", "seok", "kim"]
        variants.append(f"{tokens[1]} {tokens[2]} {tokens[0]}")
        # e.g., ["won", "seok", "kim"] -> ["kim", "won", "seok"]
        variants.append(f"{tokens[2]} {tokens[0]} {tokens[1]}")
    return list(dict.fromkeys(variants))


def check_forbidden_phrases(memo: str) -> List[str]:
    """Flag forbidden commercial, financial, or certainty words outside of caveat boilerplate."""
    clean = memo
    for caveat in [WHAT_IF_CAVEAT, LIMITATIONS_NOTICE, COMPARABLES_CAVEAT, BAND_BOUNDARY_WARNING]:
        clean = clean.replace(caveat, " ")
        caveat_pat = re.sub(r"\s+", r"\\s+", re.escape(caveat.strip()))
        clean = re.sub(caveat_pat, " ", clean, flags=re.IGNORECASE)

    failures = []
    memo_lower = clean.lower()
    for phrase in FORBIDDEN_PHRASES:
        pattern = r"\b" + re.escape(phrase.lower()) + r"\b"
        if re.search(pattern, memo_lower):
            failures.append(f"Forbidden commercial/certainty phrase found: '{phrase}'")
    return failures


def check_decimal_probabilities(memo: str, allowed_percentage_decimals: Set[str]) -> List[str]:
    """
    Flag continuous probability decimals (e.g. 0.58, 0.20, 0.02) appearing outside
    of allowed percentage citations.
    """
    failures = []
    # Match standalone decimals like 0.1234 or .58
    decimal_matches = re.findall(r"(?<![A-Za-z0-9_])0\.\d+(?![A-Za-z0-9_])", memo)
    for dec in decimal_matches:
        failures.append(f"Forbidden continuous/calibrated decimal probability found: '{dec}'")
    return failures


def check_talent_names(
    memo: str,
    talent_names: Set[str],
    pitch: Dict[str, Any],
    comparable_titles: List[str],
) -> List[str]:
    """
    Flag individual talent names mentioned in the memo.
    Excludes spans matching returned comparable titles and URLs to avoid false positives.
    Checks multi-word names (>= 2 words) in standard order and reversed variants (Western vs Eastern order).
    """
    failures = []

    # Strip URLs and comparable title spans from memo copy
    clean_memo = re.sub(r"https?://\S+", " ", memo)
    for title in comparable_titles:
        if title and len(title) > 2:
            clean_memo = re.sub(re.escape(title), " " * len(title), clean_memo, flags=re.IGNORECASE)

    # Normalize memo tokens: replace non-alphanumeric with spaces and collect n-grams
    clean_memo_lower = re.sub(r"[^a-z0-9\s]", " ", clean_memo.lower())
    memo_tokens = clean_memo_lower.split()
    n_tokens = len(memo_tokens)
    memo_ngrams: Set[str] = set()
    for n in (2, 3, 4):
        for i in range(n_tokens - n + 1):
            memo_ngrams.add(" ".join(memo_tokens[i : i + n]))

    # 1. Names from pitch input
    pitch_names: Set[str] = set()
    for field in ["directors", "writers", "lead_cast"]:
        for name in pitch.get(field, []):
            if name and isinstance(name, str) and len(name.split()) >= 2:
                pitch_names.add(name.strip())

    for p_name in pitch_names:
        variants = get_name_variants(p_name)
        matched = False
        for var in variants:
            var_norm = " ".join(re.sub(r"[^a-z0-9\s]", " ", var.lower()).split())
            if var_norm in memo_ngrams:
                failures.append(f"Individual talent from pitch named in memo: '{p_name}'")
                matched = True
                break
        if matched:
            continue

    # 2. Multi-word names from historical talent lookup
    for t_name in talent_names:
        if len(t_name.split()) >= 2 and len(t_name) >= 5:
            variants = get_name_variants(t_name)
            hit = False
            for var in variants:
                var_norm = " ".join(re.sub(r"[^a-z0-9\s]", " ", var.lower()).split())
                if var_norm in memo_ngrams:
                    failures.append(f"Historical talent name found in memo: '{t_name}'")
                    hit = True
                    break
            if hit:
                break  # Flag first hit to avoid flooding

    return failures


def verify_memo_grounding(
    memo: str,
    tool_outputs: Dict[str, Any],
    pitch: Dict[str, Any],
    talent_lookup: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    """
    Verifies that an executive decision memo is completely grounded in tool outputs
    and complies with all governance rules.

    Returns:
      (is_grounded, failures)
    """
    failures: List[str] = []

    # 1. Check forbidden promotional/certainty phrases
    failures.extend(check_forbidden_phrases(memo))

    # 2. Check 6-word synopsis echo
    failures.extend(check_synopsis_echo(memo, pitch.get("synopsis_text")))

    # Extract quotable outputs
    exp_out = tool_outputs.get("explain_pitch", {})
    comp_out = tool_outputs.get("find_comparables", {})
    wi_out = tool_outputs.get("what_if")

    exp_quotable = exp_out.get("quotable", {})
    comp_quotable = comp_out.get("quotable", {})
    wi_quotable = wi_out.get("quotable", {}) if wi_out else None

    # Collect allowed percentages
    allowed_percentages: Set[str] = set()
    allowed_decimals: Set[str] = set()

    for q_block in [exp_quotable, wi_quotable.get("before") if wi_quotable else None, wi_quotable.get("after") if wi_quotable else None]:
        if q_block and "observed_band_rate" in q_block:
            for pct in extract_percentages(q_block["observed_band_rate"]):
                allowed_percentages.add(pct)
                allowed_decimals.add(pct)

    # 3. Check percentages in memo
    memo_percentages = extract_percentages(memo)
    for pct in memo_percentages:
        if pct not in allowed_percentages:
            failures.append(f"Ungrounded percentage '{pct}%' found in memo (not in observed_band_rate)")

    # 4. Check standalone decimal probabilities
    failures.extend(check_decimal_probabilities(memo, allowed_decimals))

    # Collect allowed years
    allowed_years: Set[int] = {2022, 2025}  # Cohort range 2022-2025 from qualifier
    start_val = str(pitch.get("planned_start") or "")
    m_year = re.search(r"\b(20\d\d)\b", start_val)
    if m_year:
        allowed_years.add(int(m_year.group(1)))

    comparable_titles: List[str] = []
    comparables_list = comp_quotable.get("comparables", [])
    for rec in comparables_list:
        if "release_year" in rec and rec["release_year"]:
            try:
                allowed_years.add(int(rec["release_year"]))
            except (ValueError, TypeError):
                pass
        name_val = rec.get("name") or rec.get("title")
        if name_val:
            comparable_titles.append(str(name_val))

    # 5. Check years in memo
    memo_years = extract_years(memo)
    for yr in memo_years:
        if yr not in allowed_years:
            failures.append(f"Ungrounded year {yr} found in memo (not in pitch or comparables)")

    # 6. Check comparable title references
    # Look for title markers or check that any quoted / cited title matches returned records
    # If the memo contains a known fake title (e.g. from ChaosLLM), catch it
    if "comparables" in comp_quotable:
        comp_records = comp_quotable["comparables"]
        valid_titles_lower = {(r.get("name") or r.get("title") or "").lower() for r in comp_records}
        # Find lines in Comparable Dramas section
        if "Comparable Historical Dramas" in memo:
            comp_section = memo.split("Comparable Historical Dramas")[1].split("###")[0]
            # Match table rows like | 1 | Title | 2020 |
            table_titles = re.findall(r"\|\s*\d+\s*\|\s*([^|]+?)\s*\|", comp_section)
            for t_raw in table_titles:
                t_clean = t_raw.strip().lower()
                if t_clean and t_clean not in valid_titles_lower:
                    failures.append(f"Ungrounded comparable title in table: '{t_raw.strip()}'")

    # 7. Check talent names
    talent_names: Set[str] = set()
    if talent_lookup:
        for role in ["director", "screenwriter", "cast"]:
            talent_names.update(talent_lookup.get(role, {}).keys())

    failures.extend(check_talent_names(memo, talent_names, pitch, comparable_titles))

    # 8. Check required caveats
    if COMPARABLES_CAVEAT not in memo:
        failures.append("Missing required comparables caveat: COMPARABLES_CAVEAT")

    if LIMITATIONS_NOTICE not in memo:
        failures.append("Missing required governance notice: LIMITATIONS_NOTICE")

    if OBSERVED_RATE_QUALIFIER not in memo:
        failures.append("Missing required observed rate qualifier: OBSERVED_RATE_QUALIFIER")

    if wi_out:
        if WHAT_IF_CAVEAT not in memo:
            failures.append("Missing required what-if caveat: WHAT_IF_CAVEAT")
        if wi_out.get("band_boundary_warning"):
            if BAND_BOUNDARY_WARNING not in memo:
                failures.append("Missing required step-function warning: BAND_BOUNDARY_WARNING")

    is_grounded = len(failures) == 0
    return is_grounded, failures
