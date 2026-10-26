"""
Shared constants, governance caveats, and allowlists for agent tools, orchestrator, and grounding.
"""

# Required Caveats
COMPARABLES_CAVEAT = (
    "comparables show similar premises; their tiers are not a probability for this pitch"
)

WHAT_IF_CAVEAT = (
    "Results are associational, not causal; changing attachments does not guarantee audience reach shifts."
)

BAND_BOUNDARY_WARNING = (
    "band edges are approximate: the calibrator is a step function"
)

LIMITATIONS_NOTICE = (
    "Forecasts and attributions reflect observational correlations from historical training data, "
    "not causal treatment effects, commercial earnings predictions, or performance guarantees. "
    "Actual audience reach is subject to script execution, marketing investments, competition, and platform scheduling."
)

# Mandatory observed rate qualification sentence (Item 1)
OBSERVED_RATE_QUALIFIER = (
    "among the 2,525 evaluated test dramas (released 2022-2025, 6+ episodes, all four countries) "
    "that the model placed in this band. High = the top quarter of MyDramaList watchers among dramas of the same country and release year."
)

# Orchestrator Allowlist
ALLOWED_TOOLS = {
    "explain_pitch",
    "find_comparables",
    "what_if",
}

# Permitted levers for what_if
ALLOWED_LEVERS = {
    "network",
    "content_rating",
    "directors",
    "writers",
    "lead_cast",
    "genres",
}

# Disallowed commercial / certainty terms
FORBIDDEN_PHRASES = [
    "guaranteed",
    "guarantees",
    "guarantee",
    "profit",
    "profits",
    "profitable",
    "earnings",
    "sure to",
    "certain to",
    "surefire",
    "roi",
    "return on investment",
    "will succeed",
    "revenue",
    "box office",
    "commercial hit",
    "financial return",
    "near-certain",
    "almost certain",
    "bound to",
    "can't miss",
    "sure thing",
    "no doubt",
    "will almost certainly",
    "certain hit",
    "money-maker",
]
