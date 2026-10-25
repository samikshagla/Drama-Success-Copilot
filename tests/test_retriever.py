"""
Unit and Integration Tests for src.retriever (Milestone 6)
==========================================================
Tests cover:
a. Date filter: candidate pool contains no drama started after planned_start - 365 days.
b. Missing date convention:
   - Candidate dramas without start_date imputed to Dec 31 of release_year.
   - Query planned_start as 4-digit year imputed to Jan 1 of release_year.
c. Self-exclusion: querying an existing drama excludes it from returned neighbors.
d. Unlisted network: returns None (null in JSON) for unlisted networks, never 'Unknown'.
e. Output size: returns exactly k results (or len(pool) if pool < k).
f. Determinism: identical queries produce identical results across calls.
g. JSON-serializability: result serializes cleanly via json.dumps.
h. Absence of synopsis text: no synopsis content or synopsis field in returned metadata.
i. Clean skip if models/ artifacts or local HuggingFace cache are missing (no network calls).
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import huggingface_hub.constants
import numpy as np
import pandas as pd
import pytest

from src.retriever import (
    DEFAULT_MODEL_NAME,
    find_similar,
    get_effective_candidate_dates,
    load_retriever_artifacts,
    parse_query_start_date,
    search,
)

repo_root = Path(__file__).resolve().parent.parent

REQUIRED_MODEL_FILES = [
    repo_root / "models/retriever_embeddings.npy",
    repo_root / "models/retriever_metadata.parquet",
    repo_root / "models/retriever_info.json",
]

missing_models = [
    p.relative_to(repo_root).as_posix()
    for p in REQUIRED_MODEL_FILES
    if not p.exists()
]

# Check local Hugging Face cache existence
hf_cache = huggingface_hub.constants.HF_HUB_CACHE
hf_model_pattern = os.path.join(
    hf_cache, "models--sentence-transformers--all-MiniLM-L6-v2", "snapshots", "*"
)
cached_model_dirs = glob.glob(hf_model_pattern)
has_local_hf_model = len(cached_model_dirs) > 0
allow_network = bool(os.environ.get("ALLOW_RETRIEVER_NETWORK", False))

skip_reasons = []
if missing_models:
    skip_reasons.append(f"Missing models artifacts: {', '.join(missing_models)}")
if not has_local_hf_model and not allow_network:
    skip_reasons.append(
        "all-MiniLM-L6-v2 model weights not found in local HuggingFace cache and network download disabled"
    )

pytestmark = pytest.mark.skipif(
    bool(skip_reasons),
    reason="; ".join(skip_reasons) + ". Run src/build_retriever.py to generate artifacts.",
)


@pytest.fixture(scope="module")
def artifacts():
    """Load cached retriever embeddings and metadata."""
    return load_retriever_artifacts(repo_root / "models")


@pytest.fixture(scope="module")
def local_model():
    """Load SentenceTransformer strictly from local cache (no network calls)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        DEFAULT_MODEL_NAME,
        device="cpu",
        local_files_only=True,
    )


def test_date_filter_maturity_rule(artifacts, local_model):
    """(a) Date filter: all returned neighbors must have start_date <= planned_start - 365 days."""
    planned_start = "2023-06-01"
    planned_dt = pd.to_datetime(planned_start)
    cutoff = planned_dt - pd.Timedelta(days=365)

    res = find_similar(
        query_text="A suspense thriller about a high-stakes investigation.",
        planned_start=planned_start,
        k=10,
        artifacts=artifacts,
        model=local_model,
    )

    assert len(res["neighbors"]) > 0, "Expected at least 1 neighbor"
    for n in res["neighbors"]:
        if n["start_date"] is not None:
            n_dt = pd.to_datetime(n["start_date"])
            assert n_dt <= cutoff, (
                f"Violation of 365-day maturity rule: drama {n['name']} has start_date {n['start_date']} "
                f"after cutoff {cutoff.strftime('%Y-%m-%d')}"
            )
        else:
            imputed_dt = pd.to_datetime(f"{n['release_year']}-12-31")
            assert imputed_dt <= cutoff


def test_missing_start_date_conventions():
    """
    (b) Missing date convention test:
    - Candidate dramas without start_date imputed to Dec 31 of release_year.
    - Query planned_start as 4-digit year imputed to Jan 1 of release_year.
    """
    start_series = pd.Series(["2020-05-10", None, np.nan, "2019-11-20"])
    year_series = pd.Series([2020, 2018, 2021, 2019])
    eff_dates = get_effective_candidate_dates(start_series, year_series)

    assert eff_dates.iloc[0] == pd.to_datetime("2020-05-10")
    assert eff_dates.iloc[1] == pd.to_datetime("2018-12-31")
    assert eff_dates.iloc[2] == pd.to_datetime("2021-12-31")
    assert eff_dates.iloc[3] == pd.to_datetime("2019-11-20")

    assert parse_query_start_date("2024") == pd.to_datetime("2024-01-01")
    assert parse_query_start_date(2024) == pd.to_datetime("2024-01-01")
    assert parse_query_start_date("2024-08-15") == pd.to_datetime("2024-08-15")

    q_dt = parse_query_start_date("2023")
    cutoff = q_dt - pd.Timedelta(days=365)
    assert eff_dates.iloc[2] <= cutoff
    cand_2022 = get_effective_candidate_dates(pd.Series([None]), pd.Series([2022])).iloc[0]
    assert cand_2022 > cutoff


def test_self_exclusion(artifacts, local_model):
    """(c) Self exclusion: when querying an existing drama by URL, it must not appear in the results."""
    meta = artifacts["metadata"]
    sg_matches = meta[meta["name"].str.contains("Squid Game", case=False, na=False)]
    assert len(sg_matches) > 0, "Could not find Squid Game in metadata"
    target_row = sg_matches.iloc[0]
    target_url = target_row["url"]
    target_name = target_row["name"]

    res = find_similar(
        query_text="Desperate people play children games for cash prizes.",
        planned_start="2025-01-01",
        k=10,
        artifacts=artifacts,
        model=local_model,
        exclude_url=target_url,
    )

    retrieved_urls = [n["url"] for n in res["neighbors"]]
    retrieved_names = [n["name"] for n in res["neighbors"]]
    assert target_url not in retrieved_urls, f"Expected {target_url} to be excluded, but was retrieved"
    assert target_name not in retrieved_names, f"Expected {target_name} to be excluded, but was retrieved"


def test_unlisted_network_returns_null(artifacts, local_model):
    """(d) Unlisted network: returns None (null in JSON) for unlisted networks, never the string 'Unknown'."""
    meta = artifacts["metadata"]
    unlisted_mask = meta["network"].isna()
    assert unlisted_mask.sum() == 840, f"Expected 840 unlisted networks, found {unlisted_mask.sum()}"

    res = find_similar(
        query_text="A romantic comedy between two young professionals.",
        planned_start="2024-01-01",
        k=50,
        artifacts=artifacts,
        model=local_model,
    )

    for n in res["neighbors"]:
        assert n["network"] != "Unknown", "Network returned as string 'Unknown' instead of None/null"
        if n["network"] is not None:
            assert isinstance(n["network"], str)
            assert len(n["network"]) > 0


def test_output_size(artifacts, local_model):
    """(e) Output size: returns exactly k results (or len(pool) if pool < k)."""
    res5 = find_similar(
        query_text="A romantic comedy about two rivals who fall in love.",
        planned_start="2024-01-01",
        k=5,
        artifacts=artifacts,
        model=local_model,
    )
    assert len(res5["neighbors"]) == 5
    assert res5["k_returned"] == 5

    res_early = find_similar(
        query_text="Historical palace drama.",
        planned_start="2015-06-01",
        k=50,
        artifacts=artifacts,
        model=local_model,
    )
    assert len(res_early["neighbors"]) == res_early["candidate_pool_size"]
    assert len(res_early["neighbors"]) <= 50


def test_determinism(artifacts, local_model):
    """(f) Determinism: identical queries produce identical results across calls."""
    query = "Undercover detective infiltrates a dangerous underground criminal ring."
    planned_start = "2024-05-15"

    res1 = find_similar(query, planned_start=planned_start, k=5, artifacts=artifacts, model=local_model)
    res2 = find_similar(query, planned_start=planned_start, k=5, artifacts=artifacts, model=local_model)

    assert res1 == res2, "Non-deterministic retrieval results for identical query"


def test_json_serializable_and_expected_keys(artifacts, local_model):
    """(g) JSON-serializability: verify return structure serializes cleanly."""
    res = find_similar(
        query_text="A high-school sports drama focusing on teamwork and youth.",
        planned_start="2023-09-01",
        country="Japan",
        k=5,
        artifacts=artifacts,
        model=local_model,
    )

    dumped = json.dumps(res)
    loaded = json.loads(dumped)
    assert isinstance(loaded, dict)

    expected_top_keys = [
        "query_preview",
        "planned_start",
        "maturity_cutoff",
        "country_filter",
        "candidate_pool_size",
        "k_requested",
        "k_returned",
        "neighbors",
    ]
    for key in expected_top_keys:
        assert key in loaded, f"Missing top-level key: {key}"

    expected_neighbor_keys = [
        "name",
        "release_year",
        "country",
        "network",
        "target_tier",
        "target_pct",
        "url",
        "start_date",
        "similarity_score",
    ]
    for n in loaded["neighbors"]:
        for key in expected_neighbor_keys:
            assert key in n, f"Missing neighbor key: {key}"


def test_no_synopsis_text_in_returned_data(artifacts, local_model):
    """(h) No synopsis: verify no synopsis text is returned in metadata or neighbor dictionaries."""
    res = find_similar(
        query_text="Medical staff working against the clock in a rural emergency room.",
        planned_start="2023-01-01",
        k=5,
        artifacts=artifacts,
        model=local_model,
    )

    assert "synopsis" not in res
    assert "synopsis_clean" not in res

    for n in res["neighbors"]:
        assert "synopsis" not in n
        assert "synopsis_clean" not in n
        for k, v in n.items():
            if isinstance(v, str):
                assert len(v) <= 100, f"Field '{k}' has unexpectedly long string ({len(v)} chars): {v[:40]}..."
