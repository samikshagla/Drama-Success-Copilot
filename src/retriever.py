"""
Similar-Drama Retriever Module (Milestone 6)
=============================================
Retrieves structurally and thematically similar historical dramas based on
synopsis/pitch embeddings using sentence-transformers (all-MiniLM-L6-v2).

Key Features:
- Fixed embedding index in models/ (float32 .npy matrix, gitignored).
- Accompanying metadata table (Parquet, gitignored) containing:
  name, release_year, country, network, target_tier, target_pct, url, start_date.
  NEVER includes synopsis text.
- Candidate pool maturity rule: effective candidate start_date <= query_start - 365 days.
  - Candidate missing start_date: imputed conservatively to Dec 31 of release_year.
  - Query planned_start as year only (e.g. '2024'): imputed conservatively to Jan 1 of release_year.
- Unlisted broadcaster network returns None (null in JSON), never 'Unknown'.
- Optional country filtering.
- Self-exclusion during evaluation (via exclude_url or matching).
- Deterministic tie-breaking: ties in cosine similarity are broken by ascending corpus index.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_LICENSE = "Apache-2.0"
EMBEDDING_DIM = 384
MAX_SEQ_LENGTH = 256


def parse_query_start_date(planned_start: str | int | pd.Timestamp) -> pd.Timestamp:
    """
    Parses planned premiere start date.
    If only a 4-digit year is provided, uses Jan 1 of that year (conservative query assumption).
    """
    s = str(planned_start).strip()
    if s.isdigit() and len(s) == 4:
        return pd.to_datetime(f"{s}-01-01")
    return pd.to_datetime(s)


def get_effective_candidate_dates(
    start_dates: pd.Series, release_years: pd.Series
) -> pd.Series:
    """
    Computes effective start date for candidate pool dramas.
    If start_date is missing or NaN, imputes Dec 31 of release_year (conservative candidate assumption).
    """
    parsed = pd.to_datetime(start_dates, errors="coerce")
    missing = parsed.isna()
    if missing.any():
        imputed_years = release_years[missing].astype(int).astype(str) + "-12-31"
        parsed.loc[missing] = pd.to_datetime(imputed_years)
    return parsed


def load_retriever_artifacts(models_dir: str | Path = "models") -> dict[str, Any]:
    """
    Loads prebuilt retriever embeddings and metadata from models_dir.
    """
    models_path = Path(models_dir)
    emb_path = models_path / "retriever_embeddings.npy"
    meta_path = models_path / "retriever_metadata.parquet"

    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Retriever artifacts missing in {models_dir}. "
            "Run src.build_retriever.build_retriever_artifacts() to build them."
        )

    embeddings = np.load(emb_path)
    metadata = pd.read_parquet(meta_path)
    return {"embeddings": embeddings, "metadata": metadata}


def search(
    query_vec: np.ndarray,
    k: int,
    mask: np.ndarray | None = None,
    corpus_embeddings: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Backend search function computing cosine similarity on unit-normalized vectors.

    Parameters:
        query_vec: 1D array of shape (D,) or 2D (1, D), unit-normalized.
        k: Number of nearest neighbors to retrieve.
        mask: Optional boolean array of shape (N,) indicating allowed candidates (True=include).
        corpus_embeddings: Normalized embedding matrix of shape (N, D).

    Returns:
        indices: 1D array of integer indices in corpus, ordered by similarity descending.
        scores: 1D array of cosine similarities.
    """
    if corpus_embeddings is None:
        raise ValueError("corpus_embeddings must be provided to search().")

    q = np.asarray(query_vec, dtype=np.float32).ravel()
    sims = corpus_embeddings @ q

    if mask is not None:
        sims = np.where(mask, sims, -np.inf)
        n_available = int(np.sum(mask))
    else:
        n_available = len(sims)

    actual_k = min(k, n_available)
    if actual_k <= 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32)

    # Deterministic tie-breaking: primary key = -sims (descending), secondary key = corpus index (ascending)
    indices = np.arange(len(sims))
    order = np.lexsort((indices, -sims))
    top_indices = order[:actual_k]

    return top_indices, sims[top_indices].astype(np.float32)


def find_similar(
    query_text: str,
    planned_start: str | int | pd.Timestamp,
    country: str | None = None,
    k: int = 5,
    artifacts: dict[str, Any] | None = None,
    model: SentenceTransformer | None = None,
    exclude_url: str | None = None,
) -> dict[str, Any]:
    """
    Retrieves top-k similar dramas for a pitch under the maturity constraint.

    Parameters:
        query_text: Drama pitch or synopsis text.
        planned_start: Planned premiere start date (e.g. '2024-12-26' or '2024').
        country: Optional country string to restrict candidate pool (e.g. 'South Korea').
        k: Number of neighbors to return (default 5).
        artifacts: Pre-loaded retriever artifacts dict (optional).
        model: Pre-loaded SentenceTransformer model (optional).
        exclude_url: Optional URL string of query drama to exclude from candidates.

    Returns:
        JSON-serializable dictionary with metadata and neighbors.
        NEVER contains synopsis text.
    """
    if artifacts is None:
        artifacts = load_retriever_artifacts()

    corpus_embeddings = artifacts["embeddings"]
    metadata = artifacts["metadata"]

    planned_dt = parse_query_start_date(planned_start)
    # 12-month maturity cutoff rule
    maturity_cutoff = planned_dt - pd.Timedelta(days=365)

    # Effective candidate dates with conservative Dec 31 imputation for missing dates
    candidate_eff_dates = get_effective_candidate_dates(
        metadata["start_date"], metadata["release_year"]
    )

    # Boolean candidate mask
    mask = candidate_eff_dates <= maturity_cutoff

    if country is not None:
        mask = mask & (metadata["country"] == country)

    if exclude_url is not None:
        mask = mask & (metadata["url"] != exclude_url)

    # Encode query pitch
    if model is None:
        model = SentenceTransformer(DEFAULT_MODEL_NAME, device="cpu")

    query_vec = model.encode(query_text, normalize_embeddings=True).astype(np.float32)

    # Run search
    top_indices, top_scores = search(
        query_vec=query_vec,
        k=k,
        mask=mask.values,
        corpus_embeddings=corpus_embeddings,
    )

    neighbors = []
    for idx, score in zip(top_indices, top_scores):
        row = metadata.iloc[idx]
        raw_net = row["network"]
        # Return null (None in Python) for unlisted networks, not "Unknown"
        net_val = None if (pd.isna(raw_net) or str(raw_net).strip() in ["", "Unknown"]) else str(raw_net)

        start_val = None if pd.isna(row["start_date"]) else str(row["start_date"])

        neighbors.append(
            {
                "name": str(row["name"]),
                "release_year": int(row["release_year"]),
                "country": str(row["country"]),
                "network": net_val,
                "target_tier": str(row["target_tier"]),
                "target_pct": float(round(row["target_pct"], 4)),
                "url": str(row["url"]),
                "start_date": start_val,
                "similarity_score": float(round(float(score), 4)),
            }
        )

    q_preview = (query_text[:77] + "...") if len(query_text) > 80 else query_text

    result = {
        "query_preview": q_preview,
        "planned_start": planned_dt.strftime("%Y-%m-%d"),
        "maturity_cutoff": maturity_cutoff.strftime("%Y-%m-%d"),
        "country_filter": country,
        "candidate_pool_size": int(mask.sum()),
        "k_requested": int(k),
        "k_returned": len(neighbors),
        "neighbors": neighbors,
    }

    # Ensure JSON serializability
    json.dumps(result)
    return result
