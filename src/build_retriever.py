"""
Build Retriever Module for Drama Success Copilot (Milestone 6)
==============================================================
Builds and serializes production retriever artifacts into models/:
1. retriever_embeddings.npy: Float32 unit-normalized embeddings matrix (N, 384).
2. retriever_embeddings.urls.txt: Ordered list of corpus drama URLs.
3. retriever_embeddings.urls_sha256: SHA-256 digest of ordered corpus URLs.
4. retriever_metadata.parquet: Drama metadata table without any synopsis text.
5. retriever_info.json: Model metadata, HuggingFace revision hash, library versions, seed.

Usage:
    python src/build_retriever.py [--data-dir data/processed] [--output-dir models] [--device cpu] [--seed 42]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import pyarrow
import sentence_transformers
import torch
import transformers
from sentence_transformers import SentenceTransformer

from src.retriever import DEFAULT_MODEL_NAME, MODEL_LICENSE, MODEL_REVISION


def get_library_versions() -> dict[str, str]:
    """Records exact versions of all relevant modeling libraries."""
    return {
        "sentence_transformers": sentence_transformers.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
    }


def compute_urls_sha256(urls: list[str]) -> str:
    """Computes SHA-256 hex digest of newline-joined ordered corpus URLs."""
    urls_text = "\n".join(str(u).strip() for u in urls)
    return hashlib.sha256(urls_text.encode("utf-8")).hexdigest()


def verify_cached_embeddings_urls(cached_embeddings_path: str | Path, current_urls: list[str]) -> str:
    """
    Verifies that the cached embeddings .npy corresponds exactly to the ordered list of corpus URLs.
    Looks for <stem>.urls_sha256 or <stem>.urls.txt next to the .npy file.
    Raises FileNotFoundError if no verification file exists next to the .npy.
    Raises ValueError if the SHA-256 does not match the current corpus URLs.
    Returns the verified SHA-256 hex digest.
    """
    cached_path = Path(cached_embeddings_path)
    if not cached_path.exists():
        raise FileNotFoundError(f"Cached embeddings file not found: {cached_path}")

    # Look for verification files next to the .npy
    candidates_sha = [
        cached_path.with_name(f"{cached_path.stem}.urls_sha256"),
        cached_path.with_name(f"{cached_path.name}.urls_sha256"),
        cached_path.parent / "retriever_embeddings.urls_sha256",
    ]
    candidates_txt = [
        cached_path.with_name(f"{cached_path.stem}.urls.txt"),
        cached_path.with_name(f"{cached_path.name}.urls.txt"),
        cached_path.parent / "retriever_embeddings.urls.txt",
    ]

    expected_hash: str | None = None
    for p in candidates_sha:
        if p.exists():
            expected_hash = p.read_text(encoding="utf-8").strip().splitlines()[0].strip()
            break

    if expected_hash is None:
        for p in candidates_txt:
            if p.exists():
                lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
                expected_hash = compute_urls_sha256(lines)
                break

    if expected_hash is None:
        raise FileNotFoundError(
            f"Missing URL verification file next to cached embeddings at '{cached_path}'. "
            f"Expected '{cached_path.stem}.urls_sha256' or '{cached_path.stem}.urls.txt'."
        )

    current_hash = compute_urls_sha256(current_urls)
    if current_hash != expected_hash:
        raise ValueError(
            f"Corpus URL SHA-256 mismatch for cached embeddings '{cached_path}'. "
            f"Expected SHA-256 {expected_hash}, but current corpus has {current_hash}. "
            "Refusing to use cached embeddings that do not align with the ordered corpus."
        )

    return current_hash


def build_retriever_artifacts(
    data_dir: str | Path = "data/processed",
    output_dir: str | Path = "models",
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cpu",
    seed: int = 42,
    batch_size: int = 64,
    cached_embeddings_path: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Builds and writes retriever embeddings, metadata, and info JSON into output_dir.
    """
    t0 = time.time()
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    if verbose:
        print(f"Loading corpus from {data_path}...")

    df_syn = pd.read_csv(data_path / "synopses_clean.csv")
    df_lab = pd.read_csv(data_path / "dramas_labeled.csv")[
        [
            "url",
            "name",
            "release_year",
            "country",
            "original_network",
            "target_tier",
            "target_pct",
            "start_date",
        ]
    ]

    df = pd.merge(df_syn, df_lab, on="url")
    valid_mask = df["synopsis_clean"].astype(str).str.strip().str.len() > 0
    df_valid = df[valid_mask].copy().reset_index(drop=True)

    if verbose:
        print(f"Corpus size (non-empty synopses): {len(df_valid):,} dramas")

    corpus_urls_list = df_valid["url"].astype(str).tolist()
    corpus_urls_sha = compute_urls_sha256(corpus_urls_list)

    # 1. Embeddings
    if cached_embeddings_path:
        cached_p = Path(cached_embeddings_path)
        if not cached_p.exists():
            raise FileNotFoundError(f"Cached embeddings file not found: {cached_p}")

        # Verify URL SHA-256 stored next to the .npy
        verified_sha = verify_cached_embeddings_urls(cached_p, corpus_urls_list)
        if verbose:
            print(f"Verified corpus URL SHA-256 ({verified_sha[:10]}...) for cached embeddings.")
            print(f"Loading cached embeddings from {cached_p}...")
        embeddings = np.load(cached_p).astype(np.float32)

        if embeddings.shape[0] != len(df_valid):
            raise ValueError(
                f"Embedding row count ({embeddings.shape[0]}) does not match corpus size ({len(df_valid)})"
            )
    else:
        if verbose:
            print(f"Encoding synopses with {model_name} on {device} (batch_size={batch_size})...")
        model = SentenceTransformer(model_name, device=device)
        embeddings = model.encode(
            df_valid["synopsis_clean"].tolist(),
            batch_size=batch_size,
            show_progress_bar=verbose,
            normalize_embeddings=True,
        ).astype(np.float32)

    # Unit-normalize vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = (embeddings / norms).astype(np.float32)

    emb_file = out_path / "retriever_embeddings.npy"
    np.save(emb_file, embeddings)

    # Save ordered corpus URLs and SHA-256 next to the embeddings
    urls_file = out_path / "retriever_embeddings.urls.txt"
    urls_sha_file = out_path / "retriever_embeddings.urls_sha256"
    with open(urls_file, "w", encoding="utf-8") as f:
        f.write("\n".join(corpus_urls_list) + "\n")
    with open(urls_sha_file, "w", encoding="utf-8") as f:
        f.write(corpus_urls_sha + "\n")

    # 2. Metadata table (Strictly NO synopsis column; network=None for Unknown)
    raw_networks = df_valid["original_network"].fillna("Unknown").astype(str).str.strip()
    clean_networks = raw_networks.apply(lambda x: None if x in ["", "Unknown"] else x)

    clean_start_dates = df_valid["start_date"].apply(
        lambda x: None if pd.isna(x) or str(x).strip() in ["", "nan"] else str(x).strip()
    )

    metadata_df = pd.DataFrame(
        {
            "name": df_valid["name_y"] if "name_y" in df_valid else df_valid["name"],
            "release_year": df_valid["release_year_y"].astype(int)
            if "release_year_y" in df_valid
            else df_valid["release_year"].astype(int),
            "country": df_valid["country_y"] if "country_y" in df_valid else df_valid["country"],
            "network": clean_networks,
            "target_tier": df_valid["target_tier"].astype(str),
            "target_pct": df_valid["target_pct"].astype(float),
            "url": df_valid["url"].astype(str),
            "start_date": clean_start_dates,
        }
    )

    meta_file = out_path / "retriever_metadata.parquet"
    metadata_df.to_parquet(meta_file, index=False)

    # 3. Info JSON
    info_dict = {
        "model_name": model_name,
        "model_revision": MODEL_REVISION,
        "model_license": MODEL_LICENSE,
        "seed": seed,
        "library_versions": get_library_versions(),
        "n_dramas": len(df_valid),
        "corpus_urls_sha256": corpus_urls_sha,
        "corpus_urls_path": urls_file.name,
        "embedding_dim": int(embeddings.shape[1]),
        "embeddings_path": emb_file.name,
        "embeddings_bytes": int(emb_file.stat().st_size),
        "metadata_path": meta_file.name,
        "metadata_bytes": int(meta_file.stat().st_size),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    info_file = out_path / "retriever_info.json"
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(info_dict, f, indent=2)

    elapsed = time.time() - t0
    if verbose:
        print(f"Artifacts successfully written to {out_path}:")
        print(f"  {emb_file.name} ({emb_file.stat().st_size / (1024*1024):.2f} MB)")
        print(f"  {meta_file.name} ({meta_file.stat().st_size / 1024:.2f} KB)")
        print(f"  {info_file.name} ({info_file.stat().st_size} bytes)")
        print(f"  {urls_sha_file.name} (SHA-256: {corpus_urls_sha[:12]}...)")
        print(f"Done in {elapsed:.2f}s!")

    return info_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build retriever artifacts for Drama Success Copilot.")
    parser.add_argument("--data-dir", default="data/processed", help="Path to data/processed directory")
    parser.add_argument("--output-dir", default="models", help="Destination directory for serialized artifacts")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="SentenceTransformer model name")
    parser.add_argument("--device", default="cpu", help="Computation device (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding inference")
    parser.add_argument("--cached-embeddings", default=None, help="Path to precomputed embeddings .npy")
    args = parser.parse_args()

    build_retriever_artifacts(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        cached_embeddings_path=args.cached_embeddings,
    )
