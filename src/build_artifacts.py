"""
Build Artifacts Module for Drama Success Copilot
-----------------------------------------------
Rebuilds and serializes all production model and explainability artifacts into models/:
1. headline_model.joblib: Fitted HistGradientBoostingClassifier on V2 headline features (years <= 2021).
2. isotonic_calibrator.joblib: Fitted IsotonicRegression on expanding-window out-of-fold validation predictions.
3. background_sample.joblib: 100-drama background sample drawn from training features (seed=42).
4. talent_lookup.joblib: Pre-release creator track record lookup across all 7,535 labeled dramas.
5. metadata.joblib: Feature definitions, category encodings, vocabularies, and runtime metadata.

Usage:
    python src/build_artifacts.py [--output-dir models] [--data-dir data/processed]
"""

import argparse
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import OrdinalEncoder

from src.features import (
    FEATURE_GROUPS,
    build_talent_inverted_index,
    fit_basic_features,
    load_labeled_data,
    prepare_talent_columns,
)
from src.modeling import get_expanding_window_folds, load_modeling_data


def build_all_artifacts(
    data_dir: str = "data/processed",
    output_dir: str = "models",
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Train headline model, fit calibrator, build talent lookup, and save artifacts.
    """
    t0 = time.time()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Modeling Features
    labeled_file = str(Path(data_dir) / "dramas_labeled.csv")
    train_df, test_df = load_modeling_data(
        variant="v2",
        features_dir=data_dir,
        labeled_file=labeled_file,
    )
    if verbose:
        print(f"Loaded modeling data: train={len(train_df):,} rows, test={len(test_df):,} rows")

    # Feature subset
    features = FEATURE_GROUPS["core"] + FEATURE_GROUPS["talent_reach"]
    cat_cols = [c for c in ["country", "original_network", "content_rating"] if c in features]
    cat_indices = [features.index(c) for c in cat_cols]

    # Categorical encoding
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train = train_df[features].copy()
    X_train[cat_cols] = enc.fit_transform(X_train[cat_cols].astype(str))
    X_train_num = X_train.astype(np.float64)

    # 2. Fit Headline Model
    clf = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        categorical_features=cat_indices,
        random_state=seed,
    )
    clf.fit(X_train_num, train_df["target_tier"])
    if verbose:
        print(f"Fitted headline HistGradientBoostingClassifier on train (N={len(train_df):,})")

    # 3. Expanding-Window Pooled OOF Calibration
    folds_all = get_expanding_window_folds(train_df)
    oof_probs, oof_y = [], []
    for f in folds_all:
        f_tr = train_df[f["tr_mask"]].reset_index(drop=True)
        f_val = train_df[f["val_mask"]].reset_index(drop=True)
        X_tr_f = X_train_num.loc[f["tr_mask"]]
        X_val_f = X_train_num.loc[f["val_mask"]]

        clf_f = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            categorical_features=cat_indices,
            random_state=seed,
        )
        clf_f.fit(X_tr_f, f_tr["target_tier"])
        h_idx = list(clf_f.classes_).index("High")
        oof_probs.append(clf_f.predict_proba(X_val_f)[:, h_idx])
        oof_y.append((f_val["target_tier"].values == "High").astype(int))

    oof_probs = np.concatenate(oof_probs)
    oof_y = np.concatenate(oof_y)

    iso_calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso_calibrator.fit(oof_probs, oof_y)
    if verbose:
        print(f"Fitted IsotonicRegression on pooled expanding-window OOF predictions (N={len(oof_y):,})")

    # 4. Background Sample (100 training dramas)
    bg_sample = X_train_num.sample(n=100, random_state=seed)

    # 5. Talent Lookup from ALL 7,535 Labeled Dramas
    labeled_df = load_labeled_data(labeled_file)
    all_prep = prepare_talent_columns(labeled_df)
    dir_idx_all, scr_idx_all, cast_idx_all = build_talent_inverted_index(all_prep)

    dir_lookup = {person: [(date, float(target)) for date, target, _, _ in entries] for person, entries in dir_idx_all.items()}
    scr_lookup = {person: [(date, float(target)) for date, target, _, _ in entries] for person, entries in scr_idx_all.items()}
    cast_lookup = {person: [(date, float(target)) for date, target, _, _ in entries] for person, entries in cast_idx_all.items()}

    talent_lookup = {
        "director": dir_lookup,
        "screenwriter": scr_lookup,
        "cast": cast_lookup,
        "global_mean": 0.5022,
        "smoothing_m": 3.0,
        "n_dramas_indexed": len(labeled_df),
    }
    if verbose:
        print(f"Indexed talent: {len(dir_lookup):,} directors, {len(scr_lookup):,} screenwriters, {len(cast_lookup):,} actors from {len(labeled_df):,} dramas")

    # 6. Metadata Bundle
    train_lab = labeled_df[labeled_df["release_year"] <= 2021].reset_index(drop=True)
    basic_params = fit_basic_features(train_lab)
    metadata = {
        "features": features,
        "cat_cols": cat_cols,
        "cat_indices": cat_indices,
        "ordinal_encoder": enc,
        "valid_networks": sorted(list(basic_params["valid_networks"])),
        "valid_genres": basic_params["valid_genres"],
        "weekdays": basic_params["weekdays"],
        "rating_map": basic_params["rating_map"],
        "valid_countries": basic_params["countries"],
        "classes": list(clf.classes_),
        "high_class_idx": list(clf.classes_).index("High"),
        "sklearn_version": sklearn.__version__,
        "shap_version": shap.__version__,
        "global_mean": 0.5022,
        "smoothing_m": 3.0,
        "train_episode_range": (int(train_df["episodes"].min()), int(train_df["episodes"].max())),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # 7. Serialize Artifacts
    artifacts = {
        "headline_model.joblib": clf,
        "isotonic_calibrator.joblib": iso_calibrator,
        "background_sample.joblib": bg_sample,
        "talent_lookup.joblib": talent_lookup,
        "metadata.joblib": metadata,
    }

    saved_paths = {}
    for fname, obj in artifacts.items():
        dst = out_path / fname
        joblib.dump(obj, dst, compress=3)
        saved_paths[fname] = dst
        if verbose:
            sz_kb = dst.stat().st_size / 1024
            print(f"  Saved {fname:30s} ({sz_kb:6.1f} KB)")

    if verbose:
        print(f"Artifact build complete in {time.time() - t0:.2f}s!")

    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and serialize production artifacts for Drama Success Copilot.")
    parser.add_argument("--data-dir", default="data/processed", help="Path to data/processed directory")
    parser.add_argument("--output-dir", default="models", help="Destination directory for serialized artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    build_all_artifacts(data_dir=args.data_dir, output_dir=args.output_dir, seed=args.seed)
