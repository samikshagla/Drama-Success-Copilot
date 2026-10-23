"""
Modeling Module for Drama Success Copilot (Milestone 4)
------------------------------------------------------
Implements:
1. Feature dataset loading (V2 Headline, V1 Deployment benchmark).
2. Time-ordered train/val splitting and expanding-window CV folds.
3. Multi-class & ranking evaluation metrics with 500-fold bootstrap 95% CIs.
4. Baseline estimators (Majority Class, Stratified Random, Country-Only, Page-Completeness).
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def load_modeling_data(
    variant: str = "v2",
    features_dir: str = "data/processed",
    labeled_file: str = "data/processed/dramas_labeled.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test feature tables, and attach n_missing_credits from dramas_labeled.csv.
    
    Parameters:
        variant: 'v2' (Headline static train-only priors) or 'v1' (rolling deployment priors).
        features_dir: Path to directory containing train_features_*.csv.
        labeled_file: Path to dramas_labeled.csv.
        
    Returns:
        (train_df, test_df)
    """
    tr_path = f"{features_dir}/train_features_{variant.lower()}.csv"
    te_path = f"{features_dir}/test_features_{variant.lower()}.csv"
    
    train_df = pd.read_csv(tr_path)
    test_df = pd.read_csv(te_path)
    
    # Load and merge n_missing_credits
    lab_df = pd.read_csv(labeled_file)
    tr_lab = lab_df[lab_df["release_year"] <= 2021].reset_index(drop=True)
    te_lab = lab_df[lab_df["release_year"] > 2021].reset_index(drop=True)
    
    assert len(train_df) == len(tr_lab), "Train row count mismatch with labeled dataset!"
    assert len(test_df) == len(te_lab), "Test row count mismatch with labeled dataset!"
    
    train_df["n_missing_credits"] = tr_lab["n_missing_credits"].values
    test_df["n_missing_credits"] = te_lab["n_missing_credits"].values
    
    return train_df, test_df


def get_temporal_validation_split(
    train_df: pd.DataFrame,
    val_cutoff_year: int = 2019
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split training data into internal train (<= val_cutoff_year) and validation (> val_cutoff_year).
    Default: train on <= 2019 (N=3,520), validate on 2020-2021 (N=1,490).
    """
    tr_sub = train_df[train_df["release_year"] <= val_cutoff_year].copy().reset_index(drop=True)
    val_sub = train_df[train_df["release_year"] > val_cutoff_year].copy().reset_index(drop=True)
    return tr_sub, val_sub


def get_expanding_window_folds(train_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate time-ordered expanding-window validation folds within the training set.
    """
    folds = [
        {"fold": 1, "train_years": "<= 2017", "val_years": "2018", "val_mask": train_df["release_year"] == 2018, "tr_mask": train_df["release_year"] <= 2017},
        {"fold": 2, "train_years": "<= 2018", "val_years": "2019", "val_mask": train_df["release_year"] == 2019, "tr_mask": train_df["release_year"] <= 2018},
        {"fold": 3, "train_years": "<= 2019", "val_years": "2020-2021", "val_mask": train_df["release_year"] >= 2020, "tr_mask": train_df["release_year"] <= 2019},
    ]
    for f in folds:
        f["train_size"] = int(f["tr_mask"].sum())
        f["val_size"] = int(f["val_mask"].sum())
    return folds


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_high: Optional[np.ndarray] = None,
    target_pct: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute standard evaluation metrics across classification and ranking dimensions.
    """
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    
    y_true_binary = (y_true == "High").astype(int)
    
    if y_prob_high is not None and len(np.unique(y_prob_high)) > 1:
        roc_auc = float(roc_auc_score(y_true_binary, y_prob_high))
        pr_auc = float(average_precision_score(y_true_binary, y_prob_high))
    else:
        roc_auc = 0.5
        pr_auc = float(y_true_binary.mean())
        
    if target_pct is not None and y_prob_high is not None and len(np.unique(y_prob_high)) > 1:
        res = spearmanr(y_prob_high, target_pct)
        spearman = float(res.statistic if hasattr(res, "statistic") else res[0])
    else:
        spearman = np.nan
        
    return {
        "Accuracy": acc,
        "Macro-F1": macro_f1,
        "Balanced Acc": bal_acc,
        "ROC-AUC (High)": roc_auc,
        "PR-AUC (High)": pr_auc,
        "Spearman": spearman,
    }


def bootstrap_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_high: Optional[np.ndarray] = None,
    target_pct: Optional[np.ndarray] = None,
    n_boot: int = 500,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate predictions with a 500-fold bootstrap over test samples to compute 95% CIs.
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    point_est = compute_metrics(y_true, y_pred, y_prob_high, target_pct)
    
    boot_metrics: Dict[str, List[float]] = {k: [] for k in point_est.keys()}
    indices = np.arange(n)
    
    for _ in range(n_boot):
        b_idx = rng.choice(indices, size=n, replace=True)
        b_true = y_true[b_idx]
        b_pred = y_pred[b_idx]
        b_prob = y_prob_high[b_idx] if y_prob_high is not None else None
        b_pct = target_pct[b_idx] if target_pct is not None else None
        
        m = compute_metrics(b_true, b_pred, b_prob, b_pct)
        for k, v in m.items():
            boot_metrics[k].append(v)
            
    summary: Dict[str, Dict[str, Any]] = {}
    for k, pt in point_est.items():
        arr = np.array(boot_metrics[k])
        if np.all(np.isnan(arr)) or np.isnan(pt):
            ci_low, ci_high = np.nan, np.nan
            formatted = "N/A"
        else:
            arr_valid = arr[~np.isnan(arr)]
            ci_low = float(np.percentile(arr_valid, 2.5))
            ci_high = float(np.percentile(arr_valid, 97.5))
            formatted = f"{pt:.3f} [{ci_low:.3f}, {ci_high:.3f}]"
            
        summary[k] = {
            "point": pt,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "str": formatted,
        }
        
    return summary


# =====================================================================
# Step 1: Baseline Estimators
# =====================================================================

class MajorityClassBaseline:
    """Always predicts 'Medium' (the 50% majority class)."""
    def __init__(self, majority_class: str = "Medium"):
        self.majority_class = majority_class
        
    def fit(self, X: Any, y: Any):
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self.majority_class] * len(X))
        
    def predict_proba_high(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=float)


class StratifiedRandomBaseline:
    """Randomly samples classes according to training set class frequencies."""
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.classes_ = ["Low", "Medium", "High"]
        self.probs_ = [0.25, 0.50, 0.25]
        
    def fit(self, X: Any, y: pd.Series):
        counts = pd.Series(y).value_counts(normalize=True)
        self.classes_ = list(counts.index)
        self.probs_ = list(counts.values)
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        rng = np.random.RandomState(self.random_state)
        return rng.choice(self.classes_, size=len(X), p=self.probs_)
        
    def predict_proba_high(self, X: pd.DataFrame) -> np.ndarray:
        rng = np.random.RandomState(self.random_state)
        high_prob = self.probs_[self.classes_.index("High")] if "High" in self.classes_ else 0.25
        return rng.uniform(0.0, 1.0, size=len(X)) * high_prob


class CountryOnlyBaseline:
    """Multinomial Logistic Regression using strictly country dummy variables."""
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.encoder = OneHotEncoder(drop="first", sparse_output=False)
        self.clf = LogisticRegression(max_iter=500, random_state=random_state)
        self.classes_: List[str] = []
        
    def fit(self, train_df: pd.DataFrame, y_train: pd.Series):
        X_ohe = self.encoder.fit_transform(train_df[["country"]])
        self.clf.fit(X_ohe, y_train)
        self.classes_ = list(self.clf.classes_)
        return self
        
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        X_ohe = self.encoder.transform(test_df[["country"]])
        return self.clf.predict(X_ohe)
        
    def predict_proba_high(self, test_df: pd.DataFrame) -> np.ndarray:
        X_ohe = self.encoder.transform(test_df[["country"]])
        idx = self.classes_.index("High")
        return self.clf.predict_proba(X_ohe)[:, idx]


class PageCompletenessBaseline:
    """Logistic Regression trained strictly on missingness flags and n_missing_credits."""
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_cols = [
            "aired_on_is_missing",
            "content_rating_is_missing",
            "network_is_missing",
            "genre_is_missing",
            "n_missing_credits",
        ]
        self.clf = LogisticRegression(max_iter=500, random_state=random_state)
        self.classes_: List[str] = []
        
    def fit(self, train_df: pd.DataFrame, y_train: pd.Series):
        X = train_df[self.feature_cols].values
        self.clf.fit(X, y_train)
        self.classes_ = list(self.clf.classes_)
        return self
        
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        X = test_df[self.feature_cols].values
        return self.clf.predict(X)
        
    def predict_proba_high(self, test_df: pd.DataFrame) -> np.ndarray:
        X = test_df[self.feature_cols].values
        idx = self.classes_.index("High")
        return self.clf.predict_proba(X)[:, idx]


# =====================================================================
# Step 2: Model Training & Paired Difference Testing
# =====================================================================

HGB_TUNED_PARAMS = {
    "max_iter": 100,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "random_state": 42,
}


def paired_bootstrap_comparison(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    prob_a: np.ndarray,
    pred_b: np.ndarray,
    prob_b: np.ndarray,
    target_pct: Optional[np.ndarray] = None,
    n_boot: int = 500,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute paired bootstrap difference distributions (Model A - Model B)
    across identical resamples of test dramas.
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    indices = np.arange(n)
    
    pt_a = compute_metrics(y_true, pred_a, prob_a, target_pct)
    pt_b = compute_metrics(y_true, pred_b, prob_b, target_pct)
    
    diff_roc: List[float] = []
    diff_f1: List[float] = []
    
    for _ in range(n_boot):
        b_idx = rng.choice(indices, size=n, replace=True)
        m_a = compute_metrics(y_true[b_idx], pred_a[b_idx], prob_a[b_idx], target_pct[b_idx] if target_pct is not None else None)
        m_b = compute_metrics(y_true[b_idx], pred_b[b_idx], prob_b[b_idx], target_pct[b_idx] if target_pct is not None else None)
        diff_roc.append(m_a["ROC-AUC (High)"] - m_b["ROC-AUC (High)"])
        diff_f1.append(m_a["Macro-F1"] - m_b["Macro-F1"])
        
    diff_roc_arr = np.array(diff_roc)
    diff_f1_arr = np.array(diff_f1)
    
    pt_diff_roc = pt_a["ROC-AUC (High)"] - pt_b["ROC-AUC (High)"]
    pt_diff_f1 = pt_a["Macro-F1"] - pt_b["Macro-F1"]
    
    ci_roc = (float(np.percentile(diff_roc_arr, 2.5)), float(np.percentile(diff_roc_arr, 97.5)))
    ci_f1 = (float(np.percentile(diff_f1_arr, 2.5)), float(np.percentile(diff_f1_arr, 97.5)))
    
    ex_roc = (ci_roc[0] > 0 and ci_roc[1] > 0) or (ci_roc[0] < 0 and ci_roc[1] < 0)
    ex_f1 = (ci_f1[0] > 0 and ci_f1[1] > 0) or (ci_f1[0] < 0 and ci_f1[1] < 0)
    
    return {
        "delta_roc": {
            "point": pt_diff_roc,
            "ci_low": ci_roc[0],
            "ci_high": ci_roc[1],
            "excludes_0": ex_roc,
            "str": f"{pt_diff_roc:+.3f} [{ci_roc[0]:+.3f}, {ci_roc[1]:+.3f}]" + (" *" if ex_roc else ""),
        },
        "delta_f1": {
            "point": pt_diff_f1,
            "ci_low": ci_f1[0],
            "ci_high": ci_f1[1],
            "excludes_0": ex_f1,
            "str": f"{pt_diff_f1:+.3f} [{ci_f1[0]:+.3f}, {ci_f1[1]:+.3f}]" + (" *" if ex_f1 else ""),
        },
    }


def train_and_predict_hgb(
    tr_df: pd.DataFrame,
    te_df: pd.DataFrame,
    feature_cols: List[str],
    params: Optional[dict] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Train HistGradientBoostingClassifier and return (y_pred, y_prob_high)."""
    p = params or HGB_TUNED_PARAMS
    cat_cols = [c for c in ["country", "original_network", "content_rating"] if c in feature_cols]
    cat_indices = [feature_cols.index(c) for c in cat_cols]
    
    X_tr = tr_df[feature_cols].copy()
    X_te = te_df[feature_cols].copy()
    
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan)
        X_tr[cat_cols] = enc.fit_transform(X_tr[cat_cols])
        X_te[cat_cols] = enc.transform(X_te[cat_cols])
        
    clf = HistGradientBoostingClassifier(
        **p,
        categorical_features=cat_indices if cat_indices else None,
    )
    clf.fit(X_tr.values, tr_df["target_tier"].values)
    
    high_idx = list(clf.classes_).index("High")
    y_prob_high = clf.predict_proba(X_te.values)[:, high_idx]
    y_pred = clf.predict(X_te.values)
    return y_pred, y_prob_high


def train_and_predict_linear(
    tr_df: pd.DataFrame,
    te_df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Train LogisticRegression reference pipeline and return (y_pred, y_prob_high)."""
    cat_cols = [c for c in ["country", "original_network", "content_rating"] if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    
    transformers = []
    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipe, cat_cols))
        
    if num_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipe, num_cols))
        
    preprocessor = ColumnTransformer(transformers=transformers)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    
    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", clf)
    ])
    pipe.fit(tr_df[feature_cols], tr_df["target_tier"].values)
    
    classes = list(pipe.named_steps["clf"].classes_)
    high_idx = classes.index("High")
    y_prob_high = pipe.predict_proba(te_df[feature_cols])[:, high_idx]
    y_pred = pipe.predict(te_df[feature_cols])
    return y_pred, y_prob_high

