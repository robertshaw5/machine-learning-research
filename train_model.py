"""
Train one LightGBM classifier on feature_dataset.csv

Excludes race metadata (ids, venue, dates, race descriptors, stakes), starting price
(sp / sp_decimal), and targets.

Writes test-row predictions to a CSV and prints top-pick accuracy.

Probabilities:raw LightGBM marginals are poorly calibrated for absolute win rates. We fit
IsotonicRegression on the chronological validation split (marginal p vs outcome), refit the
model on train+val, apply that mapping to test marginals (p_win, p_win_raw), and add p_win_race by
renormalizing p_win within each race so they sum to 1.

Ranking: model_rank is 1 for the highest p_win_race in each race_id, 2 for the
next, and so on.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

# --- edit these ---
DATA_PATH = "datasets/feature_dataset.csv"
PREDICTIONS_PATH = "outputs/test_predictions.csv"
SAVE_PREDICTIONS = True

# Not used as model inputs (race / meeting context, identifiers).
RACE_META_COLS = [
    "race_id",
    "greyhound",
    "meeting_venue",
    "meeting_date",
    "race_surface",
    "prize_eur",
    "track_code",
    "race_number",
    "race_grade_ordinal",
    "race_distance",
    "wt",
]

NON_FEATURE_COLS = frozenset({"target_pos", "target_winner", "sp", "sp_decimal"})
TARGET_COL = "target_winner"


def load_xy(path: str = DATA_PATH):
    df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    if TARGET_COL not in df.columns:
        raise ValueError(f"CSV must include column {TARGET_COL!r}")

    y = df[TARGET_COL]
    mask = y.notna()
    df = df.loc[mask].copy()
    y = df[TARGET_COL]

    groups = df["race_id"] if "race_id" in df.columns else None
    meta_cols = [c for c in ("race_id", "greyhound", "meeting_date", "sp", "sp_decimal") if c in df.columns]
    meta = df[meta_cols].copy()

    drop_x = set(RACE_META_COLS) | NON_FEATURE_COLS
    feature_cols = [c for c in df.columns if c not in drop_x]
    X = df[feature_cols].select_dtypes(include=["number", "bool"])

    return X, y, groups, meta


def chronological_train_val_test_indices(
    groups: pd.Series,
    meeting_date: pd.Series,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-5:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1")
    g = groups.reset_index(drop=True).astype(str)
    raw = meeting_date.reset_index(drop=True).astype(str).str.strip()
    dt = pd.to_datetime(raw, format="%d-%b-%y", errors="coerce")
    need_fb = dt.isna() & raw.notna() & ~raw.str.lower().isin(("", "nan", "<na>", "none"))
    if need_fb.any():
        dt = dt.copy()
        dt.loc[need_fb] = pd.to_datetime(raw.loc[need_fb], errors="coerce", dayfirst=True)

    race_dt = pd.DataFrame({"rid": g, "dt": dt}).groupby("rid", sort=False)["dt"].min()
    race_dt = race_dt.reset_index().sort_values(["dt", "rid"], ascending=[True, True], na_position="last")
    ordered_races = race_dt["rid"].astype(str).tolist()
    n_r = len(ordered_races)
    if n_r < 3:
        raise ValueError("Need at least 3 distinct races for chronological train/val/test.")

    n_te = max(1, int(round(n_r * test_frac)))
    n_va = max(1, int(round(n_r * val_frac)))
    n_tr = n_r - n_te - n_va
    while n_tr < 1 and (n_te > 1 or n_va > 1):
        if n_te > 1:
            n_te -= 1
        elif n_va > 1:
            n_va -= 1
        n_tr = n_r - n_te - n_va
    if n_tr < 1:
        raise ValueError("Not enough races for this split; adjust fractions.")

    tr_set = set(ordered_races[:n_tr])
    va_set = set(ordered_races[n_tr : n_tr + n_va])
    te_set = set(ordered_races[n_tr + n_va :])
    train_mask = g.isin(tr_set).to_numpy()
    val_mask = g.isin(va_set).to_numpy()
    test_mask = g.isin(te_set).to_numpy()
    return np.flatnonzero(train_mask), np.flatnonzero(val_mask), np.flatnonzero(test_mask)


def top_pick_accuracy(
    race_id: pd.Series, y: np.ndarray, proba: np.ndarray
) -> Tuple[float, int, int]:
    """
    Per race: pick the runner with highest predicted P(win). Accuracy = share of races
    where that runner is the actual winner (exactly one winner per race expected).
    """
    g = race_id.reset_index(drop=True).astype(str)
    yv = np.asarray(y, dtype=float).ravel()
    pv = np.asarray(proba, dtype=float).ravel()
    df = pd.DataFrame({"g": g, "y": yv, "p": pv})
    hits = 0
    n_races = 0
    for _, sub in df.groupby("g", sort=False):
        j = int(sub["p"].values.argmax())
        if float(sub["y"].iloc[j]) == 1.0:
            hits += 1
        n_races += 1
    return hits / max(n_races, 1), hits, n_races


def race_renormalize(race_id: pd.Series, p: np.ndarray) -> np.ndarray:
    """Divide by the sum of p within each race so probabilities sum to 1 per race."""
    g = race_id.reset_index(drop=True).astype(str)
    pv = np.asarray(p, dtype=float).ravel()
    out = np.empty_like(pv)
    for rid in g.unique():
        m = (g == rid).to_numpy()
        s = float(pv[m].sum())
        if s <= 0 or not np.isfinite(s):
            n = int(m.sum())
            out[m] = 1.0 / max(n, 1)
        else:
            out[m] = pv[m] / s
    return out


def run() -> None:
    X, y, groups, meta = load_xy(DATA_PATH)
    if groups is None:
        raise ValueError("Column race_id is required for chronological split.")
    if "meeting_date" not in meta.columns:
        raise ValueError("Column meeting_date is required for chronological split.")

    fit_idx, val_idx, test_idx = chronological_train_val_test_indices(
        groups, meta["meeting_date"]
    )
    combine = np.concatenate([fit_idx, val_idx])
    X_fit, y_fit = X.iloc[fit_idx], y.iloc[fit_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_train, y_train = X.iloc[combine], y.iloc[combine]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    g_test = groups.iloc[test_idx]
    meta_test = meta.iloc[test_idx].reset_index(drop=True)

    imputer = SimpleImputer(strategy="median")
    if hasattr(imputer, "set_output"):
        imputer.set_output(transform="pandas")

    model = Pipeline(
        [
            ("imputer", imputer),
            (
                "clf",
                LGBMClassifier(
                    n_estimators=200,
                    max_depth=12,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ]
    )

    # 1) Fit on train only → val marginals for isotonic calibration
    model.fit(X_fit, y_fit)
    p_val = model.predict_proba(X_val)[:, 1]
    y_val_arr = y_val.to_numpy(dtype=float)
    iso: Optional[IsotonicRegression] = None
    if np.unique(y_val_arr).size >= 2:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_val, y_val_arr)
    # 2) Refit on train+val for final model
    model.fit(X_train, y_train)

    proba_raw = model.predict_proba(X_test)[:, 1]
    if iso is not None:
        proba_cal = np.clip(iso.predict(proba_raw), 0.0, 1.0)
    else:
        proba_cal = proba_raw
    proba_race = race_renormalize(g_test, proba_cal)

    pred = model.predict(X_test)
    y_arr = y_test.to_numpy(dtype=float)

    print("LightGBM | chronological split (train+val earliest, test latest)")
    print(
        "  Train:",
        len(X_fit),
        "| Val:",
        len(X_val),
        "| Train+val:",
        len(X_train),
        "| Test:",
        len(X_test),
        "| Test races:",
        g_test.nunique(),
    )
    if iso is None:
        print("  Note: isotonic calibration skipped (validation split has a single class).")
    print(classification_report(y_test, pred, digits=3))
    try:
        print("ROC-AUC (raw marginal):", round(roc_auc_score(y_test, proba_raw), 4))
    except ValueError:
        print("ROC-AUC: n/a")
    print("Brier (raw marginal):", round(brier_score_loss(y_arr, proba_raw), 4))
    print("Brier (isotonic marginal):", round(brier_score_loss(y_arr, proba_cal), 4))

    top_acc, top_hits, n_races_eval = top_pick_accuracy(g_test, y_arr, proba_race)
    print(
        "Top-pick accuracy (highest p_win_race per race):",
        round(float(top_acc), 4),
        f"({top_hits}/{n_races_eval} races)",
    )

    if SAVE_PREDICTIONS:
        out_dir = os.path.dirname(PREDICTIONS_PATH)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out = meta_test.copy()
        out["target_winner"] = y_arr
        out["p_win_raw"] = proba_raw
        out["p_win"] = proba_cal
        out["p_win_race"] = proba_race
        out["model_rank"] = (
            out.groupby("race_id", sort=False)["p_win_race"]
            .rank(ascending=False, method="first")
            .astype(int)
        )
        out.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
        print(f"Wrote test predictions: {PREDICTIONS_PATH} ({len(out)} rows)")


if __name__ == "__main__":
    run()
