# Adds fair price, edge and Kelly staking columns to the test predictions csv. 
# Outputs a table showing the number of bets and ROI for each min-edge cutoff.

import os

import numpy as np
import pandas as pd

# --- edit these ---
INPUT_CSV = "outputs/test_predictions.csv"
OUT_CSV = "outputs/test_predictions_strategy.csv"
MATRIX_CSV = "outputs/kelly_edge_matrix.csv"
WRITE_FILES = True

BANKROLL = 1.0  # scales dollar stakes in the ROI calc

#When set to True, only bets the model's top pick per race.
TOP_PICK_ONLY = True

# one row per value: only bets with edge >= this
EDGE_MINS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7)


def kelly_arrays(p_win, sp):
    """fair price, edge, full Kelly fraction of bankroll."""
    p = np.asarray(p_win, float)
    sp = np.asarray(sp, float)

    with np.errstate(divide="ignore", invalid="ignore"):
        fair = np.where(p > 0, 1.0 / p, np.nan)
    edge = np.where(np.isfinite(fair) & np.isfinite(sp), sp / fair - 1.0, np.nan)

    d = sp - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        kfull = np.where((d > 1e-12) & np.isfinite(edge), edge / d, np.nan)
    return fair, edge, kfull


def top_pick_only(df):
    if "model_rank" in df.columns:
        return df["model_rank"] == 1
    rnk = df.groupby("race_id", sort=False)["p_win_race"].rank(ascending=False, method="first")
    return rnk == 1


def edge_roi_table(df, edge_mins, bankroll):
    """One row per edge threshold: ROI (decimal), n bets, stake"""
    for c in ("kelly_full", "edge", "target_winner", "sp_decimal"):
        if c not in df.columns:
            raise ValueError("missing column: " + c)

    kf = df["kelly_full"].to_numpy(float)
    ed = df["edge"].to_numpy(float)
    w = df["target_winner"].to_numpy(float)
    sp = df["sp_decimal"].to_numpy(float)

    rows = []
    for emin in edge_mins:
        m = (kf > 0) & np.isfinite(ed) & (ed >= emin) & np.isfinite(kf)
        stk = kf[m] * bankroll
        ww, ssp = w[m], sp[m]
        ok = np.isfinite(stk) & np.isfinite(ssp) & np.isfinite(ww) & (stk > 0)
        stk, ww, ssp = stk[ok], ww[ok], ssp[ok]
        n = int(len(stk))
        if n == 0:
            rows.append({"edge_min": emin, "n_bets": 0, "roi": float("nan")})
            continue
        win = ww == 1.0
        pnl = np.where(win, stk * (ssp - 1.0), -stk)
        tot_s, tot_p = float(stk.sum()), float(pnl.sum())
        roi = tot_p / tot_s if tot_s > 0 else float("nan")
        rows.append({"edge_min": emin, "n_bets": n, "roi": roi})

    return pd.DataFrame(rows)


def run():
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    for c in ("p_win_race", "sp_decimal"):
        if c not in df.columns:
            raise ValueError("need column: " + c)

    fair, edge, kf = kelly_arrays(df["p_win_race"].values, df["sp_decimal"].values)
    df = df.copy()
    df["fair_price"] = fair
    df["edge"] = edge
    df["kelly_full"] = kf

    n0 = len(df)
    if TOP_PICK_ONLY:
        df = df.loc[top_pick_only(df)].reset_index(drop=True)
        print("top picks only:", len(df), "/", n0, "rows")
    else:
        print("all runners:", len(df), "rows")

    if len(df) == 0:
        return df

    if WRITE_FILES:
        for p in (OUT_CSV, MATRIX_CSV):
            out_dir = os.path.dirname(p)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

    if WRITE_FILES:
        df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        print("saved", OUT_CSV)

    tbl = edge_roi_table(df, EDGE_MINS, BANKROLL)
    print("\nmin edge | n_bets | ROI (decimal, pnl/total stake)\n")
    print(tbl.to_string(index=False))
    if WRITE_FILES:
        tbl.to_csv(MATRIX_CSV, index=False, encoding="utf-8-sig")
        print("\nsaved", MATRIX_CSV)
    return df


if __name__ == "__main__":
    run()
