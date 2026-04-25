# Goes through test predictions in date order, staking Kelly fractions of bankroll
import os

import matplotlib

matplotlib.use("Agg")  # write png without a display window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from find_profitable_strategy import kelly_arrays, top_pick_only

# --- edit these ---
INPUT_CSV = "outputs/test_predictions.csv"
START_BANKROLL = 1000.0

# max fraction of bankroll per bet for the graph.
KELLY_CAP = 0.1

# only place bets with edge above this.
EDGE_MIN = 0.6

TOP_PICK_ONLY = True  # one dog per race or all dogs that pass the filters

# Fraction of kelly for graph purposes.
STAKE_MULT = 0.25

# if True, stop when bankroll hits 0; if False, can go negative (weird but allowed)
STOP_IF_BUST = True

PLOT_FILE = "outputs/bankroll_sim.png"  # bankroll after each bet; set to None to skip

# compound return matrix: rows = fraction of full Kelly, cols = Kelly cap (each cell = full sim)
MATRIX_STAKE_MULTS = (0.125, 0.25, 0.5, 0.75, 1.0)
MATRIX_KELLY_CAPS = (0.01, 0.025, 0.05, 0.1, 0.2, 0.3)
MATRIX_CSV = "outputs/simulate_roi_matrix.csv"  # set None to not write file
RUN_MATRIX = True


def _meeting_dates(s):
    raw = s.astype(str).str.strip()
    out = pd.to_datetime(raw, format="%d-%b-%y", errors="coerce")
    bad = out.isna() & raw.notna() & ~raw.str.lower().isin(("", "nan", "<na>", "none"))
    if bad.any():
        out = out.copy()
        out.loc[bad] = pd.to_datetime(raw.loc[bad], errors="coerce", dayfirst=True)
    return out


def load_filtered_bets(path):
    """Same bets for every sim: sorted by date; has kelly_full and edge, no stake_frac yet."""
    df = pd.read_csv(path, low_memory=False)
    for c in ("p_win_race", "sp_decimal", "target_winner", "meeting_date"):
        if c not in df.columns:
            raise ValueError("missing column: " + c)

    _, edge, kf = kelly_arrays(
        df["p_win_race"].to_numpy(),
        df["sp_decimal"].to_numpy(),
    )
    df = df.copy()
    df["edge"] = edge
    df["kelly_full"] = kf

    m = (
        df["kelly_full"].notna()
        & (df["kelly_full"] > 0)
        & df["edge"].notna()
        & (df["edge"] >= EDGE_MIN)
    )
    if TOP_PICK_ONLY:
        m &= top_pick_only(df)
    df = df.loc[m].copy()

    df["_dt"] = _meeting_dates(df["meeting_date"])
    if "race_id" not in df.columns:
        df["race_id"] = ""
    if "greyhound" not in df.columns:
        df["greyhound"] = ""
    return df.sort_values(["_dt", "race_id", "greyhound"], kind="mergesort", na_position="last").reset_index(
        drop=True
    )


def with_stake_frac(df, stake_mult, kelly_cap):
    out = df.copy()
    out["stake_frac"] = np.minimum(
        float(stake_mult) * out["kelly_full"].to_numpy(dtype=float),
        float(kelly_cap),
    )
    return out


def simulate(df):
    out_rows = []
    b = float(START_BANKROLL)

    for i in range(len(df)):
        row = df.iloc[i]
        f = float(row["stake_frac"])
        sp = float(row["sp_decimal"])
        win = float(row["target_winner"]) > 0.5
        if not np.isfinite(f) or f <= 0 or not np.isfinite(sp) or not np.isfinite(b):
            continue
        if STOP_IF_BUST and b <= 0:
            break
        amt = f * b
        if not np.isfinite(amt) or amt <= 0:
            continue
        pnl = amt * (sp - 1.0) if win else -amt
        b2 = b + pnl
        if STOP_IF_BUST and b2 < 0:
            b2 = 0.0
        out_rows.append(
            {
                "bankroll_before": b,
                "stake": amt,
                "pnl": pnl,
                "bankroll_after": b2,
                "meeting_date": row["meeting_date"],
            }
        )
        b = b2

    return b, pd.DataFrame(out_rows)


def compound_return_decimal(final_bankroll):
    if START_BANKROLL <= 0:
        return float("nan")
    return final_bankroll / START_BANKROLL - 1.0


def roi_matrix_compound(base_df):
    """Rows: STAKE_MULT, cols: KELLY_CAP — values are compound return (decimal), same bet order each time."""
    rows_out = []
    idx = [str(x) for x in MATRIX_STAKE_MULTS]
    cols = [str(c) for c in MATRIX_KELLY_CAPS]
    for sm in MATRIX_STAKE_MULTS:
        row_vals = []
        for cap in MATRIX_KELLY_CAPS:
            d = with_stake_frac(base_df, sm, cap)
            final_b, _ = simulate(d)
            row_vals.append(compound_return_decimal(final_b))
        rows_out.append(row_vals)
    return pd.DataFrame(rows_out, index=idx, columns=cols)


def save_bankroll_plot(ledger):
    if ledger.empty or not PLOT_FILE:
        return
    out_dir = os.path.dirname(PLOT_FILE)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    y = ledger["bankroll_after"].to_numpy()
    x = np.arange(1, len(y) + 1)
    plt.figure(figsize=(9, 4))
    plt.plot(x, y, color="C0", linewidth=1.0)
    plt.axhline(START_BANKROLL, color="gray", linestyle="--", linewidth=0.9, label="start bankroll")
    plt.xlabel("bet number (in time order)")
    plt.ylabel("bankroll ($)")
    plt.title("Bankroll through the simulation")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    plt.close()
    print("saved plot", PLOT_FILE)


def run():
    base = load_filtered_bets(INPUT_CSV)
    if len(base) == 0:
        print("no rows left after filters")
        return

    scope = "top pick" if TOP_PICK_ONLY else "all filtered rows"
    print(f"{scope} | edge >= {EDGE_MIN} | {len(base)} bets in order\n")

    if RUN_MATRIX:
        mat = roi_matrix_compound(base)
        print("compound ROI (decimal), rows = STAKE_MULT, cols = KELLY_CAP")
        print("(e.g. 0.05 = +5% total return over the run)\n")
        print(mat.round(4).to_string())
        if MATRIX_CSV:
            out_dir = os.path.dirname(MATRIX_CSV)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            mat.to_csv(MATRIX_CSV, encoding="utf-8-sig")
            print("\nsaved", MATRIX_CSV)
        print()

    df = with_stake_frac(base, STAKE_MULT, KELLY_CAP)
    final_b, ledger = simulate(df)
    print(f"detail run: stake = min({STAKE_MULT}*kelly_full, {KELLY_CAP})")
    print(f"bets: {len(ledger)} (from {len(base)} rows)")
    print(f"start ${START_BANKROLL:.2f}  ->  end ${final_b:.2f}")
    if len(ledger) and START_BANKROLL > 0:
        print(f"return: {(final_b / START_BANKROLL - 1.0) * 100:.2f} %")

    if not ledger.empty:
        eod = (
            ledger.assign(_d=_meeting_dates(ledger["meeting_date"]))
            .groupby("_d", sort=True)
            .last()
            .reset_index()
        )
        print(f"days with bets: {len(eod)}, last day end bankroll: {eod['bankroll_after'].iloc[-1]:.2f}")
        save_bankroll_plot(ledger)


if __name__ == "__main__":
    run()
