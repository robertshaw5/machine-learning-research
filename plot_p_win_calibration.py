# Scatter: empirical win rate vs model probability in equal-width buckets.

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- edit these ---
INPUT_CSV = "outputs/test_predictions.csv"
OUTPUT_PNG = "outputs/p_win_calibration_scatter.png"
PROB_COL = "p_win"  # or "p_win_raw" for uncalibrated marginals from train_lgbm
BUCKET_WIDTH = 0.01  
MIN_N = 1  # minimum rows per bucket to plot


def main() -> None:
    w = float(BUCKET_WIDTH)
    if w <= 0 or w > 1:
        raise SystemExit("BUCKET_WIDTH must be in (0, 1]")
    n_bins = int(round(1.0 / w))
    if n_bins < 1 or not math.isclose(n_bins * w, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise SystemExit("BUCKET_WIDTH must divide 1.0 evenly (e.g. 0.05, 0.1, 0.01).")

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    if PROB_COL not in df.columns or "target_winner" not in df.columns:
        raise SystemExit(f"CSV must include columns {PROB_COL!r} and target_winner")

    pv = df[PROB_COL].to_numpy(dtype=float)
    y = df["target_winner"].to_numpy(dtype=float)

    k = np.clip(np.floor(pv / w).astype(np.int64), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        m = k == b
        n = int(m.sum())
        if n < MIN_N:
            continue
        mid = (b + 0.5) * w
        win_rate = float(np.mean(y[m]))
        rows.append({"bucket": b, "x_mid": mid, "n": n, "win_rate": win_rate})

    cal = pd.DataFrame(rows)
    if cal.empty:
        raise SystemExit("No buckets with enough rows; lower MIN_N or check data.")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        cal["x_mid"],
        cal["win_rate"],
        s=np.clip(cal["n"] / cal["n"].max() * 120, 8, 120),
        alpha=0.75,
        edgecolors="k",
        linewidths=0.3,
    )
    lo, hi = 0.0, 1.0
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="perfect calibration")
    ax.set_xlabel(f"Model probability (bin centre, {w:g}-wide buckets)")
    ax.set_ylabel("Actual win rate")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_title(f"Calibration: win rate vs predicted probability ({w:g} buckets)")
    fig.tight_layout()
    out_dir = os.path.dirname(OUTPUT_PNG)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUTPUT_PNG} ({len(cal)} buckets plotted)")


if __name__ == "__main__":
    main()
