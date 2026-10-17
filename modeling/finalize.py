"""Pick the best tuned model, save final forecasts + per-series errors, run checks.

Reads outputs/tuning_log.csv. Keeps a tuned model only if WRMSSE beats the baseline.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wrmsse import ROOT, TRAIN_END, WRMSSE, load_history, preds_to_matrix

OUT = ROOT / "outputs"
log = pd.read_csv(OUT / "tuning_log.csv").drop_duplicates("trial").set_index("trial")
base = log.loc["baseline", "wrmsse"]
best = log["wrmsse"].idxmin()
final = best if log.loc[best, "wrmsse"] < base else "baseline"
print(f"baseline WRMSSE {base:.4f}; best trial {best} {log.loc[best, 'wrmsse']:.4f}; keeping '{final}'")

# final 28-day forecasts (validation window d_1914..d_1941)
fc = pd.read_parquet(OUT / "trials" / f"{final}.parquet")
fc["id"] = fc["id"].astype(str)
fc.to_parquet(OUT / "forecast_28d.parquet", index=False)

meta, S, P = load_history()
actual = S[:, TRAIN_END:TRAIN_END + 28]
train = S[:, :TRAIN_END]
mat = preds_to_matrix(meta, fc)

# per-series errors (RMSSE uses the same scale definition as the project WRMSSE)
scale = np.mean(np.diff(train, axis=1).astype(float) ** 2, axis=1)
err = meta.copy()
err["rmsse"] = np.sqrt(np.mean((actual - mat) ** 2, axis=1) / scale)
err["mae"] = np.mean(np.abs(actual - mat), axis=1)
err["bias"] = np.mean(mat - actual, axis=1)
err["actual_units"] = actual.sum(axis=1)
err["forecast_units"] = mat.sum(axis=1)
err["zero_share_train"] = (train == 0).mean(axis=1)
err["zero_share_valid"] = (actual == 0).mean(axis=1)
err.to_parquet(OUT / "series_errors.parquet", index=False)

# checks
res = WRMSSE(meta, S, P, skip_leading_zeros=False).score(mat)["WRMSSE"]
print(f"check 2: re-scored saved forecast WRMSSE = {res:.4f} (log says {log.loc[final, 'wrmsse']:.4f})")
valid_start = pd.Timestamp("2016-04-25")
print("check 3: forecast dates", fc["date"].min().date(), "->", fc["date"].max().date(),
      "| all after training cutoff:", bool((fc["date"] >= valid_start).all()),
      "| rows:", len(fc), "| series:", fc["id"].nunique())
sn = pd.read_csv(OUT / "seasonal_naive_wrmsse.csv").iloc[0]
rows = [("seasonal_naive", sn.wrmsse, sn.wrmsse_official_style),
        ("lgbm_regression (Stage 6)", 0.7174, np.nan),
        ("lgbm_tweedie baseline", base, log.loc["baseline", "wrmsse_official_style"]),
        (f"lgbm_tweedie tuned ({final})", log.loc[final, "wrmsse"], log.loc[final, "wrmsse_official_style"])]
pd.DataFrame(rows, columns=["model", "wrmsse", "wrmsse_official_style"]).to_csv(
    OUT / "results_stage_b.csv", index=False)
print(pd.read_csv(OUT / "results_stage_b.csv").round(4).to_string(index=False))
