"""Fill README.md from the saved outputs (no hand-typed numbers)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
k = pd.read_csv(OUT / "insights" / "key_numbers.csv").set_index("key")["value"]
res = pd.read_csv(OUT / "results_stage_b.csv")
log = pd.read_csv(OUT / "tuning_log.csv").drop_duplicates("trial")
n_series = len(pd.read_parquet(OUT / "series_errors.parquet", columns=["id"]))

both = res.dropna()
gaps = both["wrmsse"] - both["wrmsse_official_style"]
base = log.set_index("trial").loc["baseline", "wrmsse"]
tuned = res["wrmsse"].iloc[-1]


def md_table(df):
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


r = res.copy()
r["wrmsse"] = r["wrmsse"].map("{:.4f}".format)
r["wrmsse_official_style"] = r["wrmsse_official_style"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.4f}")
r.columns = ["Model", "WRMSSE (project metric)", "WRMSSE (official-style)"]
r.loc[r.index[-1]] = [f"**{x}**" for x in r.iloc[-1]]

log["best_iter"] = log["best_iter"].astype(int)
t = log[["trial", "changes", "best_iter", "rmse", "wrmsse", "wrmsse_official_style"]].copy()
t["rmse"], t["wrmsse"], t["wrmsse_official_style"] = (t["rmse"].map("{:.4f}".format),
    t["wrmsse"].map("{:.4f}".format), t["wrmsse_official_style"].map("{:.4f}".format))
t["changes"] = t["changes"].map(lambda s: f"`{s}`")
t.columns = ["Trial", "Change vs baseline", "Rounds", "RMSE", "WRMSSE", "WRMSSE (official-style)"]

findings = f"""1. **Plan stock at store-category-week level.** Item-store-day forecasts miss {k.wape_item_daily} of units; at store-category-week level the miss is {k.wape_sc_weekly}.
2. **The model under-forecasts.** Forecast units are {k.overall_ratio} of actual overall, and {k.worst_store_ratio} at {k.worst_store} (the worst store), so add a small buffer.
3. **FOODS carries the error, slow items are cheap to get wrong.** FOODS is {k.foods_units_share} of units but {k.foods_err_share} of the error. Items that are zero on over 80% of days are {k.high_zero_err_share} of the error.
4. **SNAP days need a modest FOODS buffer.** FOODS misses {k.foods_snap_mae} units per item-day on SNAP days versus {k.foods_nosnap_mae} otherwise. Event days show no clear extra risk in this window."""

text = (ROOT / "insights" / "README.template.md").read_text().format(
    n_series=f"{n_series:,}", findings=findings, win_start=k.win_start, win_end=k.win_end,
    results_table=md_table(r), tuning_table=md_table(t), gain=f"{(base - tuned) / base:.1%}",
    base=f"{base:.4f}", tuned=f"{tuned:.4f}", n_trials=len(log[~log.trial.isin(["baseline", "combo"])]),
    gap_min=f"{gaps.min():.2f}", gap_max=f"{gaps.max():.2f}", n_event_days=k.n_event_days)
(ROOT / "README.md").write_text(text)
print("README.md written")
