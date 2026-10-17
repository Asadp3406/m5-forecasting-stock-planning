"""Fill README.md from the saved outputs (no hand-typed numbers)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
k = pd.read_csv(OUT / "insights" / "key_numbers.csv").set_index("key")["value"]
res = pd.read_csv(OUT / "results_stage_b.csv")
log = pd.read_csv(OUT / "tuning_log.csv").drop_duplicates("trial")
n_series = len(pd.read_parquet(OUT / "series_errors.parquet", columns=["id"]))

base = log.set_index("trial").loc["baseline", "wrmsse"]
tuned = res["wrmsse"].iloc[-1]


def md_table(df):
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


r = res[["model", "wrmsse"]].copy()
r["model"] = ["Seasonal naive (repeat last week)", "LightGBM, regression loss",
              "LightGBM, Tweedie loss", "LightGBM, Tweedie loss, tuned"]
r["wrmsse"] = r["wrmsse"].map("{:.4f}".format)
r.columns = ["Model", "WRMSSE"]
r.loc[r.index[-1]] = [f"**{x}**" for x in r.iloc[-1]]

t = log[["trial", "changes", "best_iter", "rmse", "wrmsse"]].copy()
t["rmse"], t["wrmsse"] = t["rmse"].map("{:.4f}".format), t["wrmsse"].map("{:.4f}".format)
t["changes"] = t["changes"].map(lambda s: f"`{s}`")
t.columns = ["Trial", "Change vs baseline", "Rounds", "RMSE", "WRMSSE"]

findings = f"""- **Plan by store, category and week, not item by day.** At item-store-day level the forecast misses {k.wape_item_daily} of units sold. Add the items up to store-category-week and the miss drops to {k.wape_sc_weekly}.
- **The model runs low.** It forecasts {k.overall_ratio} of what actually sold, and only {k.worst_store_ratio} at {k.worst_store}, the worst store. A small safety buffer makes sense.
- **FOODS carries most of the error, slow items hardly any.** FOODS is {k.foods_units_share} of units but {k.foods_err_share} of the error. Items with zero sales on over 80% of days add just {k.high_zero_err_share}, so simple reorder rules are enough for them.
- **SNAP days need a little extra FOODS stock.** FOODS misses {k.foods_snap_mae} units per item-day on SNAP days against {k.foods_nosnap_mae} on other days. Event days showed no clear extra risk in this window."""

text = (ROOT / "insights" / "README.template.md").read_text().format(
    n_series=f"{n_series:,}", findings=findings, win_start=k.win_start, win_end=k.win_end,
    results_table=md_table(r), tuning_table=md_table(t), gain=f"{(base - tuned) / base:.1%}",
    base=f"{base:.4f}", tuned=f"{tuned:.4f}", n_trials=len(log[~log.trial.isin(["baseline", "combo"])]))
(ROOT / "README.md").write_text(text)
print("README.md written")
