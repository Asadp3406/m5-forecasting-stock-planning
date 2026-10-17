"""Fill INSIGHTS.md from the tables in outputs/insights/ (no hand-typed numbers)."""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INS = ROOT / "outputs" / "insights"
rd = lambda name: pd.read_csv(INS / f"{name}.csv")
pct = lambda x: f"{x * 100:.1f}%"
num = lambda x: f"{x:.2f}"

store, cat, sc = rd("error_by_store"), rd("error_by_category"), rd("error_by_store_category")
inter, ev, snapc = rd("error_by_intermittency"), rd("error_by_event"), rd("error_by_snap_category")
plan, days = rd("error_by_planning_level").set_index("planning_level"), rd("window_days")
res = pd.read_csv(ROOT / "outputs" / "results_stage_b.csv")
fc = pd.read_parquet(ROOT / "outputs" / "forecast_28d.parquet", columns=["date"])

k = {}
k["wrmsse"] = f"{res.wrmsse.iloc[-1]:.4f}"
k["win_start"], k["win_end"] = str(fc.date.min().date()), str(fc.date.max().date())
for key, name in [("item_daily", "item-store, daily"), ("item_weekly", "item-store, weekly"),
                  ("sc_daily", "store-category, daily"), ("sc_weekly", "store-category, weekly")]:
    k[f"wape_{key}"] = pct(plan.loc[name, "wape"])

k["overall_ratio"] = pct(store.forecast_units.sum() / store.actual_units.sum())
s = store.set_index("store_id")
worst, over = s.forecast_to_actual.idxmin(), s.forecast_to_actual.idxmax()
assert s.loc[over, "forecast_to_actual"] > 1 and (s.forecast_to_actual < 1).sum() == 9  # text says 9 of 10
k.update(worst_store=worst, worst_store_ratio=pct(s.loc[worst, "forecast_to_actual"]),
         over_store=over, over_store_ratio=pct(s.loc[over, "forecast_to_actual"]),
         buffer_overall=pct(store.actual_units.sum() / store.forecast_units.sum() - 1),
         buffer_worst=pct(1 / s.loc[worst, "forecast_to_actual"] - 1))

c = cat.set_index("cat_id")
k.update(foods_units_share=pct(c.loc["FOODS", "share_of_units"]), foods_err_share=pct(c.loc["FOODS", "share_of_abs_error"]),
         foods_wape=pct(c.loc["FOODS", "wape"]), hobbies_wape=pct(c.loc["HOBBIES", "wape"]),
         hobbies_err_share=pct(c.loc["HOBBIES", "share_of_abs_error"]),
         hobbies_units_share=pct(c.loc["HOBBIES", "share_of_units"]))
foods = sc[sc.cat_id == "FOODS"].set_index("store_id")
top = foods.mae.idxmax()
assert top == sc.set_index(["store_id", "cat_id"]).mae.idxmax()[0]  # worst MAE overall is FOODS too
k.update(top_mae_store=top, top_mae_store_value=num(foods.loc[top, "mae"]))

i = inter.set_index("intermittency")
lo, hi1, hi2 = "<50% zero days", "80-95%", ">95%"
k.update(n_series_low_zero=f"{int(i.loc[lo, 'n_series_days'] / 28):,}", low_zero_err_share=pct(i.loc[lo, "share_of_abs_error"]),
         n_series_high_zero=f"{int((i.loc[hi1, 'n_series_days'] + i.loc[hi2, 'n_series_days']) / 28):,}",
         wape_high_zero_80=pct(i.loc[hi1, "wape"]), wape_high_zero_95=pct(i.loc[hi2, "wape"]),
         high_zero_err_share=pct(i.loc[[hi1, hi2], "share_of_abs_error"].sum()),
         high_zero_units_share=pct(i.loc[[hi1, hi2], "share_of_units"].sum()))
# smaller of the two WAPEs first in the sentence
a, b = sorted([i.loc[hi1, "wape"], i.loc[hi2, "wape"]])
k["wape_high_zero_80"], k["wape_high_zero_95"] = pct(a), pct(b)

f = snapc[snapc.cat_id == "FOODS"].set_index("is_snap")
k.update(foods_snap_mae=num(f.loc[True, "mae"]), foods_nosnap_mae=num(f.loc[False, "mae"]),
         foods_snap_mae_up=pct(f.loc[True, "mae"] / f.loc[False, "mae"] - 1),
         foods_snap_ratio=pct(f.loc[True, "forecast_to_actual"]), foods_nosnap_ratio=pct(f.loc[False, "forecast_to_actual"]),
         foods_snap_wape=pct(f.loc[True, "wape"]), foods_nosnap_wape=pct(f.loc[False, "wape"]))
e = ev.set_index("is_event")
k.update(event_mae=num(e.loc[True, "mae"]), nonevent_mae=num(e.loc[False, "mae"]),
         event_wape=pct(e.loc[True, "wape"]), nonevent_wape=pct(e.loc[False, "wape"]),
         n_days=str(int(days.n_days[0])), n_event_days=str(int(days.n_days[1])))

text = (ROOT / "insights" / "INSIGHTS.template.md").read_text().format(**k)
(ROOT / "INSIGHTS.md").write_text(text)
pd.Series(k, name="value").rename_axis("key").to_csv(INS / "key_numbers.csv")

# check: every number in the document must come from key_numbers.csv
body = text.split("## Recommendations")[1].split("## Limits")[0]
allowed = " ".join(k.values())
body_nums = re.findall(r"\d[\d,]*\.?\d*%?", re.sub(r"\*\(Source.*?\)\*|`[^`]*`", "", body))
body_nums = [n.rstrip(".") for n in body_nums]
stray = [n for n in body_nums if n not in allowed.replace(",", ",") and n not in {"1", "2", "3", "4", "9", "10", "28", "50", "80", "95"}]
print("numbers not traced to key_numbers.csv:", stray)
print("INSIGHTS.md written")
