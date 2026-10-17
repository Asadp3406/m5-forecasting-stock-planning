"""Where is the tuned Tweedie model weak? Writes tables to outputs/insights/.

Uses the saved Stage B forecast (d_1914..d_1941) and actual sales. Error measures:
  MAE   = average absolute miss per series-day (units)
  WAPE  = total absolute miss / total actual units (comparable across big and small groups)
  bias  = average (forecast - actual); ratio = forecast units / actual units (<1 = under-forecast)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modeling"))
from wrmsse import ROOT, TRAIN_END, load_history

OUT = ROOT / "outputs"
INS = OUT / "insights"

meta, S, P = load_history()
fc = pd.read_parquet(OUT / "forecast_28d.parquet")
fc["id"] = fc["id"].astype(str)
dates = sorted(fc["date"].unique())
pred = fc.pivot(index="id", columns="date", values="pred").loc[meta["id"]].to_numpy()
actual = S[:, TRAIN_END:TRAIN_END + 28].astype(float)

# long table: one row per series-day
cal = pd.read_csv(ROOT / "raw" / "calendar.csv", parse_dates=["date"]).set_index("date").loc[dates]
n = len(meta)
df = pd.DataFrame({
    "series": np.repeat(np.arange(n), 28),
    "date": np.tile(dates, n),
    "actual": actual.ravel(),
    "pred": pred.ravel(),
})
for c in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
    df[c] = meta[c].to_numpy().repeat(28)
df["err"] = df["pred"] - df["actual"]
df["abs_err"] = df["err"].abs()
cal_d = cal.reset_index()
df = df.merge(cal_d[["date", "event_name_1", "snap_CA", "snap_TX", "snap_WI", "weekday"]], on="date")
df["is_event"] = df["event_name_1"].notna()
df["is_snap"] = np.select([df.state_id == "CA", df.state_id == "TX"],
                          [df.snap_CA == 1, df.snap_TX == 1], df.snap_WI == 1)

# intermittency: share of zero-sales days in the last 365 training days
df["zero_share"] = ((S[:, TRAIN_END - 365:TRAIN_END] == 0).mean(axis=1))[df["series"]]
df["intermittency"] = pd.cut(df["zero_share"], [-0.01, 0.5, 0.8, 0.95, 1.0],
                             labels=["<50% zero days", "50-80%", "80-95%", ">95%"])


def summarise(g):
    """Error summary for a group of series-days."""
    return pd.Series({
        "n_series_days": len(g),
        "actual_units": g["actual"].sum(),
        "forecast_units": g["pred"].sum(),
        "mae": g["abs_err"].mean(),
        "rmse": np.sqrt((g["err"] ** 2).mean()),
        "bias": g["err"].mean(),
        "wape": g["abs_err"].sum() / g["actual"].sum(),
        "forecast_to_actual": g["pred"].sum() / g["actual"].sum(),
        "share_of_abs_error": g["abs_err"].sum() / df["abs_err"].sum(),
        "share_of_units": g["actual"].sum() / df["actual"].sum(),
    })


def table(by, name):
    t = df.groupby(by, observed=True)[["actual", "pred", "err", "abs_err"]].apply(summarise)
    t = t.reset_index()
    t.to_csv(INS / f"{name}.csv", index=False)
    return t


table("store_id", "error_by_store")
table("cat_id", "error_by_category")
table(["store_id", "cat_id"], "error_by_store_category")
table("intermittency", "error_by_intermittency")
table("is_event", "error_by_event")
table(["cat_id", "is_snap"], "error_by_snap_category")
table(["is_snap"], "error_by_snap")
table("weekday", "error_by_weekday")
table(["cat_id", "is_event"], "error_by_event_category")

# number of distinct event / SNAP days in the window (small samples need a warning)
days = cal_d.assign(is_event=cal_d.event_name_1.notna())
pd.DataFrame({"n_days": [len(days), int(days.is_event.sum())],
              "what": ["days in window", "event days in window"]}).to_csv(INS / "window_days.csv", index=False)

# does planning at a coarser level cut the error? daily item-store vs weekly store-category
df["week"] = (pd.to_datetime(df["date"]) - pd.Timestamp(dates[0])).dt.days // 7
levels = {"item-store, daily": ["series", "date"],
          "item-store, weekly": ["series", "week"],
          "store-category, daily": ["store_id", "cat_id", "date"],
          "store-category, weekly": ["store_id", "cat_id", "week"]}
rows = []
for name, cols in levels.items():
    a = df.groupby(cols)[["actual", "pred"]].sum()
    rows.append({"planning_level": name, "wape": (a.pred - a.actual).abs().sum() / a.actual.sum(),
                 "n_units_planned": len(a)})
pd.DataFrame(rows).to_csv(INS / "error_by_planning_level.csv", index=False)

# intermittent items: how many series, and how much of total error they carry
print("done; files in", INS)
