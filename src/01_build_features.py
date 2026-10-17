import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path("/Users/asadkhan.pathan/Documents/New project/m5 forecasting ")
PROCESSED_DIR = BASE_DIR / "data" / "processed"

df = pd.read_parquet(PROCESSED_DIR / "sales_long.parquet")
df = df.sort_values(["id", "date"]).reset_index(drop=True)

# --- Step 1: calendar features (Stage 3) ---------------------------------
# wday: 1=Saturday, 2=Sunday, ..., 7=Friday (M5's calendar convention).
df["is_weekend"] = df["wday"].isin([1, 2]).astype("int8")

df["snap_active"] = np.select(
    [df["state_id"] == "CA", df["state_id"] == "TX", df["state_id"] == "WI"],
    [df["snap_CA"], df["snap_TX"], df["snap_WI"]],
).astype("int8")

df["has_event"] = (df["event_name_1"].notna() | df["event_name_2"].notna()).astype("int8")

print("calendar features done")

# --- Step 2: lag features (28+ days, valid for the full 28-day horizon) --
LAG_DAYS = [28, 35, 42, 49, 56]
for lag in LAG_DAYS:
    df[f"lag_{lag}"] = df.groupby("id", observed=True)["sales"].shift(lag)

print("lag features done")

# --- Step 3: rolling window features (built on lag_28, not raw sales) ----
df["sales_lag28"] = df.groupby("id", observed=True)["sales"].shift(28)
for window in [7, 28]:
    df[f"rolling_mean_{window}_lag28"] = (
        df.groupby("id", observed=True)["sales_lag28"]
        .rolling(window)
        .mean()
        .reset_index(level=0, drop=True)
    )

print("rolling features done")

# --- Step 4: price features -----------------------------------------------
df["price_change_vs_last_week"] = df["sell_price"] - df.groupby("id", observed=True)["sell_price"].shift(7)

expanding_avg_price = (
    df.groupby("id", observed=True)["sell_price"]
    .apply(lambda s: s.shift(1).expanding().mean())
    .reset_index(level=0, drop=True)
)
df["price_rel_to_own_avg"] = df["sell_price"] / expanding_avg_price

print("price features done")

# --- Step 5: categorical encoding (label encoding via .cat.codes) --------
cat_features = [
    "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "event_name_1", "event_type_1", "event_name_2", "event_type_2",
]
for col in cat_features:
    df[f"{col}_enc"] = df[col].cat.codes

print("categorical encoding done")

# --- Save -------------------------------------------------------------
out_path = PROCESSED_DIR / "features.parquet"
df.to_parquet(out_path, index=False)
print(f"saved to {out_path}")
print(f"file size on disk: {out_path.stat().st_size / 1e6:.1f} MB")
print(df.shape)
