import pandas as pd
from pathlib import Path

# --- Paths -------------------------------------------------------------
# Project root is the parent of this src/ folder. Keeping RAW_DIR and
# PROCESSED_DIR both anchored to BASE_DIR avoids the earlier bug where
# PROCESSED_DIR was relative and would land inside src/ instead of at the
# project root, depending on which directory the script was run from.
BASE_DIR = Path("/Users/asadkhan.pathan/Documents/New project/m5 forecasting ")
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

# --- Step 1: load sales (wide format) -----------------------------------
# Load the *evaluation* file, not validation — it's a superset containing
# 28 extra days of actuals we'll want later for full 28-day-ahead checks.
# id columns are cast to category up front: they repeat constantly across
# rows, so storing them as small integer codes instead of raw strings saves
# a large amount of memory once we melt to 59M+ rows.
sales = pd.read_csv(
    RAW_DIR / "sales_train_evaluation.csv",
    dtype={col: "category" for col in id_cols},
)

print("=== raw sales (wide) ===")
print(sales.shape)
print(sales.dtypes.head(8))
print(f"{sales.memory_usage(deep=True).sum() / 1e6:.1f} MB\n")

# --- Step 2: downcast day columns ---------------------------------------
# Day columns (d_1 ... d_1941) load as int64 by default (8 bytes/value),
# but daily sales are small non-negative counts. We check the actual max
# before downcasting (rather than assuming) to avoid silent overflow.
day_cols = [c for c in sales.columns if c.startswith("d_")]

max_val = sales[day_cols].max().max()
print(f"max daily sales value: {max_val}")
assert max_val < 32767, "value too large for int16 — do not downcast blindly"

sales[day_cols] = sales[day_cols].astype("int16")
print(f"after downcast: {sales.memory_usage(deep=True).sum() / 1e6:.1f} MB\n")

# --- Step 3: melt wide -> long -------------------------------------------
# Reshape so each row is one (series, day) observation instead of one row
# per series with 1941 day columns. This is what makes groupby/lag/rolling
# operations in later stages straightforward.
sales_long = sales.melt(
    id_vars=id_cols,
    value_vars=day_cols,
    var_name="d",
    value_name="sales",
)
# 'd' only has 1941 unique values (d_1..d_1941) repeated 59M times -> category.
sales_long["d"] = sales_long["d"].astype("category")

print("=== after melt ===")
print(sales_long.shape)  # expect (30490 * 1941, 8) = (59181090, 8)
print(f"{sales_long.memory_usage(deep=True).sum() / 1e6:.1f} MB\n")

# --- Step 4: merge in calendar (adds real dates, events, SNAP flags) ----
# calendar.csv has one row per 'd', so this merge is a simple lookup and
# must NOT change the row count of sales_long — we assert that explicitly
# rather than assuming the merge behaved.
calendar = pd.read_csv(
    RAW_DIR / "calendar.csv",
    dtype={
        "event_name_1": "category",
        "event_type_1": "category",
        "event_name_2": "category",
        "event_type_2": "category",
        "snap_CA": "int8",
        "snap_TX": "int8",
        "snap_WI": "int8",
    },
    parse_dates=["date"],
)

before_rows = len(sales_long)
sales_long = sales_long.merge(calendar, on="d", how="left")
assert len(sales_long) == before_rows, "row count changed after calendar merge!"

print("=== after calendar merge ===")
print(sales_long.shape)
null_counts = sales_long.isna().sum()
print(null_counts[null_counts > 0], "\n")

# --- Step 5: merge in sell_prices (weekly price per store/item) ---------
# Prices are keyed by (store_id, item_id, wm_yr_wk) — weekly, not daily —
# which is why we needed wm_yr_wk from the calendar merge first. Some
# (store, item, week) combos won't have a price because that item wasn't
# yet sold in that store; those become NaN, which is meaningful (not a bug)
# and must NOT be silently dropped here.
prices = pd.read_csv(
    RAW_DIR / "sell_prices.csv",
    dtype={
        "store_id": "category",
        "item_id": "category",
    },
)
prices["sell_price"] = prices["sell_price"].astype("float32")

before_rows = len(sales_long)
sales_long = sales_long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
assert len(sales_long) == before_rows, "row count changed after price merge!"

n_missing_price = sales_long["sell_price"].isna().sum()
pct_missing_price = n_missing_price / len(sales_long) * 100
print("=== after price merge (before dtype fix) ===")
print(f"missing sell_price: {n_missing_price} rows ({pct_missing_price:.1f}%)")
print(sales_long.shape)
print(f"{sales_long.memory_usage(deep=True).sum() / 1e6:.1f} MB\n")

# --- Step 6: fix categorical dtype blowup from the merges ---------------
# Merging two categorical columns that come from different source files
# (sales_long's store_id/item_id vs prices' store_id/item_id) have
# different underlying category sets, so pandas silently falls back to
# plain object/string dtype for the join keys. That throws away the
# memory savings category dtype gave us. Fix: re-cast explicitly afterward.
cat_cols = [
    "id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "d",
    "event_name_1", "event_type_1", "event_name_2", "event_type_2",
    "weekday",  # only 7 unique values (Monday..Sunday) but loaded as a
                # full string column from calendar.csv — same fix as the id cols.
]
for col in cat_cols:
    sales_long[col] = sales_long[col].astype("category")

# wm_yr_wk, wday, month, year all arrived from the calendar merge as int64
# by default. They're all small integers, so we downcast each to the
# smallest type that safely holds its range:
#   wm_yr_wk ~ 11101-11621  -> int32 (int16 would also fit, int32 for headroom)
#   wday     1-7            -> int8
#   month    1-12           -> int8
#   year     2011-2016      -> int16 (too large for int8, whose max is 127)
sales_long["wm_yr_wk"] = sales_long["wm_yr_wk"].astype("int32")
sales_long["wday"] = sales_long["wday"].astype("int8")
sales_long["month"] = sales_long["month"].astype("int8")
sales_long["year"] = sales_long["year"].astype("int16")

print("=== after dtype fix ===")
print(sales_long.dtypes)
total_mb = sales_long.memory_usage(deep=True).sum() / 1e6
print(f"total memory: {total_mb:.1f} MB\n")

# Per-column breakdown so we can SEE where memory goes instead of guessing.
# This is the same evidence-first habit as checking max_val before
# downcasting sales — don't assume a fix worked, look at the numbers.
mem_by_col = sales_long.memory_usage(deep=True).sort_values(ascending=False) / 1e6
print("memory per column (MB), descending:")
print(mem_by_col)
print()

# --- Step 7: save ---------------------------------------------------------
# Parquet preserves dtypes on reload (category stays category, int16 stays
# int16) so later stages don't have to redo all this downcasting work.
out_path = PROCESSED_DIR / "sales_long.parquet"
sales_long.to_parquet(out_path, index=False)

print(f"saved to {out_path}")
print(f"file size on disk: {out_path.stat().st_size / 1e6:.1f} MB")
