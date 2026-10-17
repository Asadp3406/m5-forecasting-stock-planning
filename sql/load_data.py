"""Load calendar, prices, sales and saved predictions into PostgreSQL.

Settings come from environment variables: PGHOST, PGPORT, PGDATABASE,
PGUSER, PGPASSWORD (psycopg reads them automatically).
Run from the project root:  python sql/load_data.py
"""
import io
from pathlib import Path

import pandas as pd
import psycopg
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
PROC = ROOT / "data" / "processed"

LAST_D = 1941          # last day with actual sales
KEEP_DAYS = 730        # load only the last ~2 years of sales
FIRST_D = LAST_D - KEEP_DAYS + 1
CHUNK = 1_000_000      # rows per COPY batch

# prediction files -> model name stored in the table
PRED_FILES = {
    "valid_predictions.parquet": "lgbm_regression",
    "valid_predictions_tweedie.parquet": "lgbm_tweedie",
}


def copy_df(cur, table, df):
    """Bulk-load a DataFrame into a table with COPY, in chunks."""
    cols = ", ".join(df.columns)
    for start in range(0, len(df), CHUNK):
        buf = io.StringIO()
        df.iloc[start:start + CHUNK].to_csv(buf, index=False, header=False)
        buf.seek(0)
        with cur.copy(f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv)") as cp:
            cp.write(buf.read())
    print(f"  {table}: {len(df):,} rows loaded")


def require_empty(cur, table):
    """Stop instead of overwriting data that is already loaded."""
    cur.execute(f"SELECT count(*) FROM {table}")
    if cur.fetchone()[0] > 0:
        raise SystemExit(f"Table {table} already has rows; not loading again.")


def main():
    with psycopg.connect() as conn, conn.cursor() as cur:
        cur.execute((ROOT / "sql" / "00_schema.sql").read_text())
        for t in ["calendar", "series", "sell_prices", "sales", "predictions"]:
            require_empty(cur, t)

        # calendar
        cal = pd.read_csv(RAW / "calendar.csv")
        cal["d_num"] = cal["d"].str[2:].astype(int)
        cal = cal.rename(columns={"snap_CA": "snap_ca", "snap_TX": "snap_tx",
                                  "snap_WI": "snap_wi"})
        cal = cal[["d_num", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
                   "event_name_1", "event_type_1", "event_name_2", "event_type_2",
                   "snap_ca", "snap_tx", "snap_wi"]]
        copy_df(cur, "calendar", cal)
        first_date = cal.loc[cal.d_num == FIRST_D, "date"].iloc[0]

        # sales: only the last KEEP_DAYS days
        cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "d", "sales"]
        sales = pq.read_table(PROC / "sales_long.parquet", columns=cols + ["date"],
                              filters=[("date", ">=", pd.Timestamp(first_date))]).to_pandas()
        for c in sales.select_dtypes("category").columns:
            sales[c] = sales[c].astype(str)
        sales["d_num"] = sales["d"].str[2:].astype(int)

        # series dimension table
        dims = sales[cols[:6]].drop_duplicates("id").sort_values("id").reset_index(drop=True)
        dims.insert(0, "series_id", dims.index)
        copy_df(cur, "series", dims)
        key = dict(zip(dims["id"], dims["series_id"]))

        # prices (all history; the table is small)
        prices = pd.read_csv(RAW / "sell_prices.csv")
        copy_df(cur, "sell_prices", prices)

        sales["series_id"] = sales["id"].map(key)
        copy_df(cur, "sales", sales[["series_id", "d_num", "sales"]])

        # predictions
        dnum_by_date = dict(zip(cal["date"].astype(str), cal["d_num"]))
        for fname, model in PRED_FILES.items():
            p = pd.read_parquet(PROC / fname, columns=["id", "date", "pred"])
            out = pd.DataFrame({
                "series_id": p["id"].astype(str).map(key),
                "d_num": p["date"].dt.strftime("%Y-%m-%d").map(dnum_by_date),
                "model": model,
                "y_pred": p["pred"],
            })
            copy_df(cur, "predictions", out)

        conn.commit()
    print("Done.")


if __name__ == "__main__":
    main()
