"""Compare row counts in PostgreSQL with the source files. Uses PG* env vars."""
from pathlib import Path

import pandas as pd
import psycopg
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
KEEP_DAYS = 730

# expected row counts, computed from the source files
sales_meta = pq.ParquetFile(PROC / "sales_long.parquet").metadata.num_rows
n_series = 30_490
expected = {
    "calendar": len(pd.read_csv(ROOT / "raw" / "calendar.csv")),
    "sell_prices": len(pd.read_csv(ROOT / "raw" / "sell_prices.csv")),
    "series": n_series,
    "sales": n_series * KEEP_DAYS,  # last 730 days only (full file has 59,181,090 rows)
    "predictions": 2 * pq.ParquetFile(PROC / "valid_predictions.parquet").metadata.num_rows,
}
assert sales_meta == 59_181_090

ok = True
with psycopg.connect() as conn, conn.cursor() as cur:
    for table, exp in expected.items():
        cur.execute(f"SELECT count(*) FROM {table}")
        got = cur.fetchone()[0]
        ok &= got == exp
        print(f"{table:12s} expected={exp:>11,} db={got:>11,} {'OK' if got == exp else 'MISMATCH'}")
    # sales in the DB must equal the source sales for the same days
    cur.execute("SELECT sum(sales) FROM sales")
    db_sum = cur.fetchone()[0]
    cur.execute("SELECT min(date) FROM sales JOIN calendar USING (d_num)")
    first_date = pd.Timestamp(cur.fetchone()[0])
src = pq.read_table(PROC / "sales_long.parquet", columns=["sales", "date"],
                    filters=[("date", ">=", first_date)]).to_pandas()
src_sum = int(src["sales"].sum())
ok &= db_sum == src_sum
print(f"sum(sales)   expected={src_sum:>11,} db={db_sum:>11,} {'OK' if db_sum == src_sum else 'MISMATCH'}")
print("ALL CHECKS PASSED" if ok else "CHECKS FAILED")
