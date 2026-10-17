"""Add the tuned Tweedie forecasts (Stage B) to the predictions table.

Run from project root:  python sql/load_tuned_predictions.py
Uses PG* environment variables. Refuses to run twice.
"""
from pathlib import Path

import pandas as pd
import psycopg

ROOT = Path(__file__).resolve().parent.parent
MODEL = "lgbm_tweedie_tuned"

with psycopg.connect() as conn, conn.cursor() as cur:
    cur.execute("SELECT count(*) FROM predictions WHERE model = %s", (MODEL,))
    if cur.fetchone()[0] > 0:
        raise SystemExit(f"{MODEL} already loaded; nothing to do.")

    fc = pd.read_parquet(ROOT / "outputs" / "forecast_28d.parquet")
    fc["id"] = fc["id"].astype(str)
    cur.execute("SELECT id, series_id FROM series")
    series_key = dict(cur.fetchall())
    cur.execute("SELECT date, d_num FROM calendar")
    d_key = {str(d): n for d, n in cur.fetchall()}

    out = pd.DataFrame({
        "series_id": fc["id"].map(series_key),
        "d_num": fc["date"].dt.strftime("%Y-%m-%d").map(d_key),
        "model": MODEL,
        "y_pred": fc["pred"],
    })
    assert out.notna().all().all()  # every row found its series and day
    with cur.copy("COPY predictions (series_id, d_num, model, y_pred) FROM STDIN") as cp:
        for row in out.itertuples(index=False):
            cp.write_row(row)
    conn.commit()
    print(f"loaded {len(out):,} rows for {MODEL}")
