"""Forecast API: 28-day sales forecast for one item in one store.

Run from project root:  uvicorn api.main:app --reload
Database settings come from PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD.
"""
from datetime import date
from typing import Literal

import psycopg
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

app = FastAPI(title="M5 Forecast API")

# model names as stored in the predictions table
Model = Literal["lgbm_tweedie_tuned", "lgbm_tweedie", "lgbm_regression"]


class Point(BaseModel):
    date: date
    predicted_units: float


class ForecastResponse(BaseModel):
    store_id: str
    item_id: str
    model: str
    forecast: list[Point]


def run_query(sql, params=()):
    """Open a connection (settings from PG* env vars), run one query, return rows."""
    try:
        with psycopg.connect(connect_timeout=5) as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    except psycopg.OperationalError:
        raise HTTPException(status_code=503, detail="Database not reachable")


@app.get("/health")
def health():
    """Checks that the API can talk to the database."""
    run_query("SELECT 1")
    return {"status": "ok"}


@app.get("/forecast", response_model=ForecastResponse)
def forecast(store_id: str = Query(..., examples=["CA_1"]),
             item_id: str = Query(..., examples=["FOODS_3_090"]),
             model: Model = "lgbm_tweedie_tuned"):
    """Return the 28 daily forecasts (2016-04-25 to 2016-05-22) for one series."""
    rows = run_query("""
        SELECT c.date, p.y_pred
        FROM predictions p
        JOIN series   s USING (series_id)
        JOIN calendar c USING (d_num)
        WHERE s.store_id = %s AND s.item_id = %s AND p.model = %s
        ORDER BY c.date""", (store_id, item_id, model))
    if not rows:
        raise HTTPException(status_code=404,
                            detail=f"No forecast for store_id={store_id}, item_id={item_id}")
    return ForecastResponse(store_id=store_id, item_id=item_id, model=model,
                            forecast=[Point(date=d, predicted_units=round(y, 4)) for d, y in rows])
