# M5 Forecasting: from raw sales to stock-planning decisions

I wanted to know where a retail demand forecast can be trusted and where it can't, and what that means for how much stock to order. This project takes the M5 Walmart data (30,490 item-store series, daily unit sales from 2011 to 2016) through a SQL layer, a tuned LightGBM model, an error analysis, a dashboard and a small API.

**The question:** which stores, categories and day types (events, SNAP days) are hardest to forecast, and what does that mean for stock planning?

The raw data is not in this repo. You can get it from the [M5 Forecasting - Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) competition page.

## What I found

- **Plan by store, category and week, not item by day.** At item-store-day level the forecast misses 73.5% of units sold. Add the items up to store-category-week and the miss drops to 6.9%.
- **The model runs low.** It forecasts 94.3% of what actually sold, and only 88.5% at CA_2, the worst store. A small safety buffer makes sense.
- **FOODS carries most of the error, slow items hardly any.** FOODS is 67.6% of units but 61.5% of the error. Items with zero sales on over 80% of days add just 9.6%, so simple reorder rules are enough for them.
- **SNAP days need a little extra FOODS stock.** FOODS misses 1.48 units per item-day on SNAP days against 1.33 on other days. Event days showed no clear extra risk in this window.

The full write-up, with sources for each number, is in [INSIGHTS.md](INSIGHTS.md).

## How it fits together

```mermaid
flowchart LR
    A[Raw M5 CSVs<br/>sales, calendar, prices] --> B[Notebooks 01-06<br/>long format, EDA, features,<br/>baselines, LightGBM, WRMSSE]
    B --> C[(PostgreSQL<br/>sales, calendar, prices,<br/>series, predictions)]
    B --> D[modeling/<br/>tune Tweedie LightGBM]
    D -->|forecast_28d.parquet| C
    C --> E[sql/ KPI queries]
    C --> F[Streamlit dashboard<br/>app/dashboard.py]
    C --> G[FastAPI /forecast<br/>api/main.py]
    G --> H[Docker image]
    D --> I[insights/ error analysis<br/>INSIGHTS.md]
```

## Results

I trained on everything up to day 1913 and forecast the next 28 days (2016-04-25 to 2016-05-22). The split is by time, every lag feature is at least 28 days old, and nothing is split at random. The score is WRMSSE, the M5 competition metric, where lower is better.

| Model | WRMSSE |
|---|---|
| Seasonal naive (repeat last week) | 0.9128 |
| LightGBM, regression loss | 0.7174 |
| LightGBM, Tweedie loss | 0.7036 |
| **LightGBM, Tweedie loss, tuned** | **0.6948** |

Tuning the Tweedie model improved WRMSSE from 0.7036 to 0.6948, a gain of 1.3%. I tried 7 single changes and only `num_leaves=63` helped.

<details><summary>All tuning trials</summary>

| Trial | Change vs baseline | Rounds | RMSE | WRMSSE |
|---|---|---|---|---|
| baseline | `{}` | 110.0 | 2.1741 | 0.7036 |
| vp_1.1 | `{'tweedie_variance_power': 1.1}` | 140.0 | 2.1803 | 0.7049 |
| vp_1.3 | `{'tweedie_variance_power': 1.3}` | 188.0 | 2.1776 | 0.7144 |
| lr_0.05 | `{'learning_rate': 0.05}` | 267.0 | 2.1704 | 0.7060 |
| leaves_63 | `{'num_leaves': 63}` | 108.0 | 2.1679 | 0.6948 |
| leaves_127_mdl100 | `{'num_leaves': 127, 'min_data_in_leaf': 100}` | 499.0 | 2.1480 | 0.7113 |
| subsample_0.8 | `{'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1}` | 542.0 | 2.1627 | 0.7109 |
| l2_10 | `{'lambda_l2': 10}` | 94.0 | 2.1746 | 0.7044 |
| combo | `{'num_leaves': 63}` | 108.0 | 2.1679 | 0.6948 |

`combo` merges every single change that beat the baseline. Only one did, so it repeats `leaves_63`.
</details>

## Dashboard

The dashboard reads from PostgreSQL. It opens with the answer to the business question, then has four tabs. On the Stock plan tab you pick a store and category, set a safety buffer, and see how many of the 4 forecast weeks it would have covered and how many units would be short or left over.

![Stock plan](docs/1_stock_plan.png)
![Where the forecast is weak](docs/2_where_forecast_is_weak.png)
![Forecast vs actual](docs/3_forecast_vs_actual.png)
![Sales trends](docs/4_sales_trends.png)

## API

`GET /forecast?store_id=CA_1&item_id=FOODS_3_090` returns the 28 daily forecasts for that item. Add `model=` to pick `lgbm_tweedie_tuned` (the default), `lgbm_tweedie` or `lgbm_regression`. An unknown item returns 404, and if the database is down you get 503. `GET /health` checks the database connection.

## What's in the repo

```
notebooks/ src/       data prep, EDA, baselines, features, CV, LightGBM, WRMSSE
sql/                  schema, loaders, KPI queries
modeling/             WRMSSE metric, tuning, final forecasts
insights/             error analysis and the scripts that write INSIGHTS.md and this README
app/                  Streamlit dashboard (theme in .streamlit/)
api/                  FastAPI service and Dockerfile
outputs/              forecasts, per-series errors, tuning log, insight tables
docs/                 dashboard screenshots
```

The M5 files go in `raw/` and the parquet files the notebooks create go in `data/processed/`. Both are left out of the repo because of their size.

## Running it

You need Python 3.12 or newer and a local PostgreSQL.

```bash
python -m venv myenv && source myenv/bin/activate
pip install -r requirements.txt

# database settings come from environment variables (see .env.example)
export PGHOST=localhost PGPORT=5432 PGDATABASE=m5 PGUSER=<your_user>   # add PGPASSWORD if needed
createdb m5

python sql/load_data.py                    # calendar, prices, last 730 days of sales, saved predictions
python sql/check_counts.py                 # row counts vs source files
python sql/load_tuned_predictions.py       # adds the tuned forecasts

# optional: redo the modelling (about 15 minutes)
python modeling/tune_tweedie.py --run baseline
python modeling/tune_tweedie.py --run singles
python modeling/tune_tweedie.py --run combo
python modeling/finalize.py
python insights/analyse_errors.py && python insights/build_insights.py

streamlit run app/dashboard.py             # http://localhost:8501
uvicorn api.main:app --port 8000           # http://localhost:8000/docs
```

The three KPI queries run with `psql -d m5 -f sql/01_sales_by_store_cat_month.sql` (and `02_...`, `03_...`).

To run the API in Docker:

```bash
docker build -f api/Dockerfile -t m5-api .
docker run -p 8000:8000 -e PGHOST=host.docker.internal -e PGDATABASE=m5 -e PGUSER=<your_user> m5-api
```
If your database needs a password, pass `-e PGPASSWORD=...` when you run the container. It is never stored in the image.
