# M5 Forecasting: from raw sales to stock-planning decisions

**Business question:** Which stores, categories and day types (events, SNAP days) are hardest to forecast, and what does that mean for stock planning?

Built on the [M5 Forecasting - Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) data (Walmart, 30,490 item-store series, daily unit sales, 2011-2016). The raw data is not included in this repo.

## What the analysis found

1. **Plan stock at store-category-week level.** Item-store-day forecasts miss 73.5% of units; at store-category-week level the miss is 6.9%.
2. **The model under-forecasts.** Forecast units are 94.3% of actual overall, and 88.5% at CA_2 (the worst store), so add a small buffer.
3. **FOODS carries the error, slow items are cheap to get wrong.** FOODS is 67.6% of units but 61.5% of the error. Items that are zero on over 80% of days are 9.6% of the error.
4. **SNAP days need a modest FOODS buffer.** FOODS misses 1.48 units per item-day on SNAP days versus 1.33 otherwise. Event days show no clear extra risk in this window.

Full write-up with the numbers and their sources: [INSIGHTS.md](INSIGHTS.md).

## Architecture

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

Validation setup: train on days up to d_1913, forecast the next 28 days (d_1914 to d_1941, 2016-04-25 to 2016-05-22). Time-based split, lags at least 28 days old, no random splits. Lower WRMSSE is better.

| Model | WRMSSE (project metric) | WRMSSE (official-style) |
|---|---|---|
| seasonal_naive | 0.9128 | 0.8697 |
| lgbm_regression (Stage 6) | 0.7174 | n/a |
| lgbm_tweedie baseline | 0.7036 | 0.6691 |
| **lgbm_tweedie tuned (leaves_63)** | **0.6948** | **0.6604** |

- **Project metric** is the WRMSSE from the original Stage 6 notebook (average over 12 hierarchy levels).
- **Official-style** follows the competition's definition, which ignores the days before an item's first sale when scaling the error. It scores 0.03 to 0.04 lower in this project. Both are shown because the project numbers cannot be compared directly with published M5 scores.
- Tuned Tweedie improves the baseline Tweedie by 1.3% (WRMSSE 0.7036 to 0.6948). Of 7 single changes tried, only `num_leaves=63` beat the baseline.

<details><summary>All tuning trials</summary>

| Trial | Change vs baseline | Rounds | RMSE | WRMSSE | WRMSSE (official-style) |
|---|---|---|---|---|---|
| baseline | `{}` | 110 | 2.1741 | 0.7036 | 0.6691 |
| vp_1.1 | `{'tweedie_variance_power': 1.1}` | 140 | 2.1803 | 0.7049 | 0.6703 |
| vp_1.3 | `{'tweedie_variance_power': 1.3}` | 188 | 2.1776 | 0.7144 | 0.6798 |
| lr_0.05 | `{'learning_rate': 0.05}` | 267 | 2.1704 | 0.7060 | 0.6715 |
| leaves_63 | `{'num_leaves': 63}` | 108 | 2.1679 | 0.6948 | 0.6604 |
| leaves_127_mdl100 | `{'num_leaves': 127, 'min_data_in_leaf': 100}` | 499 | 2.1480 | 0.7113 | 0.6775 |
| subsample_0.8 | `{'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1}` | 542 | 2.1627 | 0.7109 | 0.6766 |
| l2_10 | `{'lambda_l2': 10}` | 94 | 2.1746 | 0.7044 | 0.6697 |
| combo | `{'num_leaves': 63}` | 108 | 2.1679 | 0.6948 | 0.6604 |

`combo` merges every single change that beat the baseline. Only one did, so it repeats `leaves_63`.
</details>

## Dashboard

Streamlit app reading from PostgreSQL. A headline panel answers the business question first. Four tabs follow: a **Stock plan** page (pick a store and category, set a safety buffer, see how many of the 4 forecast weeks would have been covered, and the units short or left over), where the forecast is weak, forecast vs actual, and sales trends. The theme is set in `.streamlit/config.toml`.

![Stock plan](docs/1_stock_plan.png)
![Where the forecast is weak](docs/2_where_forecast_is_weak.png)
![Forecast vs actual](docs/3_forecast_vs_actual.png)
![Sales trends](docs/4_sales_trends.png)

## API

`GET /forecast?store_id=CA_1&item_id=FOODS_3_090` returns the 28 daily forecasts. Optional `model` parameter: `lgbm_tweedie_tuned` (default), `lgbm_tweedie`, `lgbm_regression`. Returns 404 for an unknown series and 503 if the database is unreachable. `GET /health` checks the database connection.

## Project structure

```
raw/                  M5 CSVs (not in repo)
data/processed/       parquet files from the notebooks (not in repo)
notebooks/ src/       Stages 0-6: data prep, EDA, baselines, features, CV, LightGBM
sql/                  schema, loaders, KPI queries
modeling/             WRMSSE metric, Tweedie tuning, final forecasts
insights/             error analysis and the generators for INSIGHTS.md / README.md
app/                  Streamlit dashboard (theme in .streamlit/)
api/                  FastAPI service and Dockerfile
outputs/              forecasts, per-series errors, tuning log, insight tables
docs/                 dashboard screenshots
```

## How to run

Prerequisites: Python 3.12 or newer, PostgreSQL running locally, and the M5 files in `raw/` plus the parquet files in `data/processed/` (created by the Stage 0 to 5 notebooks and scripts).

```bash
python -m venv myenv && source myenv/bin/activate
pip install -r requirements.txt

# database settings come from environment variables (see .env.example); no passwords in files
export PGHOST=localhost PGPORT=5432 PGDATABASE=m5 PGUSER=<your_user>   # add PGPASSWORD if needed
createdb m5

python sql/load_data.py                    # loads calendar, prices, last 730 days of sales, saved predictions
python sql/check_counts.py                 # row counts vs source files
python sql/load_tuned_predictions.py       # adds the tuned forecasts (needs outputs/forecast_28d.parquet)

# optional: reproduce the modelling results (about 15 minutes in total)
python modeling/tune_tweedie.py --run baseline
python modeling/tune_tweedie.py --run singles
python modeling/tune_tweedie.py --run combo
python modeling/finalize.py
python insights/analyse_errors.py && python insights/build_insights.py

streamlit run app/dashboard.py             # dashboard on http://localhost:8501
uvicorn api.main:app --port 8000           # API on http://localhost:8000/docs
```

SQL KPI queries: `psql -d m5 -f sql/01_sales_by_store_cat_month.sql` (also `02_...` and `03_...`).

Docker:

```bash
docker build -f api/Dockerfile -t m5-api .
docker run -p 8000:8000 -e PGHOST=host.docker.internal -e PGDATABASE=m5 -e PGUSER=<your_user> m5-api
```
Pass `PGPASSWORD` with `-e` at run time if your database needs one. It is never stored in the image.

## Limitations

- **One validation window.** All results are for a single 28-day window. The tuned model was chosen on that same window, so its small gain (1.3%) may be partly selection luck. A second fold would confirm it.
- **WRMSSE definition.** The metric lives in `modeling/wrmsse.py` and reproduces the Stage 6 scores for both models. It differs from the official M5 scorer as described above.
- **SQL history.** PostgreSQL holds the last 730 days of sales (about 22 million rows), not the full 59 million. Model training used the full history from the parquet files.
- **No cost data.** Buffer suggestions in INSIGHTS.md are based on forecast bias only. There was no stockout or holding-cost data.
- **SNAP and event findings are associations,** and the window has only 4 event days.
- **API and dashboard serve saved forecasts.** They do not run the model live, and they cover only the validation window.
