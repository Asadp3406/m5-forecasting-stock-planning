-- Schema for the M5 forecasting project. Safe to re-run (IF NOT EXISTS).

-- One row per day (from calendar.csv)
CREATE TABLE IF NOT EXISTS calendar (
    d_num        INT PRIMARY KEY,      -- 1 = 2011-01-29 ... 1941 = 2016-05-22
    date         DATE NOT NULL UNIQUE,
    wm_yr_wk     INT NOT NULL,
    weekday      TEXT NOT NULL,
    wday         SMALLINT NOT NULL,
    month        SMALLINT NOT NULL,
    year         SMALLINT NOT NULL,
    event_name_1 TEXT,
    event_type_1 TEXT,
    event_name_2 TEXT,
    event_type_2 TEXT,
    snap_ca      SMALLINT NOT NULL,
    snap_tx      SMALLINT NOT NULL,
    snap_wi      SMALLINT NOT NULL
);

-- One row per product-store series (30,490 rows)
CREATE TABLE IF NOT EXISTS series (
    series_id INT PRIMARY KEY,
    id        TEXT NOT NULL UNIQUE,    -- e.g. HOBBIES_1_001_CA_1_evaluation
    item_id   TEXT NOT NULL,
    dept_id   TEXT NOT NULL,
    cat_id    TEXT NOT NULL,
    store_id  TEXT NOT NULL,
    state_id  TEXT NOT NULL
);

-- Weekly price per store and item (from sell_prices.csv)
CREATE TABLE IF NOT EXISTS sell_prices (
    store_id   TEXT NOT NULL,
    item_id    TEXT NOT NULL,
    wm_yr_wk   INT NOT NULL,
    sell_price NUMERIC(8,2) NOT NULL,
    PRIMARY KEY (store_id, item_id, wm_yr_wk)
);

-- Daily unit sales per series (long format, last ~2 years only)
CREATE TABLE IF NOT EXISTS sales (
    series_id INT NOT NULL REFERENCES series(series_id),
    d_num     INT NOT NULL REFERENCES calendar(d_num),
    sales     SMALLINT NOT NULL,
    PRIMARY KEY (series_id, d_num)
);
CREATE INDEX IF NOT EXISTS idx_sales_d ON sales (d_num);

-- Saved model forecasts for the validation window (d_1914 - d_1941)
CREATE TABLE IF NOT EXISTS predictions (
    series_id INT NOT NULL REFERENCES series(series_id),
    d_num     INT NOT NULL REFERENCES calendar(d_num),
    model     TEXT NOT NULL,           -- 'lgbm_regression', 'lgbm_tweedie', ...
    y_pred    DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (series_id, d_num, model)
);
