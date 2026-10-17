"""M5 forecasting dashboard (Streamlit). Reads everything from PostgreSQL.

Run from project root:  streamlit run app/dashboard.py
Database settings come from PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD.
"""
import altair as alt
import pandas as pd
import psycopg
import streamlit as st

st.set_page_config(page_title="M5 forecast and stock plan", layout="wide")

# fixed colour per category (never changes when filters change)
CAT_COLORS = {"FOODS": "#2a78d6", "HOBBIES": "#eb6834", "HOUSEHOLD": "#1baf7a"}
ACTUAL, FORECAST = "#2a78d6", "#eb6834"
MODELS = {"lgbm_tweedie_tuned": "Tweedie (tuned)", "lgbm_tweedie": "Tweedie (baseline)",
          "lgbm_regression": "Regression"}


@st.cache_resource
def get_conn():
    return psycopg.connect(autocommit=True)  # settings from PG* env vars


@st.cache_data(show_spinner=False)
def query(sql, params=()):
    """Run a SQL query and return a DataFrame."""
    with get_conn().cursor() as cur:
        cur.execute(sql, params)
        cols = [c.name for c in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def pick_filters():
    """Sidebar filters; returns (stores, cats, model)."""
    all_stores = query("SELECT DISTINCT store_id FROM series ORDER BY 1")["store_id"].tolist()
    all_cats = list(CAT_COLORS)
    st.sidebar.header("Filters")
    stores = st.sidebar.multiselect("Stores", all_stores, default=all_stores)
    cats = st.sidebar.multiselect("Categories", all_cats, default=all_cats)
    model = st.sidebar.selectbox("Forecast model", list(MODELS), format_func=MODELS.get)
    st.sidebar.caption("Forecast window: 28 days after the training cutoff "
                       "(2016-04-25 to 2016-05-22), never seen in training.")
    return stores or all_stores, cats or all_cats, model


def cat_color_scale():
    return alt.Scale(domain=list(CAT_COLORS), range=list(CAT_COLORS.values()))


# ---------- page 1: sales trends ----------
def sales_trends(stores, cats):
    # monthly units; keep only full calendar months (data starts mid-May 2014)
    df = query("""
        SELECT c.year, c.month, s.cat_id, SUM(f.sales) AS units,
               COUNT(DISTINCT c.d_num) AS n_days
        FROM sales f JOIN series s USING (series_id) JOIN calendar c USING (d_num)
        WHERE s.store_id = ANY(%s) AND s.cat_id = ANY(%s)
        GROUP BY c.year, c.month, s.cat_id""", (stores, cats))
    dim = pd.to_datetime(dict(year=df.year, month=df.month, day=1)).dt.days_in_month
    df = df[df["n_days"] == dim]
    df["month_start"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))

    total = int(df["units"].sum())
    top_cat = df.groupby("cat_id")["units"].sum().idxmax()
    c1, c2, c3 = st.columns(3)
    c1.metric("Units sold (full months shown)", f"{total:,}")
    c2.metric("Months shown", df["month_start"].nunique())
    c3.metric("Biggest category", top_cat)

    st.subheader("Monthly units sold by category")
    line = alt.Chart(df).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=40)).encode(
        x=alt.X("month_start:T", title=None, axis=alt.Axis(format="%b %Y", labelAngle=-45)),
        y=alt.Y("units:Q", title="Units sold"),
        color=alt.Color("cat_id:N", scale=cat_color_scale(), title="Category"),
        tooltip=[alt.Tooltip("month_start:T", title="Month", format="%b %Y"),
                 "cat_id", alt.Tooltip("units:Q", format=",")])
    st.altair_chart(line, width="stretch")

    st.subheader("Units sold by store")
    by_store = query("""
        SELECT s.store_id, SUM(f.sales) AS units
        FROM sales f JOIN series s USING (series_id)
        WHERE s.store_id = ANY(%s) AND s.cat_id = ANY(%s)
        GROUP BY s.store_id ORDER BY units DESC""", (stores, cats))
    bars = alt.Chart(by_store).mark_bar(color=ACTUAL, cornerRadiusEnd=4).encode(
        y=alt.Y("store_id:N", sort="-x", title=None), x=alt.X("units:Q", title="Units sold"),
        tooltip=["store_id", alt.Tooltip("units:Q", format=",")])
    st.altair_chart(bars, width="stretch")
    with st.expander("Table view"):
        st.dataframe(df[["month_start", "cat_id", "units"]].sort_values(["month_start", "cat_id"]))


# ---------- page 2: forecast vs actual ----------
def forecast_vs_actual(stores, cats, model):
    st.subheader("Forecast vs actual, validation window")
    col1, col2 = st.columns(2)
    store = col1.selectbox("Store", ["All selected stores"] + stores)
    cat = col2.selectbox("Category", ["All selected categories"] + cats)
    store_f = stores if store.startswith("All") else [store]
    cat_f = cats if cat.startswith("All") else [cat]

    item = None
    if not store.startswith("All") and not cat.startswith("All"):
        items = query("SELECT item_id FROM series WHERE store_id=%s AND cat_id=%s ORDER BY 1",
                      (store, cat))["item_id"].tolist()
        choice = st.selectbox("Item (optional)", ["All items in this store and category"] + items)
        item = None if choice.startswith("All") else choice

    df = query("""
        SELECT c.date, SUM(f.sales) AS actual, SUM(p.y_pred) AS forecast
        FROM predictions p
        JOIN sales f USING (series_id, d_num) JOIN series s USING (series_id)
        JOIN calendar c USING (d_num)
        WHERE p.model = %s AND s.store_id = ANY(%s) AND s.cat_id = ANY(%s)
          AND (%s::text IS NULL OR s.item_id = %s)
        GROUP BY c.date ORDER BY c.date""", (model, store_f, cat_f, item, item))

    wape = (df.forecast - df.actual).abs().sum() / df.actual.sum() if df.actual.sum() else float("nan")
    ratio = df.forecast.sum() / df.actual.sum() if df.actual.sum() else float("nan")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Actual units (28 days)", f"{int(df.actual.sum()):,}")
    c2.metric("Forecast units (28 days)", f"{df.forecast.sum():,.0f}")
    c3.metric("Error on daily totals (WAPE)", f"{wape:.1%}")
    c4.metric("Forecast / actual", f"{ratio:.1%}")

    long = df.melt("date", var_name="series", value_name="units")
    long["series"] = long["series"].str.capitalize()
    chart = alt.Chart(long).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=36)).encode(
        x=alt.X("date:T", title=None), y=alt.Y("units:Q", title="Units"),
        color=alt.Color("series:N", title=None,
                        scale=alt.Scale(domain=["Actual", "Forecast"], range=[ACTUAL, FORECAST])),
        tooltip=[alt.Tooltip("date:T", format="%a %d %b"), "series",
                 alt.Tooltip("units:Q", format=",.1f")])
    st.altair_chart(chart, width="stretch")
    st.caption("Single items sell few units a day, so their daily forecast is noisy. "
               "Totals over many items are much steadier (see Insights, recommendation 1).")
    with st.expander("Table view"):
        st.dataframe(df)


# ---------- page 3: error by store, category and day type ----------
def error_page(stores, cats, model):
    metric = st.radio("Error measure", ["WAPE (share of units missed)", "MAE (units per item-day)"],
                      horizontal=True)
    use_wape = metric.startswith("WAPE")

    err = query("""
        SELECT s.store_id, s.cat_id,
               SUM(ABS(p.y_pred - f.sales)) AS abs_err, SUM(f.sales) AS actual,
               SUM(p.y_pred) AS forecast, COUNT(*) AS n
        FROM predictions p JOIN sales f USING (series_id, d_num) JOIN series s USING (series_id)
        WHERE p.model = %s AND s.store_id = ANY(%s) AND s.cat_id = ANY(%s)
        GROUP BY s.store_id, s.cat_id""", (model, stores, cats))
    err["WAPE"] = err.abs_err / err.actual
    err["MAE"] = err.abs_err / err.n
    err["forecast_to_actual"] = err.forecast / err.actual
    col = "WAPE" if use_wape else "MAE"
    fmt = ".0%" if use_wape else ".2f"

    st.subheader(f"{col} by store and category (item-store-day level)")
    heat = alt.Chart(err).mark_rect(stroke="white", strokeWidth=2).encode(
        x=alt.X("cat_id:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("store_id:N", title=None),
        color=alt.Color(f"{col}:Q", scale=alt.Scale(range=["#d6e6fa", "#0b3d78"]), title=col,
                        legend=alt.Legend(format=fmt)),
        tooltip=["store_id", "cat_id", alt.Tooltip(f"{col}:Q", format=fmt),
                 alt.Tooltip("forecast_to_actual:Q", format=".1%", title="Forecast / actual")])
    text = heat.mark_text(fontSize=12).encode(
        text=alt.Text(f"{col}:Q", format=fmt),
        color=alt.condition(alt.datum[col] > err[col].median(), alt.value("white"), alt.value("black")))
    st.altair_chart(heat + text, width="stretch")

    st.subheader("Error by day type")
    day = query("""
        SELECT CASE WHEN c.event_name_1 IS NOT NULL THEN 'Event day' ELSE 'No event' END AS event,
               (CASE s.state_id WHEN 'CA' THEN c.snap_ca WHEN 'TX' THEN c.snap_tx
                                ELSE c.snap_wi END = 1) AS is_snap,
               s.cat_id,
               SUM(ABS(p.y_pred - f.sales)) AS abs_err, SUM(f.sales) AS actual, COUNT(*) AS n,
               COUNT(DISTINCT c.d_num) AS n_days
        FROM predictions p JOIN sales f USING (series_id, d_num) JOIN series s USING (series_id)
        JOIN calendar c USING (d_num)
        WHERE p.model = %s AND s.store_id = ANY(%s) AND s.cat_id = ANY(%s)
        GROUP BY 1, 2, 3""", (model, stores, cats))
    day["is_snap"] = day["is_snap"].map({True: "SNAP day", False: "No SNAP"})

    def summarise(group_col):
        g = day.groupby(group_col).agg(abs_err=("abs_err", "sum"), actual=("actual", "sum"), n=("n", "sum"))
        g["WAPE"], g["MAE"] = g.abs_err / g.actual, g.abs_err / g.n
        return g.reset_index()

    left, right = st.columns(2)
    for holder, name in [(left, "event"), (right, "is_snap")]:
        g = summarise(name)
        ch = alt.Chart(g).mark_bar(color=ACTUAL, cornerRadiusEnd=4).encode(
            y=alt.Y(f"{name}:N", title=None), x=alt.X(f"{col}:Q", title=col, axis=alt.Axis(format=fmt)),
            tooltip=[name, alt.Tooltip(f"{col}:Q", format=fmt)]).properties(height=alt.Step(44))
        holder.altair_chart(ch, width="stretch")
    st.caption("The window contains only 4 event days out of 28, so event-day results are "
               "indicative only. SNAP and event gaps are associations, not proven causes.")
    with st.expander("Table view"):
        st.dataframe(err[["store_id", "cat_id", "WAPE", "MAE", "forecast_to_actual"]]
                     .sort_values("WAPE", ascending=False))


# ---------- headline: the answer to the business question ----------
def headline(stores, cats, model):
    """Top banner: live numbers that answer the business question."""
    h = query("""
        WITH r AS (
            SELECT s.store_id, s.cat_id, (p.d_num - 1914) / 7 AS wk,
                   p.y_pred, f.sales
            FROM predictions p JOIN sales f USING (series_id, d_num) JOIN series s USING (series_id)
            WHERE p.model = %s AND s.store_id = ANY(%s) AND s.cat_id = ANY(%s))
        SELECT (SELECT SUM(ABS(y_pred - sales)) / SUM(sales) FROM r)           AS wape_item_day,
               (SELECT SUM(ABS(a - b)) / SUM(b) FROM
                   (SELECT SUM(y_pred) AS a, SUM(sales) AS b FROM r GROUP BY store_id, cat_id, wk) t) AS wape_sc_week,
               (SELECT SUM(y_pred) / SUM(sales) FROM r)                         AS ratio""",
              (model, stores, cats)).iloc[0]
    st.markdown("#### The answer in three numbers")
    c1, c2, c3 = st.columns(3)
    c1.metric("Units missed, item-store-day", f"{h.wape_item_day:.1%}",
              help="Total absolute forecast error divided by total units sold, at the finest level.")
    c2.metric("Units missed, store-category-week", f"{h.wape_sc_week:.1%}",
              help="Same measure after adding items up to store, category and week.")
    c3.metric("Forecast / actual units", f"{h.ratio:.1%}",
              help="Below 100% means the model under-forecasts.")
    st.markdown(
        "**Forecasts are hard at item-day level and reliable at store-category-week level, "
        "and they run low. Plan stock in weekly category totals and add a buffer.** "
        "Open the *Stock plan* tab to test a buffer.")


# ---------- page: stock plan ----------
def stock_plan(stores, cats, model):
    st.subheader("Stock plan: turn the forecast into an order quantity")
    col1, col2 = st.columns(2)
    store = col1.selectbox("Store", ["All selected stores"] + stores, key="sp_store")
    cat = col2.selectbox("Category", ["All selected categories"] + cats, key="sp_cat")
    store_f = stores if store.startswith("All") else [store]
    cat_f = cats if cat.startswith("All") else [cat]

    wk = query("""
        SELECT (p.d_num - 1914) / 7 + 1 AS week, SUM(p.y_pred) AS forecast, SUM(f.sales) AS actual
        FROM predictions p JOIN sales f USING (series_id, d_num) JOIN series s USING (series_id)
        WHERE p.model = %s AND s.store_id = ANY(%s) AND s.cat_id = ANY(%s)
        GROUP BY 1 ORDER BY 1""", (model, store_f, cat_f))

    # suggested buffer = how much the model under-forecast in this window
    suggested = max(0.0, wk.actual.sum() / wk.forecast.sum() - 1)
    buffer = st.slider("Safety buffer on top of the forecast", 0, 30, round(suggested * 100),
                       format="%d%%") / 100
    st.caption(f"Default = {suggested:.1%}, the amount the model under-forecast for this selection. "
               "It comes from the same 4 weeks it is tested on, so the result below is optimistic. "
               "In real use, set the buffer from earlier weeks.")

    wk["order"] = wk.forecast * (1 + buffer)
    wk["gap"] = wk.order - wk.actual                 # > 0 surplus, < 0 shortage
    short, surplus = (-wk.gap.clip(upper=0)).sum(), wk.gap.clip(lower=0).sum()
    covered = int((wk.gap >= 0).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Units to order (4 weeks)", f"{wk.order.sum():,.0f}")
    c2.metric("Weeks fully covered", f"{covered} of {len(wk)}")
    c3.metric("Units short", f"{short:,.0f}")
    c4.metric("Units left over", f"{surplus:,.0f}")

    long = wk.melt("week", value_vars=["forecast", "order", "actual"], var_name="what", value_name="units")
    names = {"forecast": "Forecast", "order": "Order (forecast + buffer)", "actual": "Actually sold"}
    long["what"] = long["what"].map(names)
    chart = alt.Chart(long).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("week:O", title="Week of the forecast window", axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("what:N", sort=list(names.values())),
        y=alt.Y("units:Q", title="Units"),
        color=alt.Color("what:N", title=None, sort=list(names.values()),
                        scale=alt.Scale(domain=list(names.values()), range=[FORECAST, "#52514e", ACTUAL])),
        tooltip=["week", "what", alt.Tooltip("units:Q", format=",.0f")])
    st.altair_chart(chart, width="stretch")
    st.caption("Short = a week where the order is below what sold. Left over = order above what sold. "
               "Raising the buffer trades shortages for leftover stock.")
    with st.expander("Table view"):
        st.dataframe(wk.rename(columns={"gap": "order minus actual"}).round(1))


def main():
    st.title("Where is the forecast weak, and what does it mean for stock?")
    st.caption("M5 Walmart item-store daily sales, 28-day forecasts. Live from PostgreSQL.")
    try:
        stores, cats, model = pick_filters()
        headline(stores, cats, model)
    except Exception as e:  # most likely: database not reachable or env vars missing
        st.error(f"Cannot reach the database. Check the PG* environment variables. ({e})")
        st.stop()
    tabs = st.tabs(["Stock plan", "Where the forecast is weak", "Forecast vs actual", "Sales trends"])
    with tabs[0]:
        stock_plan(stores, cats, model)
    with tabs[1]:
        error_page(stores, cats, model)
    with tabs[2]:
        forecast_vs_actual(stores, cats, model)
    with tabs[3]:
        sales_trends(stores, cats)


main()
