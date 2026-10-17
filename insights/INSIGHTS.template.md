# INSIGHTS — Where the forecast is weak, and what to do about it

**Question:** Which stores, categories and day types (events, SNAP days) are hardest to forecast, and what does that mean for stock planning?

**How to read this.** Model: tuned Tweedie LightGBM (WRMSSE {wrmsse}). Window: the 28 days {win_start} to {win_end}, never seen in training. *WAPE* = total units missed / total units sold (lower is better, comparable across big and small groups). *Forecast-to-actual* below 100% means the model under-forecasts. Every number below is filled in by `insights/build_insights.py` from the tables in `outputs/insights/`.

## Recommendations

### 1. Plan stock at store-category-week level, not item-by-day
Forecast error shrinks sharply as you plan at a coarser level: {wape_item_daily} of units are missed at item-store-day level, {wape_item_weekly} at item-store-week, {wape_sc_daily} at store-category-day and {wape_sc_weekly} at store-category-week. Use the model to set category-level order volumes per store each week, and treat item-level splits as an allocation of that total, not a precise daily prediction.
*(Source: `error_by_planning_level.csv`)*

### 2. Add a small buffer: the model under-forecasts almost everywhere
Across all stores the forecast totals {overall_ratio} of actual units. Nine of ten stores are under-forecast; the worst is {worst_store} at {worst_store_ratio}, while {over_store} is slightly over ({over_store_ratio}). Raise baseline order quantities by roughly {buffer_overall} chainwide and by {buffer_worst} at {worst_store}; do not add a buffer at {over_store}.
*(Source: `error_by_store.csv`)*

### 3. Spend review effort on FOODS; manage slow items with simple rules
- FOODS is {foods_units_share} of units but {foods_err_share} of all forecast error, and the largest absolute misses are in {top_mae_store} FOODS (MAE {top_mae_store_value} units per item-day).
- Items that sell on more than half of days ({n_series_low_zero} series) produce {low_zero_err_share} of all error. Slow items (zero sales on over 80% of days, {n_series_high_zero} series) have WAPE of {wape_high_zero_80} to {wape_high_zero_95} (misses larger than the units sold) but only {high_zero_err_share} of the error and {high_zero_units_share} of units. They are hard to forecast but cheap to be wrong about, so use min/max reorder points or category-level pooling for them instead of chasing a better item forecast.
- HOBBIES has the highest WAPE of any category ({hobbies_wape}) versus {foods_wape} for FOODS, but it is only {hobbies_units_share} of units and {hobbies_err_share} of the error. Low volume per item, so keep stock lean and accept some misses.
*(Sources: `error_by_store_category.csv`, `error_by_category.csv`, `error_by_intermittency.csv`)*

### 4. SNAP days need a modest FOODS buffer; event days show no clear extra risk
- On SNAP days the average miss for FOODS is {foods_snap_mae} units per item-day versus {foods_nosnap_mae} on other days ({foods_snap_mae_up} higher), and the model under-forecasts slightly more ({foods_snap_ratio} vs {foods_nosnap_ratio} of actual). Relative error (WAPE) is not worse ({foods_snap_wape} vs {foods_nosnap_wape}), so the extra miss likely tracks higher sales volume. Keep FOODS stock a little deeper in SNAP weeks, not dramatically so.
- Event days: the window only contains {n_event_days} event days of {n_days}, so this data cannot support a firm conclusion. MAE is {event_mae} on event days versus {nonevent_mae} on other days, but WAPE is {event_wape} versus {nonevent_wape} (no worse). Test this on more event periods before changing any stocking rule.
*(Sources: `error_by_snap_category.csv`, `error_by_event.csv`, `window_days.csv`)*

## Limits of this analysis
- One validation window (28 days), and the tuned model was chosen on that same window, so the numbers are slightly optimistic.
- WAPE and MAE describe unit errors. No cost-of-stockout or holding-cost data was available, so buffer sizes are indicative, not optimal.
- SNAP and event differences are associations from one window, not proven causes.
