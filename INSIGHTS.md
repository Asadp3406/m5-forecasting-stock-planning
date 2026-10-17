# INSIGHTS — Where the forecast is weak, and what to do about it

**Question:** Which stores, categories and day types (events, SNAP days) are hardest to forecast, and what does that mean for stock planning?

**How to read this.** Model: tuned Tweedie LightGBM (WRMSSE 0.6948). Window: the 28 days 2016-04-25 to 2016-05-22, never seen in training. *WAPE* = total units missed / total units sold (lower is better, comparable across big and small groups). *Forecast-to-actual* below 100% means the model under-forecasts. Every number below is filled in by `insights/build_insights.py` from the tables in `outputs/insights/`.

## Recommendations

### 1. Plan stock at store-category-week level, not item-by-day
Forecast error shrinks sharply as you plan at a coarser level: 73.5% of units are missed at item-store-day level, 38.6% at item-store-week, 8.9% at store-category-day and 6.9% at store-category-week. Use the model to set category-level order volumes per store each week, and treat item-level splits as an allocation of that total, not a precise daily prediction.
*(Source: `error_by_planning_level.csv`)*

### 2. Add a small buffer: the model under-forecasts almost everywhere
Across all stores the forecast totals 94.3% of actual units. Nine of ten stores are under-forecast; the worst is CA_2 at 88.5%, while CA_3 is slightly over (101.8%). Raise baseline order quantities by roughly 6.1% chainwide and by 13.1% at CA_2; do not add a buffer at CA_3.
*(Source: `error_by_store.csv`)*

### 3. Spend review effort on FOODS; manage slow items with simple rules
- FOODS is 67.6% of units but 61.5% of all forecast error, and the largest absolute misses are in WI_2 FOODS (MAE 1.80 units per item-day).
- Items that sell on more than half of days (10,184 series) produce 59.6% of all error. Slow items (zero sales on over 80% of days, 7,979 series) have WAPE of 130.8% to 140.6% (misses larger than the units sold) but only 9.6% of the error and 5.0% of units. They are hard to forecast but cheap to be wrong about, so use min/max reorder points or category-level pooling for them instead of chasing a better item forecast.
- HOBBIES has the highest WAPE of any category (101.1%) versus 66.9% for FOODS, but it is only 9.4% of units and 12.9% of the error. Low volume per item, so keep stock lean and accept some misses.
*(Sources: `error_by_store_category.csv`, `error_by_category.csv`, `error_by_intermittency.csv`)*

### 4. SNAP days need a modest FOODS buffer; event days show no clear extra risk
- On SNAP days the average miss for FOODS is 1.48 units per item-day versus 1.33 on other days (11.2% higher), and the model under-forecasts slightly more (92.7% vs 93.9% of actual). Relative error (WAPE) is not worse (65.6% vs 67.7%), so the extra miss likely tracks higher sales volume. Keep FOODS stock a little deeper in SNAP weeks, not dramatically so.
- Event days: the window only contains 4 event days of 28, so this data cannot support a firm conclusion. MAE is 1.12 on event days versus 1.05 on other days, but WAPE is 71.8% versus 73.8% (no worse). Test this on more event periods before changing any stocking rule.
*(Sources: `error_by_snap_category.csv`, `error_by_event.csv`, `window_days.csv`)*

## Limits of this analysis
- One validation window (28 days), and the tuned model was chosen on that same window, so the numbers are slightly optimistic.
- WAPE and MAE describe unit errors. No cost-of-stockout or holding-cost data was available, so buffer sizes are indicative, not optimal.
- SNAP and event differences are associations from one window, not proven causes.
