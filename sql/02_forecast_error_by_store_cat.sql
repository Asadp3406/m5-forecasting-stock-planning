-- KPI: forecast error per store and category (Tweedie model, validation window)
-- MAE = average absolute miss; RMSE punishes big misses; bias > 0 means over-forecast.
SELECT s.store_id,
       s.cat_id,
       COUNT(*)                                          AS n_rows,
       ROUND(AVG(ABS(p.y_pred - f.sales))::numeric, 4)   AS mae,
       ROUND(SQRT(AVG(POWER(p.y_pred - f.sales, 2)))::numeric, 4) AS rmse,
       ROUND(AVG(p.y_pred - f.sales)::numeric, 4)        AS bias,
       ROUND((SUM(p.y_pred) / NULLIF(SUM(f.sales), 0))::numeric, 4) AS forecast_to_actual_ratio
FROM predictions p
JOIN sales  f USING (series_id, d_num)
JOIN series s USING (series_id)
WHERE p.model = 'lgbm_tweedie'
GROUP BY s.store_id, s.cat_id
ORDER BY mae DESC;
