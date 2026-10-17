-- KPI: forecast error on event days, SNAP days and normal days (Tweedie, validation window)
-- SNAP flag is taken for the series' own state. Event = any event on that day.
WITH rows_ AS (
    SELECT s.cat_id,
           (c.event_name_1 IS NOT NULL)                  AS is_event,
           CASE s.state_id WHEN 'CA' THEN c.snap_ca
                           WHEN 'TX' THEN c.snap_tx
                           ELSE c.snap_wi END = 1         AS is_snap,
           p.y_pred, f.sales
    FROM predictions p
    JOIN sales    f USING (series_id, d_num)
    JOIN series   s USING (series_id)
    JOIN calendar c USING (d_num)
    WHERE p.model = 'lgbm_tweedie'
)
SELECT is_event,
       is_snap,
       COUNT(*)                                          AS n_rows,
       ROUND(AVG(ABS(y_pred - sales))::numeric, 4)       AS mae,
       ROUND(SQRT(AVG(POWER(y_pred - sales, 2)))::numeric, 4) AS rmse,
       ROUND(AVG(y_pred - sales)::numeric, 4)            AS bias
FROM rows_
GROUP BY is_event, is_snap
ORDER BY is_event, is_snap;
