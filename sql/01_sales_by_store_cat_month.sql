-- KPI: units sold per store, category and month
SELECT c.year,
       c.month,
       s.store_id,
       s.cat_id,
       SUM(f.sales) AS units_sold
FROM sales f
JOIN series   s USING (series_id)
JOIN calendar c USING (d_num)
GROUP BY c.year, c.month, s.store_id, s.cat_id
ORDER BY c.year, c.month, s.store_id, s.cat_id;
