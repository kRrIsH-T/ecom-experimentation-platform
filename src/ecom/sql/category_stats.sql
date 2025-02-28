-- Revenue, AOV and conversion by category, with ranks and Pareto (cumulative) shares.

-- name: category_stats
WITH orders AS (
    SELECT category_top, category_code, user_session, user_id, SUM(price) AS order_value
    FROM ecom.events_clean
    WHERE event_type = 'purchase'
    GROUP BY category_top, category_code, user_session, user_id
),
per_category AS (
    SELECT
        category_top,
        category_code,
        COUNT(*) AS orders,
        COUNT(DISTINCT user_id) AS buyers,
        SUM(order_value) AS revenue,
        AVG(order_value) AS aov,
        PERCENTILE_APPROX(order_value, 0.5) AS median_order_value
    FROM orders
    GROUP BY category_top, category_code
),
traffic AS (
    SELECT category_top, category_code, COUNT(DISTINCT user_session) AS view_sessions
    FROM ecom.events_clean
    WHERE event_type = 'view'
    GROUP BY category_top, category_code
)
SELECT
    c.*,
    t.view_sessions,
    c.orders / t.view_sessions AS session_conversion,
    RANK() OVER (ORDER BY c.revenue DESC) AS revenue_rank,
    RANK() OVER (PARTITION BY c.category_top ORDER BY c.revenue DESC) AS rank_within_top,
    c.revenue / SUM(c.revenue) OVER () AS revenue_share,
    SUM(c.revenue) OVER (ORDER BY c.revenue DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        / SUM(c.revenue) OVER () AS cumulative_share
FROM per_category AS c
JOIN traffic AS t ON c.category_top = t.category_top AND c.category_code = t.category_code
ORDER BY revenue_rank;

-- name: brand_pareto
WITH per_brand AS (
    SELECT brand, SUM(price) AS revenue, COUNT(*) AS purchases
    FROM ecom.events_clean
    WHERE event_type = 'purchase'
    GROUP BY brand
)
SELECT
    brand,
    revenue,
    purchases,
    RANK() OVER (ORDER BY revenue DESC) AS revenue_rank,
    revenue / SUM(revenue) OVER () AS revenue_share,
    SUM(revenue) OVER (ORDER BY revenue DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        / SUM(revenue) OVER () AS cumulative_share,
    ROW_NUMBER() OVER (ORDER BY revenue DESC) / COUNT(*) OVER () AS brand_share
FROM per_brand
ORDER BY revenue_rank;
