-- Headline KPIs and time patterns for the overview tab and EDA.

-- name: overview
SELECT
    COUNT(*) AS events,
    COUNT(DISTINCT user_id) AS users,
    COUNT(DISTINCT user_session) AS sessions,
    COUNT(DISTINCT product_id) AS products,
    SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchases,
    SUM(CASE WHEN event_type = 'purchase' THEN price ELSE 0 END) AS revenue,
    COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN user_id END) AS buyers,
    COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN user_session END)
        / COUNT(DISTINCT user_session) AS session_conversion,
    COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN user_id END)
        / COUNT(DISTINCT user_id) AS user_conversion,
    MIN(event_date) AS first_date,
    MAX(event_date) AS last_date
FROM ecom.events_clean;

-- name: daily_kpis
SELECT
    session_date,
    COUNT(*) AS sessions,
    COUNT(DISTINCT user_id) AS users,
    SUM(CASE WHEN purchased THEN 1 ELSE 0 END) AS purchasing_sessions,
    SUM(revenue) AS revenue,
    SUM(CASE WHEN purchased THEN 1 ELSE 0 END) / COUNT(*) AS conversion,
    SUM(revenue) / NULLIF(SUM(CASE WHEN purchased THEN 1 ELSE 0 END), 0) AS aov,
    AVG(SUM(revenue)) OVER (ORDER BY session_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
        AS revenue_7d_avg
FROM ecom.sessions
GROUP BY session_date
ORDER BY session_date;

-- name: hour_dow
SELECT
    dow,
    start_hour,
    COUNT(*) AS sessions,
    SUM(CASE WHEN purchased THEN 1 ELSE 0 END) / COUNT(*) AS conversion
FROM ecom.sessions
GROUP BY dow, start_hour
ORDER BY dow, start_hour;

-- name: revenue_concentration
WITH per_user AS (
    SELECT user_id, SUM(CASE WHEN event_type = 'purchase' THEN price ELSE 0 END) AS revenue
    FROM ecom.events_clean
    GROUP BY user_id
),
ranked AS (
    SELECT revenue, NTILE(1000) OVER (ORDER BY revenue DESC, user_id) AS permille
    FROM per_user
)
SELECT
    SUM(CASE WHEN permille <= 1 THEN revenue ELSE 0 END) / SUM(revenue) AS top_0_1pct_share,
    SUM(CASE WHEN permille <= 10 THEN revenue ELSE 0 END) / SUM(revenue) AS top_1pct_share,
    SUM(CASE WHEN permille <= 100 THEN revenue ELSE 0 END) / SUM(revenue) AS top_10pct_share,
    SUM(CASE WHEN revenue > 0 THEN 1 ELSE 0 END) / COUNT(*) AS buyer_share
FROM ranked;
