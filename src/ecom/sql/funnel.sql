-- View -> cart -> purchase funnels. A user (or session) only counts at a stage if it also
-- reached every earlier stage, so the funnel is monotonic by construction.

-- name: funnel_overall
WITH user_stage AS (
    SELECT
        user_id,
        MAX(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS viewed,
        MAX(CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END) AS carted,
        MAX(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchased
    FROM ecom.events_clean
    GROUP BY user_id
),
stages AS (
    SELECT 1 AS stage_order, 'view' AS stage, SUM(viewed) AS users FROM user_stage
    UNION ALL
    SELECT 2, 'cart', SUM(viewed * carted) FROM user_stage
    UNION ALL
    SELECT 3, 'purchase', SUM(viewed * carted * purchased) FROM user_stage
)
SELECT
    stage_order,
    stage,
    users,
    users / FIRST_VALUE(users) OVER (ORDER BY stage_order) AS pct_of_viewers,
    users / LAG(users) OVER (ORDER BY stage_order) AS step_conversion
FROM stages
ORDER BY stage_order;

-- name: funnel_sessions
WITH stages AS (
    SELECT 1 AS stage_order, 'session' AS stage, COUNT(*) AS sessions FROM ecom.sessions
    UNION ALL
    SELECT 2, 'cart', SUM(CASE WHEN carted THEN 1 ELSE 0 END) FROM ecom.sessions
    UNION ALL
    SELECT 3, 'purchase', SUM(CASE WHEN carted AND purchased THEN 1 ELSE 0 END) FROM ecom.sessions
)
SELECT
    stage_order,
    stage,
    sessions,
    sessions / FIRST_VALUE(sessions) OVER (ORDER BY stage_order) AS pct_of_sessions,
    sessions / LAG(sessions) OVER (ORDER BY stage_order) AS step_conversion
FROM stages
ORDER BY stage_order;

-- name: funnel_by_category
WITH per_category AS (
    SELECT
        category_top,
        COUNT(DISTINCT CASE WHEN event_type = 'view' THEN user_session END) AS view_sessions,
        COUNT(DISTINCT CASE WHEN event_type = 'cart' THEN user_session END) AS cart_sessions,
        COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN user_session END) AS purchase_sessions,
        SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS views,
        SUM(CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END) AS carts,
        SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchases,
        SUM(CASE WHEN event_type = 'purchase' THEN price ELSE 0 END) AS revenue
    FROM ecom.events_clean
    GROUP BY category_top
)
SELECT
    *,
    cart_sessions / view_sessions AS view_to_cart,
    purchase_sessions / NULLIF(cart_sessions, 0) AS cart_to_purchase,
    purchase_sessions / view_sessions AS conversion,
    RANK() OVER (ORDER BY purchase_sessions / view_sessions DESC) AS conversion_rank
FROM per_category
ORDER BY views DESC;

-- name: funnel_by_brand
WITH per_brand AS (
    SELECT
        brand,
        COUNT(DISTINCT CASE WHEN event_type = 'view' THEN user_session END) AS view_sessions,
        COUNT(DISTINCT CASE WHEN event_type = 'cart' THEN user_session END) AS cart_sessions,
        COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN user_session END) AS purchase_sessions,
        SUM(CASE WHEN event_type = 'purchase' THEN price ELSE 0 END) AS revenue
    FROM ecom.events_clean
    WHERE brand <> 'unknown'
    GROUP BY brand
)
SELECT
    *,
    cart_sessions / view_sessions AS view_to_cart,
    purchase_sessions / view_sessions AS conversion,
    RANK() OVER (ORDER BY view_sessions DESC) AS traffic_rank
FROM per_brand
ORDER BY view_sessions DESC
LIMIT 25;
