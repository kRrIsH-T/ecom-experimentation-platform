-- Weekly first-seen cohorts x weeks-since-first-seen retention matrix.
-- Caveat: the data starts on 2019-10-01, so the first cohort also contains users who were
-- active before the window (left censoring); later cohorts are clean.

-- name: cohort_retention
WITH activity AS (
    SELECT DISTINCT user_id, CAST(DATE_TRUNC('week', event_time) AS DATE) AS activity_week
    FROM ecom.events_clean
),
first_seen AS (
    SELECT user_id, MIN(activity_week) AS cohort_week
    FROM activity
    GROUP BY user_id
),
sizes AS (
    SELECT cohort_week, COUNT(*) AS cohort_size
    FROM first_seen
    GROUP BY cohort_week
),
joined AS (
    SELECT
        f.cohort_week,
        CAST(DATEDIFF(a.activity_week, f.cohort_week) / 7 AS INT) AS week_number,
        a.user_id
    FROM activity AS a
    JOIN first_seen AS f ON a.user_id = f.user_id
)
SELECT
    j.cohort_week,
    j.week_number,
    COUNT(DISTINCT j.user_id) AS active_users,
    s.cohort_size,
    COUNT(DISTINCT j.user_id) / s.cohort_size AS retention
FROM joined AS j
JOIN sizes AS s ON j.cohort_week = s.cohort_week
GROUP BY j.cohort_week, j.week_number, s.cohort_size
ORDER BY j.cohort_week, j.week_number;

-- name: cohort_buyer_retention
-- Of each weekly cohort of first-time buyers, the share that buys again in later weeks.
WITH purchase_weeks AS (
    SELECT DISTINCT user_id, CAST(DATE_TRUNC('week', event_time) AS DATE) AS purchase_week
    FROM ecom.events_clean
    WHERE event_type = 'purchase'
),
first_purchase AS (
    SELECT user_id, MIN(purchase_week) AS cohort_week
    FROM purchase_weeks
    GROUP BY user_id
)
SELECT
    f.cohort_week,
    CAST(DATEDIFF(p.purchase_week, f.cohort_week) / 7 AS INT) AS week_number,
    COUNT(DISTINCT p.user_id) AS buyers,
    COUNT(DISTINCT p.user_id) / MAX(COUNT(DISTINCT p.user_id)) OVER (PARTITION BY f.cohort_week)
        AS repeat_rate
FROM purchase_weeks AS p
JOIN first_purchase AS f ON p.user_id = f.user_id
GROUP BY f.cohort_week, CAST(DATEDIFF(p.purchase_week, f.cohort_week) / 7 AS INT)
ORDER BY f.cohort_week, week_number;
