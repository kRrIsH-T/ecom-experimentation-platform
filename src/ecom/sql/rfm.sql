-- RFM segmentation of buyers with NTILE quintiles (5 = best).
-- Frequency is heavily tied at 1 order, so its quintile is ordered by (frequency, monetary).

-- name: rfm_users [table]
WITH ref AS (
    SELECT DATE_ADD(MAX(event_date), 1) AS ref_date FROM ecom.events_clean
),
buyers AS (
    SELECT
        e.user_id,
        DATEDIFF(MAX(r.ref_date), MAX(e.event_date)) AS recency_days,
        COUNT(DISTINCT e.user_session) AS frequency,
        SUM(e.price) AS monetary
    FROM ecom.events_clean AS e
    CROSS JOIN ref AS r
    WHERE e.event_type = 'purchase'
    GROUP BY e.user_id
),
scored AS (
    SELECT
        *,
        NTILE(5) OVER (ORDER BY recency_days DESC, user_id) AS r_score,
        NTILE(5) OVER (ORDER BY frequency, monetary, user_id) AS f_score,
        NTILE(5) OVER (ORDER BY monetary, user_id) AS m_score
    FROM buyers
)
SELECT
    *,
    CONCAT(r_score, f_score, m_score) AS rfm_code,
    CASE
        WHEN r_score >= 4 AND f_score >= 4 THEN 'champions'
        WHEN r_score <= 2 AND f_score >= 4 THEN 'at_risk'
        WHEN f_score >= 4 THEN 'loyal'
        WHEN r_score >= 4 AND f_score <= 2 THEN 'new_customers'
        WHEN r_score >= 3 AND f_score >= 3 THEN 'potential_loyalists'
        WHEN r_score <= 2 AND f_score <= 2 THEN 'hibernating'
        ELSE 'need_attention'
    END AS segment
FROM scored;

-- name: rfm_summary
SELECT
    segment,
    COUNT(*) AS users,
    AVG(recency_days) AS avg_recency_days,
    AVG(frequency) AS avg_frequency,
    AVG(monetary) AS avg_monetary,
    SUM(monetary) AS revenue,
    SUM(monetary) / SUM(SUM(monetary)) OVER () AS revenue_share,
    COUNT(*) / SUM(COUNT(*)) OVER () AS user_share
FROM ecom.rfm_users
GROUP BY segment
ORDER BY revenue DESC;
