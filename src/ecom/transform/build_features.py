"""Gold ``user_features``: point-in-time behavioural features per user and snapshot.

A snapshot is defined by a cutoff date. Features only see events in
``[cutoff - feature_days, cutoff)``; nothing at or after the cutoff can leak in.
"""

from __future__ import annotations

from datetime import date, timedelta

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

NUMERIC_FEATURES = [
    "recency_days",
    "tenure_days",
    "days_active",
    "n_sessions",
    "n_events",
    "n_views",
    "n_carts",
    "n_purchases",
    "monetary",
    "avg_order_value",
    "view_to_cart",
    "cart_to_purchase",
    "cart_abandonment_rate",
    "distinct_categories",
    "distinct_brands",
    "distinct_products",
    "avg_price_viewed",
    "max_price_viewed",
    "relative_price_level",
    "price_sensitivity",
    "weekend_share",
    "evening_share",
    "avg_session_duration_s",
    "events_per_session",
    "sessions_last_7d",
    "views_last_7d",
    "carts_last_7d",
    "purchases_last_7d",
    "activity_trend",
]
CATEGORICAL_FEATURES = ["top_category"]
SECONDS_PER_DAY = 86_400.0


def _ts(d: date) -> F.Column:
    # Parse in the session time zone (UTC); a Python datetime literal would use local time.
    return F.to_timestamp(F.lit(f"{d.isoformat()} 00:00:00"))


def _is(event_type: str) -> F.Column:
    return (F.col("event_type") == event_type).cast("int")


def _safe_div(num: str | F.Column, den: str | F.Column) -> F.Column:
    num = F.col(num) if isinstance(num, str) else num
    den = F.col(den) if isinstance(den, str) else den
    return F.when(den > 0, num / den).otherwise(F.lit(0.0))


def feature_window(events: DataFrame, cutoff: date, feature_days: int) -> DataFrame:
    start = cutoff - timedelta(days=feature_days)
    return events.where((F.col("event_time") >= _ts(start)) & (F.col("event_time") < _ts(cutoff)))


def build_user_features(
    events: DataFrame, cutoff: date, feature_days: int, snapshot: str
) -> DataFrame:
    """Compute one row of features per user active in the window before ``cutoff``."""
    cutoff_ts = _ts(cutoff)
    recent_ts = _ts(cutoff - timedelta(days=7))
    ev = feature_window(events, cutoff, feature_days).withColumn(
        "is_recent", (F.col("event_time") >= recent_ts).cast("int")
    )

    sessions = ev.groupBy("user_id", "user_session").agg(
        F.min("event_time").alias("s_start"),
        F.max("event_time").alias("s_end"),
        F.max(_is("cart")).alias("s_cart"),
        F.max(_is("purchase")).alias("s_purchase"),
    )
    sess_agg = sessions.groupBy("user_id").agg(
        F.count(F.lit(1)).alias("n_sessions"),
        F.avg(F.col("s_end").cast("long") - F.col("s_start").cast("long")).alias(
            "avg_session_duration_s"
        ),
        F.sum("s_cart").alias("cart_sessions"),
        F.sum(F.when((F.col("s_cart") == 1) & (F.col("s_purchase") == 0), 1).otherwise(0)).alias(
            "abandoned_sessions"
        ),
        F.sum((F.col("s_start") >= recent_ts).cast("int")).alias("sessions_last_7d"),
    )

    views = ev.where(F.col("event_type") == "view")
    cat_median = views.groupBy("category_code").agg(
        F.expr("percentile_approx(price, 0.5)").alias("cat_median_price")
    )
    ev = ev.join(F.broadcast(cat_median), "category_code", "left")

    is_view = F.col("event_type") == "view"
    is_cart = F.col("event_type") == "cart"
    is_purchase = F.col("event_type") == "purchase"
    agg = ev.groupBy("user_id").agg(
        F.max("event_time").alias("last_ts"),
        F.min("event_time").alias("first_ts"),
        F.countDistinct("event_date").alias("days_active"),
        F.count(F.lit(1)).alias("n_events"),
        F.sum(_is("view")).alias("n_views"),
        F.sum(_is("cart")).alias("n_carts"),
        F.sum(_is("purchase")).alias("n_purchases"),
        F.sum(F.when(is_purchase, F.col("price")).otherwise(0.0)).alias("monetary"),
        F.countDistinct("category_code").alias("distinct_categories"),
        F.countDistinct("brand").alias("distinct_brands"),
        F.countDistinct("product_id").alias("distinct_products"),
        F.avg(F.when(is_view, F.col("price"))).alias("avg_price_viewed"),
        F.max(F.when(is_view, F.col("price"))).alias("max_price_viewed"),
        F.avg(F.when(is_cart, F.col("price"))).alias("avg_price_carted"),
        F.avg(F.when(is_view, F.col("price") / F.col("cat_median_price"))).alias(
            "relative_price_level"
        ),
        F.avg(F.col("is_weekend").cast("double")).alias("weekend_share"),
        F.avg(F.col("event_hour").between(18, 23).cast("double")).alias("evening_share"),
        F.sum(F.when(is_view, F.col("is_recent")).otherwise(0)).alias("views_last_7d"),
        F.sum(F.when(is_cart, F.col("is_recent")).otherwise(0)).alias("carts_last_7d"),
        F.sum(F.when(is_purchase, F.col("is_recent")).otherwise(0)).alias("purchases_last_7d"),
    )

    top_cat = (
        views.groupBy("user_id", "category_top")
        .count()
        .withColumn(
            "rn",
            F.row_number().over(
                Window.partitionBy("user_id").orderBy(F.desc("count"), F.asc("category_top"))
            ),
        )
        .where("rn = 1")
        .select("user_id", F.col("category_top").alias("top_category"))
    )

    feats = (
        agg.join(sess_agg, "user_id", "left")
        .join(top_cat, "user_id", "left")
        .withColumn(
            "recency_days",
            (cutoff_ts.cast("long") - F.col("last_ts").cast("long")) / SECONDS_PER_DAY,
        )
        .withColumn(
            "tenure_days",
            (cutoff_ts.cast("long") - F.col("first_ts").cast("long")) / SECONDS_PER_DAY,
        )
        .withColumn("avg_order_value", _safe_div("monetary", "n_purchases"))
        .withColumn("view_to_cart", _safe_div("n_carts", "n_views"))
        .withColumn("cart_to_purchase", _safe_div("n_purchases", "n_carts"))
        .withColumn("cart_abandonment_rate", _safe_div("abandoned_sessions", "cart_sessions"))
        .withColumn(
            "price_sensitivity",
            F.when(
                (F.col("n_carts") > 0) & (F.col("avg_price_viewed") > 0),
                F.greatest(
                    F.lit(-1.0),
                    F.least(F.lit(1.0), 1 - F.col("avg_price_carted") / F.col("avg_price_viewed")),
                ),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn("events_per_session", _safe_div("n_events", "n_sessions"))
        # Sessions in the last 7 days relative to the window's average 7-day volume
        .withColumn(
            "activity_trend",
            _safe_div(F.col("sessions_last_7d") * F.lit(feature_days / 7.0), F.col("n_sessions")),
        )
        .withColumn("top_category", F.coalesce("top_category", F.lit("unknown")))
        .fillna(0.0, subset=["avg_price_viewed", "max_price_viewed", "relative_price_level"])
        .withColumn("snapshot", F.lit(snapshot))
        .withColumn("cutoff", F.to_date(F.lit(cutoff.isoformat())))
    )
    return feats.select("user_id", *NUMERIC_FEATURES, *CATEGORICAL_FEATURES, "snapshot", "cutoff")
