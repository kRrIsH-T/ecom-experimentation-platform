"""Silver ``sessions``: one row per (user_id, user_session)."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def _is(event_type: str):
    return (F.col("event_type") == event_type).cast("int")


def sessionize(events: DataFrame) -> DataFrame:
    """Aggregate clean events into sessions.

    Revenue is the sum of purchase-event prices. ``main_category`` is the most frequent
    top-level category in the session (ties broken alphabetically for determinism).
    """
    agg = events.groupBy("user_id", "user_session").agg(
        F.min("event_time").alias("session_start"),
        F.max("event_time").alias("session_end"),
        F.count(F.lit(1)).alias("n_events"),
        F.sum(_is("view")).alias("n_views"),
        F.sum(_is("cart")).alias("n_carts"),
        F.sum(_is("purchase")).alias("n_purchases"),
        F.sum(F.when(F.col("event_type") == "purchase", F.col("price")).otherwise(0.0)).alias(
            "revenue"
        ),
        F.countDistinct("product_id").alias("n_products"),
        F.countDistinct("category_top").alias("n_categories"),
        F.avg(F.when(F.col("event_type") == "view", F.col("price"))).alias("avg_view_price"),
    )
    cat_counts = events.groupBy("user_session", "category_top").count()
    top_cat = (
        cat_counts.withColumn(
            "rn",
            F.row_number().over(
                Window.partitionBy("user_session").orderBy(F.desc("count"), F.asc("category_top"))
            ),
        )
        .where("rn = 1")
        .select("user_session", F.col("category_top").alias("main_category"))
    )
    return (
        agg.join(top_cat, "user_session", "left")
        .withColumn(
            "duration_s",
            F.col("session_end").cast("long") - F.col("session_start").cast("long"),
        )
        .withColumn("purchased", F.col("n_purchases") > 0)
        .withColumn("carted", F.col("n_carts") > 0)
        .withColumn("session_date", F.to_date("session_start"))
        .withColumn("start_hour", F.hour("session_start"))
        .withColumn("dow", ((F.dayofweek("session_start") + 5) % 7) + 1)
        .withColumn("is_weekend", F.col("dow") >= 6)
    )
