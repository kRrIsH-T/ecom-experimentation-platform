"""Label construction: did the user purchase in the ``label_days`` after the cutoff?

The population is every user with at least one event in the feature window, so the label
window is strictly after everything the features can see (no leakage by construction).
"""

from __future__ import annotations

from datetime import date, timedelta

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def _ts(d: date) -> F.Column:
    return F.to_timestamp(F.lit(f"{d.isoformat()} 00:00:00"))


def label_window(events: DataFrame, cutoff: date, label_days: int) -> DataFrame:
    end = cutoff + timedelta(days=label_days)
    return events.where((F.col("event_time") >= _ts(cutoff)) & (F.col("event_time") < _ts(end)))


def build_labels(events: DataFrame, cutoff: date, label_days: int, snapshot: str) -> DataFrame:
    """Per-user purchase outcome inside ``[cutoff, cutoff + label_days)``."""
    purchases = label_window(events, cutoff, label_days).where(F.col("event_type") == "purchase")
    return (
        purchases.groupBy("user_id")
        .agg(
            F.lit(1).alias("label"),
            F.sum("price").alias("label_revenue"),
            F.count(F.lit(1)).alias("label_purchases"),
            F.min("event_time").alias("first_purchase_time"),
        )
        .withColumn("snapshot", F.lit(snapshot))
    )


def attach_labels(features: DataFrame, labels: DataFrame) -> DataFrame:
    """Left-join labels onto the feature population; users without purchases get label 0."""
    return (
        features.join(labels.drop("snapshot"), "user_id", "left")
        .withColumn("label", F.coalesce("label", F.lit(0)).cast("int"))
        .withColumn("label_revenue", F.coalesce("label_revenue", F.lit(0.0)))
        .withColumn("label_purchases", F.coalesce("label_purchases", F.lit(0)))
    )
