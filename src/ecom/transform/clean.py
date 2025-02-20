"""Bronze events -> silver ``events_clean`` (deduplicated, valid, enriched)."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ecom.ingest.to_parquet import EVENT_TYPES


def clean_events(df: DataFrame) -> DataFrame:
    """Drop duplicates and invalid rows; add category_top, hour, day-of-week, weekend flag."""
    return (
        df.dropDuplicates()
        .where(F.col("event_time").isNotNull())
        .where(F.col("user_id").isNotNull() & F.col("user_session").isNotNull())
        .where(F.col("event_type").isin(*EVENT_TYPES))
        .where(F.col("price") > 0)
        .withColumn(
            "category_top",
            F.coalesce(F.split(F.col("category_code"), r"\.").getItem(0), F.lit("unknown")),
        )
        .withColumn("category_code", F.coalesce("category_code", F.lit("unknown")))
        .withColumn("brand", F.coalesce("brand", F.lit("unknown")))
        .withColumn("event_hour", F.hour("event_time"))
        # dayofweek: 1=Sunday..7=Saturday -> ISO 1=Monday..7=Sunday
        .withColumn("dow", ((F.dayofweek("event_time") + 5) % 7) + 1)
        .withColumn("is_weekend", F.col("dow") >= 6)
    )
