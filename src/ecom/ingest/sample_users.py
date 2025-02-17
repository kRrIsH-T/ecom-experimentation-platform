"""Deterministic user-level sampling of bronze events.

Sampling whole users (not rows) keeps every user's full history intact, so sessions,
features and labels computed on the sample are unbiased per-user statistics.
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ecom.config import Config
from ecom.layout import bronze_full_path, table_path

log = logging.getLogger(__name__)

BUCKETS = 10_000


def user_bucket(col: str = "user_id", seed: int = 42):
    """Stable bucket in [0, BUCKETS) from a seeded 64-bit hash of the user id."""
    return F.pmod(F.xxhash64(F.col(col), F.lit(seed)), F.lit(BUCKETS))


def sample_events(df: DataFrame, rate: float, seed: int) -> DataFrame:
    if rate >= 1.0:
        return df
    return df.where(user_bucket("user_id", seed) < F.lit(round(rate * BUCKETS)))


def write_sample(spark: SparkSession, cfg: Config) -> str:
    """Materialise the working ``events`` table for the configured sample rate."""
    target = table_path(cfg, "events")
    if cfg.sample_rate >= 1.0:
        log.info("sample_rate=1.0 - using the full bronze table directly")
        return str(target)
    full = spark.read.parquet(str(bronze_full_path(cfg)))
    sampled = sample_events(full, cfg.sample_rate, cfg.seed)
    sampled.repartition("event_date").write.mode("overwrite").partitionBy("event_date").parquet(
        str(target)
    )
    log.info("Wrote %.2f%% user sample to %s", cfg.sample_rate * 100, target)
    return str(target)
