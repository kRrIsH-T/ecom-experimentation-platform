"""Spark pipeline steps: ingest -> transform (silver) -> features/labels (gold) -> SQL."""

from __future__ import annotations

import json
import logging
import time
from datetime import date

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from ecom.config import Config
from ecom.ingest.sample_users import write_sample
from ecom.ingest.to_parquet import convert
from ecom.layout import bronze_full_path, outputs_dir, source_outputs_dir, table_path
from ecom.spark import register_table
from ecom.sql_runner import run_queries
from ecom.transform.build_features import build_user_features
from ecom.transform.clean import clean_events
from ecom.transform.labels import attach_labels, build_labels
from ecom.transform.sessionize import sessionize

log = logging.getLogger(__name__)


def snapshots(cfg: Config) -> dict[str, date]:
    """Snapshot name -> cutoff date (train/valid/test are labelled, scoring is not)."""
    splits = {k: date.fromisoformat(v) for k, v in cfg.get("labels.splits").items()}
    splits["scoring"] = date.fromisoformat(cfg.get("labels.scoring_cutoff"))
    return splits


def register_events(spark: SparkSession, cfg: Config) -> None:
    register_table(spark, "events_full", str(bronze_full_path(cfg)), partitioned=True)
    register_table(spark, "events", str(table_path(cfg, "events")), partitioned=True)


def step_ingest(spark: SparkSession, cfg: Config, force: bool = False) -> dict:
    """Raw CSV -> bronze (full) -> user sample; registers ``ecom.events[_full]``."""
    t0 = time.perf_counter()
    marker = bronze_full_path(cfg) / "_SUCCESS"
    if force or not marker.exists():
        dq = convert(spark, cfg)
    else:
        log.info("Bronze already present at %s (use --force to rebuild)", marker.parent)
        dq_path = source_outputs_dir(cfg) / "data_quality.json"
        dq = json.loads(dq_path.read_text()) if dq_path.exists() else {}
    write_sample(spark, cfg)
    register_events(spark, cfg)
    log.info("ingest finished in %.1fs", time.perf_counter() - t0)
    return dq


def step_transform(spark: SparkSession, cfg: Config) -> None:
    """Bronze events -> silver ``events_clean`` + ``sessions``."""
    events = spark.table("ecom.events")
    clean_path = table_path(cfg, "events_clean")
    clean_events(events).repartition("event_date").write.mode("overwrite").partitionBy(
        "event_date"
    ).parquet(str(clean_path))
    register_table(spark, "events_clean", str(clean_path), partitioned=True)

    sessions_path = table_path(cfg, "sessions")
    sessionize(spark.table("ecom.events_clean")).write.mode("overwrite").parquet(str(sessions_path))
    register_table(spark, "sessions", str(sessions_path))


def step_features(spark: SparkSession, cfg: Config) -> dict[str, dict]:
    """Gold ``user_features`` (all snapshots), ``labels`` and the labelled ``model_dataset``."""
    events = spark.table("ecom.events_clean")
    events.cache()
    feature_days = int(cfg.get("labels.feature_days"))
    label_days = int(cfg.get("labels.label_days"))
    feats_path = table_path(cfg, "user_features")
    labels_path = table_path(cfg, "labels")
    stats: dict[str, dict] = {}
    for i, (name, cutoff) in enumerate(snapshots(cfg).items()):
        mode = "overwrite" if i == 0 else "append"
        feats = build_user_features(events, cutoff, feature_days, name)
        feats.write.mode(mode).partitionBy("snapshot").parquet(str(feats_path))
        if name != "scoring":
            build_labels(events, cutoff, label_days, name).write.mode(mode).partitionBy(
                "snapshot"
            ).parquet(str(labels_path))
    events.unpersist()
    register_table(spark, "user_features", str(feats_path), partitioned=True)
    register_table(spark, "labels", str(labels_path), partitioned=True)

    feats = spark.table("ecom.user_features").where(F.col("snapshot") != "scoring")
    labels = spark.table("ecom.labels")
    dataset = None
    for name in cfg.get("labels.splits"):
        part = attach_labels(
            feats.where(F.col("snapshot") == name), labels.where(F.col("snapshot") == name)
        ).withColumn("split", F.lit(name))
        dataset = part if dataset is None else dataset.unionByName(part)
    ds_path = table_path(cfg, "model_dataset")
    dataset.write.mode("overwrite").parquet(str(ds_path))
    register_table(spark, "model_dataset", str(ds_path))
    dupes = spark.table("ecom.model_dataset").groupBy("split", "user_id").count().where("count > 1")
    if dupes.limit(1).count():
        raise RuntimeError("model_dataset has duplicate (split, user_id) rows - stale partitions?")
    for row in (
        spark.table("ecom.model_dataset")
        .groupBy("split")
        .agg(F.count(F.lit(1)).alias("users"), F.avg("label").alias("positive_rate"))
        .collect()
    ):
        stats[row["split"]] = {"users": row["users"], "positive_rate": row["positive_rate"]}
    stats["scoring"] = {
        "users": spark.table("ecom.user_features").where(F.col("snapshot") == "scoring").count()
    }
    (outputs_dir(cfg) / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    log.info("model dataset: %s", stats)
    return stats


def step_sql(spark: SparkSession, cfg: Config) -> dict:
    return run_queries(spark, cfg)


def ensure_tables(spark: SparkSession, cfg: Config) -> None:
    """Re-register every existing table (the metastore may have been reset)."""
    register_events(spark, cfg)
    for name, partitioned in [
        ("events_clean", True),
        ("sessions", False),
        ("user_features", True),
        ("labels", True),
        ("model_dataset", False),
    ]:
        path = table_path(cfg, name)
        if path.exists():
            register_table(spark, name, str(path), partitioned=partitioned)
