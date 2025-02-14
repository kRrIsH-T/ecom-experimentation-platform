"""Raw CSV -> typed bronze Parquet partitioned by event_date, plus data-quality checks."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ecom.config import Config
from ecom.ingest.load_raw import RAW_SCHEMA_DDL, raw_files
from ecom.layout import bronze_full_path, source_outputs_dir

log = logging.getLogger(__name__)

EVENT_TYPES = ("view", "cart", "purchase")


def read_raw(spark: SparkSession, paths: list[str]) -> DataFrame:
    """Read Kaggle-format CSVs with an explicit schema and cast to analysis types."""
    raw = (
        spark.read.option("header", True)
        .option("mode", "PERMISSIVE")
        .schema(RAW_SCHEMA_DDL)
        .csv(paths)
    )
    return parse_raw(raw)


def parse_raw(raw: DataFrame) -> DataFrame:
    ts = F.to_timestamp(F.regexp_replace("event_time", r" UTC$", ""), "yyyy-MM-dd HH:mm:ss")
    return raw.select(
        ts.alias("event_time"),
        F.lower(F.trim("event_type")).alias("event_type"),
        F.col("product_id"),
        F.col("category_id"),
        F.when(F.trim("category_code") == "", None)
        .otherwise(F.col("category_code"))
        .alias("category_code"),
        F.when(F.trim("brand") == "", None).otherwise(F.lower("brand")).alias("brand"),
        F.col("price").cast("double").alias("price"),
        F.col("user_id"),
        F.when(F.trim("user_session") == "", None)
        .otherwise(F.col("user_session"))
        .alias("user_session"),
    ).withColumn("event_date", F.to_date("event_time"))


def data_quality(df: DataFrame) -> dict:
    """Null rates, duplicates, invalid prices, event mix and coverage of a bronze frame."""
    total = df.count()
    null_exprs = [F.sum(F.col(c).isNull().cast("long")).alias(c) for c in df.columns]
    nulls = df.agg(*null_exprs).first().asDict()
    distinct_rows = df.distinct().count()
    price = df.agg(
        F.sum((F.col("price") < 0).cast("long")).alias("negative"),
        F.sum((F.col("price") == 0).cast("long")).alias("zero"),
        F.expr("percentile_approx(price, array(0.01, 0.5, 0.99))").alias("pcts"),
        F.max("price").alias("max"),
    ).first()
    mix = {r["event_type"]: r["count"] for r in df.groupBy("event_type").count().collect()}
    # Format inside Spark: collect() would convert timestamps to the machine's local zone.
    span = df.agg(
        F.date_format(F.min("event_time"), "yyyy-MM-dd HH:mm:ss 'UTC'").alias("min_ts"),
        F.date_format(F.max("event_time"), "yyyy-MM-dd HH:mm:ss 'UTC'").alias("max_ts"),
        F.countDistinct("event_date").alias("days"),
        F.approx_count_distinct("user_id").alias("users"),
        F.approx_count_distinct("user_session").alias("sessions"),
        F.approx_count_distinct("product_id").alias("products"),
    ).first()
    unknown_types = sum(v for k, v in mix.items() if k not in EVENT_TYPES)
    return {
        "rows": total,
        "null_rate": {k: (v or 0) / total for k, v in nulls.items()},
        "duplicate_rows": total - distinct_rows,
        "duplicate_rate": (total - distinct_rows) / total,
        "negative_prices": int(price["negative"] or 0),
        "zero_prices": int(price["zero"] or 0),
        "price_p01_p50_p99": [float(x) for x in price["pcts"]],
        "price_max": float(price["max"]),
        "event_type_counts": mix,
        "unknown_event_types": unknown_types,
        "min_event_time": str(span["min_ts"]),
        "max_event_time": str(span["max_ts"]),
        "days": int(span["days"]),
        "approx_users": int(span["users"]),
        "approx_sessions": int(span["sessions"]),
        "approx_products": int(span["products"]),
    }


def render_quality_report(dq: dict, source: str) -> str:
    rows = dq["rows"]
    lines = [
        "# Data quality report",
        "",
        f"_Source: **{source}** · generated {datetime.now(UTC):%Y-%m-%d %H:%M UTC} by "
        "`ecom ingest`_",
        "",
        f"- Rows: **{rows:,}** over **{dq['days']}** days "
        f"({dq['min_event_time']} → {dq['max_event_time']})",
        f"- Approx. distinct users / sessions / products: {dq['approx_users']:,} / "
        f"{dq['approx_sessions']:,} / {dq['approx_products']:,}",
        f"- Exact duplicate rows: **{dq['duplicate_rows']:,}** ({dq['duplicate_rate']:.3%}) - "
        "dropped in silver",
        f"- Negative prices: **{dq['negative_prices']:,}**; zero prices: **{dq['zero_prices']:,}** "
        "- dropped in silver",
        "- Price p1 / p50 / p99 / max: "
        + " / ".join(f"{p:,.2f}" for p in dq["price_p01_p50_p99"])
        + f" / {dq['price_max']:,.2f}",
        f"- Unknown event types: {dq['unknown_event_types']:,}",
        "",
        "## Event type distribution",
        "",
        "| event_type | rows | share |",
        "|---|---:|---:|",
    ]
    for k, v in sorted(dq["event_type_counts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v:,} | {v / rows:.2%} |")
    lines += ["", "## Null rates", "", "| column | null rate |", "|---|---:|"]
    for k, v in dq["null_rate"].items():
        lines.append(f"| {k} | {v:.3%} |")
    lines += [
        "",
        "Missing `category_code` and `brand` are expected in this dataset (many products are "
        "uncategorised); they are kept and mapped to `unknown`. Rows missing `user_session` "
        "cannot be sessionised and are dropped in silver.",
        "",
    ]
    return "\n".join(lines)


def convert(spark: SparkSession, cfg: Config) -> dict:
    """Convert all raw CSVs for the active source into bronze ``events_full``."""
    files = [str(p) for p in raw_files(cfg)]
    log.info("Reading %d raw file(s) from %s", len(files), Path(files[0]).parent)
    df = read_raw(spark, files)
    target = bronze_full_path(cfg)
    df.repartition("event_date").write.mode("overwrite").partitionBy("event_date").parquet(
        str(target)
    )
    bronze = spark.read.parquet(str(target))
    dq = data_quality(bronze)
    (source_outputs_dir(cfg) / "data_quality.json").write_text(
        json.dumps(dq, indent=2, default=str)
    )
    cfg.paths.reports.mkdir(parents=True, exist_ok=True)
    (cfg.paths.reports / "data_quality.md").write_text(
        render_quality_report(dq, cfg.source), encoding="utf-8"
    )
    log.info("Bronze written to %s (%s rows)", target, f"{dq['rows']:,}")
    return dq
