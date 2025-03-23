"""Same aggregation in pandas, DuckDB and Spark on the full bronze events.

Query: per (event_date, top-level category) -> events, purchases, revenue, distinct users.
Wall time and peak resident memory of the whole process tree (Python + the Spark JVM) are
sampled every 50 ms. Each engine runs once to warm up, then ``repeats`` timed runs.
"""

from __future__ import annotations

import contextlib
import json
import logging
import statistics
import threading
import time
from collections.abc import Callable

import psutil

from ecom.config import Config
from ecom.layout import bronze_full_path, source_outputs_dir

log = logging.getLogger(__name__)

COLUMNS = ["event_date", "event_type", "category_code", "price", "user_id"]


class PeakRSS:
    """Context manager sampling the RSS of this process and all its children."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()

    def _rss(self) -> int:
        proc = psutil.Process()
        total = proc.memory_info().rss
        for child in proc.children(recursive=True):
            with contextlib.suppress(psutil.Error):
                total += child.memory_info().rss
        return total

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._rss())
            time.sleep(self.interval)

    def __enter__(self) -> PeakRSS:
        self.baseline = self._rss()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


def _pandas(path: str) -> int:
    import pandas as pd

    df = pd.read_parquet(path, columns=COLUMNS)
    df["category_top"] = df["category_code"].str.split(".", n=1).str[0].fillna("unknown")
    df["is_purchase"] = df["event_type"] == "purchase"
    df["revenue"] = df["price"].where(df["is_purchase"], 0.0)
    out = df.groupby(["event_date", "category_top"], observed=True).agg(
        events=("event_type", "size"),
        purchases=("is_purchase", "sum"),
        revenue=("revenue", "sum"),
        users=("user_id", "nunique"),
    )
    return len(out)


def _duckdb(path: str) -> int:
    import duckdb

    con = duckdb.connect()
    q = f"""
        SELECT event_date, COALESCE(split_part(category_code, '.', 1), 'unknown') AS category_top,
               COUNT(*) AS events, SUM((event_type = 'purchase')::INT) AS purchases,
               SUM(CASE WHEN event_type = 'purchase' THEN price ELSE 0 END) AS revenue,
               COUNT(DISTINCT user_id) AS users
        FROM read_parquet('{path}/**/*.parquet', hive_partitioning = true)
        GROUP BY 1, 2"""
    return len(con.execute(q).fetchall())


def _spark_runner(cfg: Config) -> Callable[[str], int]:
    from pyspark.sql import functions as F

    from ecom.spark import get_spark

    spark = get_spark(cfg, "ecom-pipeline")

    def run(path: str) -> int:
        df = spark.read.parquet(path)
        out = (
            df.withColumn(
                "category_top",
                F.coalesce(F.split("category_code", r"\.").getItem(0), F.lit("unknown")),
            )
            .groupBy("event_date", "category_top")
            .agg(
                F.count(F.lit(1)).alias("events"),
                F.sum((F.col("event_type") == "purchase").cast("int")).alias("purchases"),
                F.sum(
                    F.when(F.col("event_type") == "purchase", F.col("price")).otherwise(0.0)
                ).alias("revenue"),
                F.countDistinct("user_id").alias("users"),
            )
        )
        return len(out.collect())

    return run


def run_benchmark(cfg: Config, repeats: int = 2) -> dict:
    import pyarrow.parquet as pq

    path = str(bronze_full_path(cfg))
    rows = sum(f.metadata.num_rows for f in pq.ParquetDataset(path).fragments)
    engines: dict[str, Callable[[str], int]] = {
        "pandas": _pandas,
        "duckdb": _duckdb,
        "spark": _spark_runner(cfg),
    }
    results = {}
    for name, fn in engines.items():
        fn(path)  # warm-up (file cache, JIT, Spark planning)
        times, peaks = [], []
        groups = 0
        for _ in range(repeats):
            with PeakRSS() as mem:
                t0 = time.perf_counter()
                groups = fn(path)
                times.append(time.perf_counter() - t0)
            peaks.append(mem.peak - mem.baseline)
        results[name] = {
            "seconds_median": statistics.median(times),
            "seconds_runs": times,
            "peak_extra_rss_mb": max(peaks) / 2**20,
            "groups": groups,
        }
        log.info(
            "benchmark %-6s %.2fs, +%.0f MB peak RSS, %d groups",
            name,
            results[name]["seconds_median"],
            results[name]["peak_extra_rss_mb"],
            groups,
        )
    payload = {
        "rows": int(rows),
        "query": __doc__.split("\n\n")[1].strip(),
        "machine": {"cpus": psutil.cpu_count(), "ram_gb": psutil.virtual_memory().total / 2**30},
        "results": results,
    }
    (source_outputs_dir(cfg) / "benchmark.json").write_text(json.dumps(payload, indent=2))
    return payload
