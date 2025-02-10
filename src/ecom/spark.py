"""SparkSession factory shared by every pipeline step."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ecom.config import Config
from ecom.env import local_dir
from ecom.tools import ensure_runtime

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

log = logging.getLogger(__name__)

HIVE_DB = "ecom"


def get_spark(cfg: Config, app_name: str = "ecom", hive: bool | None = None) -> SparkSession:
    """Create (or reuse) a local SparkSession.

    With ``hive=True`` the session uses an embedded Derby Hive metastore under
    ``data/metastore`` so table definitions survive across pipeline steps.
    """
    ensure_runtime(cfg)
    from pyspark.sql import SparkSession

    use_hive = cfg.get("spark.hive", True) if hive is None else hive
    paths = cfg.paths
    tmp = local_dir("tmp").as_posix()
    java_opts = f"-Duser.timezone=UTC -Djava.io.tmpdir={tmp}"
    if use_hive:
        paths.metastore.mkdir(parents=True, exist_ok=True)
        java_opts += f" -Dderby.system.home={paths.metastore.as_posix()}"
    builder = (
        SparkSession.builder.appName(app_name)
        .master(cfg.get("spark.master", "local[*]"))
        .config("spark.driver.memory", cfg.get("spark.driver_memory", "4g"))
        .config("spark.sql.shuffle.partitions", str(cfg.get("spark.shuffle_partitions", 16)))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.adaptive.enabled", "true")
        # Standard UTC-adjusted microsecond timestamps (not legacy INT96) so pandas/pyarrow
        # and DuckDB read the same instants Spark wrote.
        .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.extraJavaOptions", java_opts)
        .config("spark.executor.extraJavaOptions", java_opts)
        # Shuffle/spill go to SPARK_LOCAL_DIRS and scratch space to <root>/.cache (ecom.env)
        .config("spark.jars.ivy", local_dir("ivy2").as_posix())
        .config("spark.sql.warehouse.dir", paths.warehouse.as_uri())
    )
    if use_hive:
        db_path = (paths.metastore / "metastore_db").as_posix()
        builder = (
            builder.config(
                "spark.hadoop.javax.jdo.option.ConnectionURL",
                f"jdbc:derby:;databaseName={db_path};create=true",
            )
            .config("spark.hadoop.hive.exec.scratchdir", local_dir("hive/scratch").as_uri())
            .config(
                "spark.hadoop.hive.exec.local.scratchdir",
                local_dir("hive/local-scratch").as_posix(),
            )
            .config(
                "spark.hadoop.hive.downloaded.resources.dir",
                local_dir("hive/resources").as_posix() + "/${hive.session.id}_resources",
            )
            .config("spark.hadoop.hive.querylog.location", local_dir("hive/scratch").as_posix())
            .enableHiveSupport()
        )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    if use_hive:
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {HIVE_DB}")
    return spark


def register_table(spark: SparkSession, name: str, path: str, partitioned: bool = False) -> None:
    """(Re)register an external Parquet table in the ``ecom`` Hive database."""
    from pathlib import Path

    location = Path(path).resolve().as_uri()
    spark.sql(f"DROP TABLE IF EXISTS {HIVE_DB}.{name}")
    spark.catalog.createTable(f"{HIVE_DB}.{name}", path=location, source="parquet")
    if partitioned:
        spark.sql(f"MSCK REPAIR TABLE {HIVE_DB}.{name}")
    log.info("Registered %s.%s -> %s", HIVE_DB, name, location)


def read_table(spark: SparkSession, cfg: Config, name: str):
    """Read an ``ecom`` table from Hive when available, else straight from its Parquet path."""
    if spark.catalog.tableExists(name, HIVE_DB):
        return spark.table(f"{HIVE_DB}.{name}")
    from ecom.layout import table_path

    return spark.read.parquet(str(table_path(cfg, name)))
