"""Run the named queries in ``src/ecom/sql/*.sql`` against the Hive tables.

Each file holds one or more blocks introduced by ``-- name: <query_name>``. A trailing
``[table]`` tag materialises the result as a gold Parquet table registered in Hive (used by
later queries and by the Mongo loader); other results are small and saved as CSV for the
dashboard and reports.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import SparkSession

from ecom.config import Config
from ecom.layout import outputs_dir, sample_tag
from ecom.spark import register_table

log = logging.getLogger(__name__)

SQL_DIR = Path(__file__).parent / "sql"
SQL_FILES = ["kpis.sql", "funnel.sql", "cohorts.sql", "category_stats.sql", "rfm.sql"]
_NAME_RE = re.compile(r"^--\s*name:\s*(\w+)\s*(\[table\])?\s*$", re.MULTILINE)


@dataclass(frozen=True)
class NamedQuery:
    name: str
    sql: str
    materialize: bool
    source: str


def parse_sql_file(path: Path) -> list[NamedQuery]:
    text = path.read_text(encoding="utf-8")
    matches = list(_NAME_RE.finditer(text))
    queries = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() : end].strip().rstrip(";").strip()
        queries.append(NamedQuery(m.group(1), body, bool(m.group(2)), path.name))
    return queries


def all_queries(files: list[str] | None = None) -> list[NamedQuery]:
    return [q for f in files or SQL_FILES for q in parse_sql_file(SQL_DIR / f)]


def run_queries(spark: SparkSession, cfg: Config, files: list[str] | None = None) -> dict:
    out = outputs_dir(cfg)
    results = {}
    for q in all_queries(files):
        df = spark.sql(q.sql)
        if q.materialize:
            path = cfg.paths.gold / cfg.source / sample_tag(cfg.sample_rate) / q.name
            df.write.mode("overwrite").parquet(str(path))
            register_table(spark, q.name, str(path))
            results[q.name] = str(path)
            log.info("[%s] %s -> table %s", q.source, q.name, path)
        else:
            pdf = df.toPandas()
            pdf.to_csv(out / f"{q.name}.csv", index=False)
            results[q.name] = len(pdf)
            log.info("[%s] %s -> %d rows", q.source, q.name, len(pdf))
    return results
