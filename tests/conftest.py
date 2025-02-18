from __future__ import annotations

from pathlib import Path

import pytest

from ecom.config import Config, load_config


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """Project config whose data/report/model paths all live under a temp directory."""
    base = load_config()
    data = base.as_dict()
    data["synthetic"]["n_users"] = 3000
    data["data"]["sample_rate"] = 1.0
    data["spark"]["hive"] = False
    cfg = Config(data, tmp_path)
    # Reuse the project's downloaded tools (JDK, winutils) instead of fetching them again.
    object.__setattr__(cfg.paths, "tools", base.paths.tools)
    cfg.paths.ensure()
    return cfg


@pytest.fixture(scope="session")
def spark():
    from ecom.spark import get_spark

    base = load_config()
    data = base.as_dict()
    data["spark"].update(
        {"master": "local[2]", "shuffle_partitions": 2, "hive": False, "driver_memory": "1g"}
    )
    session = get_spark(Config(data, base.root), "ecom-tests", hive=False)
    session.sql("CREATE DATABASE IF NOT EXISTS ecom")
    yield session
    session.stop()


def make_events(spark, rows: list[tuple]):
    """Build a clean-events DataFrame from (ts, type, product, category_code, brand, price,
    user, session) tuples, applying the real parsing and cleaning code."""
    from ecom.ingest.to_parquet import parse_raw
    from ecom.transform.clean import clean_events

    raw = spark.createDataFrame(
        [
            (ts + " UTC", et, pid, 1, cat, brand, price, uid, sess)
            for ts, et, pid, cat, brand, price, uid, sess in rows
        ],
        "event_time string, event_type string, product_id long, category_id long, "
        "category_code string, brand string, price double, user_id long, user_session string",
    )
    return clean_events(parse_raw(raw))
