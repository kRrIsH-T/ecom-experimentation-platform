"""Integration test: the real Spark steps on a tiny synthetic dataset."""

import pandas as pd
import pytest

from ecom.ingest.synthetic import generate
from ecom.layout import outputs_dir, table_path
from ecom.pipeline import step_features, step_ingest, step_sql, step_transform
from ecom.sql_runner import all_queries

pytestmark = [pytest.mark.spark, pytest.mark.slow]


def test_sql_files_parse():
    queries = all_queries()
    names = [q.name for q in queries]
    assert len(names) == len(set(names))
    assert {"funnel_overall", "cohort_retention", "rfm_users", "category_stats"} <= set(names)
    assert [q.name for q in queries if q.materialize] == ["rfm_users"]


def test_pipeline_end_to_end(spark, cfg):
    generate(cfg, n_users=3000)
    dq = step_ingest(spark, cfg)
    assert dq["duplicate_rows"] > 0 and dq["zero_prices"] >= 0
    assert (cfg.paths.reports / "data_quality.md").exists()

    step_transform(spark, cfg)
    clean = spark.table("ecom.events_clean")
    assert clean.count() == clean.distinct().count(), "silver has no duplicates"
    assert clean.where("price <= 0").count() == 0

    stats = step_features(spark, cfg)
    assert {"train", "valid", "test", "scoring"} <= set(stats)
    ds = pd.read_parquet(table_path(cfg, "model_dataset"))
    assert set(ds["split"]) == {"train", "valid", "test"}
    assert not ds.duplicated(["split", "user_id"]).any()

    step_sql(spark, cfg)
    out = outputs_dir(cfg)
    funnel = pd.read_csv(out / "funnel_overall.csv")
    assert funnel["users"].is_monotonic_decreasing
    assert funnel["stage"].tolist() == ["view", "cart", "purchase"]
    sessions = pd.read_csv(out / "funnel_sessions.csv")
    assert sessions["sessions"].is_monotonic_decreasing
    cohorts = pd.read_csv(out / "cohort_retention.csv")
    assert (cohorts.loc[cohorts.week_number == 0, "retention"] == 1.0).all()
    rfm = pd.read_csv(out / "rfm_summary.csv")
    assert rfm["user_share"].sum() == pytest.approx(1.0)
    cats = pd.read_csv(out / "category_stats.csv")
    assert cats["cumulative_share"].iloc[-1] == pytest.approx(1.0)
