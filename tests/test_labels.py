"""Leakage tests: features must not see the label window and labels must not see the past."""

from datetime import date

import pytest
from pyspark.sql import functions as F

from ecom.transform.build_features import build_user_features
from ecom.transform.labels import attach_labels, build_labels
from tests.conftest import make_events

pytestmark = pytest.mark.spark

CUTOFF = date(2019, 10, 22)

ROWS = [
    ("2019-10-15 12:00:00", "view", 1, "kids.toys", "lego", 50.0, 1, "a"),
    ("2019-10-21 23:59:59", "purchase", 1, "kids.toys", "lego", 50.0, 1, "a"),  # before cutoff
    ("2019-10-22 00:00:00", "purchase", 1, "kids.toys", "lego", 70.0, 1, "b"),  # first label sec
    ("2019-10-28 23:59:59", "purchase", 1, "kids.toys", "lego", 5.0, 1, "c"),  # last label sec
    ("2019-10-29 00:00:00", "purchase", 1, "kids.toys", "lego", 999.0, 1, "d"),  # after window
    ("2019-10-16 12:00:00", "view", 2, "kids.toys", "lego", 50.0, 2, "e"),
    ("2019-10-30 12:00:00", "purchase", 2, "kids.toys", "lego", 50.0, 2, "f"),  # after window
    ("2019-10-23 12:00:00", "purchase", 3, "kids.toys", "lego", 50.0, 3, "g"),  # no history
]


def test_label_window_boundaries(spark):
    labels = {
        r.user_id: r for r in build_labels(make_events(spark, ROWS), CUTOFF, 7, "t").collect()
    }
    assert set(labels) == {1, 3}
    assert labels[1].label_revenue == pytest.approx(75.0)
    assert labels[1].label_purchases == 2


def test_population_comes_from_feature_window(spark):
    events = make_events(spark, ROWS)
    feats = build_user_features(events, CUTOFF, 21, "t")
    ds = {
        r.user_id: r for r in attach_labels(feats, build_labels(events, CUTOFF, 7, "t")).collect()
    }
    assert set(ds) == {1, 2}, "user 3 has no pre-cutoff activity so cannot be scored"
    assert ds[1].label == 1 and ds[2].label == 0


def test_features_ignore_everything_after_cutoff(spark):
    """Features computed on the full history equal features computed on truncated history."""
    events = make_events(spark, ROWS)
    truncated = events.where(F.col("event_time") < F.lit("2019-10-22 00:00:00").cast("timestamp"))
    full = build_user_features(events, CUTOFF, 21, "t").orderBy("user_id").collect()
    trunc = build_user_features(truncated, CUTOFF, 21, "t").orderBy("user_id").collect()
    assert full == trunc
    assert {r.user_id: r.monetary for r in full}[1] == pytest.approx(50.0)
