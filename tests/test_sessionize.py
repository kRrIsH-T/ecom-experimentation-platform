import pytest

from ecom.transform.sessionize import sessionize
from tests.conftest import make_events

pytestmark = pytest.mark.spark


def test_session_aggregates(spark):
    rows = [
        ("2019-10-05 10:00:00", "view", 1, "electronics.smartphone", "apple", 100.0, 7, "s1"),
        ("2019-10-05 10:01:00", "view", 2, "electronics.smartphone", "apple", 300.0, 7, "s1"),
        ("2019-10-05 10:02:00", "cart", 1, "electronics.smartphone", "apple", 100.0, 7, "s1"),
        ("2019-10-05 10:05:00", "purchase", 1, "electronics.smartphone", "apple", 100.0, 7, "s1"),
        ("2019-10-05 10:06:00", "view", 3, "kids.toys", "lego", 20.0, 7, "s1"),
        ("2019-10-07 09:00:00", "view", 3, "kids.toys", "lego", 20.0, 8, "s2"),
        ("2019-10-07 09:00:00", "view", 3, "kids.toys", "lego", 20.0, 8, "s2"),  # duplicate
    ]
    s = {r.user_session: r for r in sessionize(make_events(spark, rows)).collect()}
    s1, s2 = s["s1"], s["s2"]
    assert s1.n_events == 5 and s1.n_views == 3 and s1.n_carts == 1 and s1.n_purchases == 1
    assert s1.duration_s == 360
    assert s1.revenue == pytest.approx(100.0)
    assert s1.purchased and s1.carted
    assert s1.main_category == "electronics"
    assert s1.is_weekend  # 2019-10-05 is a Saturday
    assert s1.avg_view_price == pytest.approx(140.0)
    assert s2.n_events == 1, "exact duplicates are removed before sessionising"
    assert not s2.purchased and s2.duration_s == 0
    assert s2.dow == 1  # Monday


def test_main_category_tie_breaks_alphabetically(spark):
    rows = [
        ("2019-10-01 10:00:00", "view", 1, "kids.toys", "lego", 20.0, 1, "s"),
        ("2019-10-01 10:01:00", "view", 2, "apparel.shoes", "nike", 50.0, 1, "s"),
    ]
    (row,) = sessionize(make_events(spark, rows)).collect()
    assert row.main_category == "apparel"
