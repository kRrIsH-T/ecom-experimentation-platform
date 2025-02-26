from datetime import date

import pytest

from ecom.transform.build_features import NUMERIC_FEATURES, build_user_features
from tests.conftest import make_events

pytestmark = pytest.mark.spark

CUTOFF = date(2019, 10, 22)

ROWS = [
    # user 1: two sessions, carts the cheaper phone, buys once, abandons one cart
    ("2019-10-10 10:00:00", "view", 1, "electronics.smartphone", "apple", 400.0, 1, "a"),
    ("2019-10-10 10:01:00", "view", 2, "electronics.smartphone", "xiaomi", 200.0, 1, "a"),
    ("2019-10-10 10:02:00", "cart", 2, "electronics.smartphone", "xiaomi", 200.0, 1, "a"),
    ("2019-10-10 10:03:00", "purchase", 2, "electronics.smartphone", "xiaomi", 200.0, 1, "a"),
    ("2019-10-19 20:00:00", "view", 3, "kids.toys", "lego", 50.0, 1, "b"),
    ("2019-10-19 20:10:00", "cart", 3, "kids.toys", "lego", 50.0, 1, "b"),
    # user 2: a single view on a weekday morning
    ("2019-10-02 09:00:00", "view", 4, "kids.toys", "lego", 30.0, 2, "c"),
    # outside the window: before the start and at/after the cutoff
    ("2019-09-30 23:59:59", "view", 5, "kids.toys", "lego", 30.0, 2, "old"),
    ("2019-10-22 00:00:00", "purchase", 3, "kids.toys", "lego", 50.0, 1, "future"),
]


@pytest.fixture
def feats(spark):
    df = build_user_features(make_events(spark, ROWS), CUTOFF, 21, "train")
    return {r.user_id: r for r in df.collect()}


def test_schema_and_population(feats):
    assert set(feats) == {1, 2}
    row = feats[1]
    for col in NUMERIC_FEATURES:
        assert getattr(row, col) is not None, col


def test_user_one_values(feats):
    u = feats[1]
    assert u.n_sessions == 2
    assert u.n_views == 3 and u.n_carts == 2 and u.n_purchases == 1
    assert u.monetary == pytest.approx(200.0), "the purchase at the cutoff must not count"
    assert u.avg_order_value == pytest.approx(200.0)
    assert u.cart_abandonment_rate == pytest.approx(0.5)
    assert u.recency_days == pytest.approx((2 * 24 * 3600 + 3 * 3600 + 50 * 60) / 86400)
    assert u.days_active == 2
    assert u.sessions_last_7d == 1 and u.views_last_7d == 1 and u.carts_last_7d == 1
    assert u.purchases_last_7d == 0
    # carted prices average 125 vs viewed 216.67 -> price-sensitive
    assert u.price_sensitivity == pytest.approx(1 - 125 / (650 / 3))
    assert u.top_category == "electronics"
    assert u.distinct_categories == 2
    assert u.evening_share == pytest.approx(2 / 6)
    assert u.weekend_share == pytest.approx(2 / 6)  # 2019-10-19 is a Saturday


def test_user_two_values(feats):
    u = feats[2]
    assert u.n_views == 1 and u.n_carts == 0
    assert u.view_to_cart == 0.0 and u.cart_abandonment_rate == 0.0
    assert u.price_sensitivity == 0.0
    # kids.toys views are priced {30, 50}; percentile_approx returns an observed value (30)
    assert u.relative_price_level == pytest.approx(30 / 30)
