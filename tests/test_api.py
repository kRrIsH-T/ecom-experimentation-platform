from datetime import UTC, datetime

import mongomock
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ecom.serving import api
from ecom.serving.load_mongo import ensure_indexes, tier, to_documents
from ecom.transform.build_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


class FakeModel:
    """Deterministic stand-in for the champion pipeline: p = sigmoid(0.1 * n_views - 3)."""

    def predict_proba(self, X: pd.DataFrame):
        p = 1 / (1 + np.exp(-(0.1 * X["n_views"].to_numpy(dtype=float) - 3)))
        return np.column_stack([1 - p, p])


def _rows(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.uniform(0, 10, n) for c in NUMERIC_FEATURES})
    df["n_views"] = np.arange(n, dtype=float)
    df["top_category"] = "electronics"
    df["user_id"] = np.arange(100, 100 + n)
    df["propensity"] = FakeModel().predict_proba(df)[:, 1]
    df["decile"] = np.ceil(df["propensity"].rank(ascending=False) / n * 10).astype(int)
    df["segment"] = np.where(df["user_id"] % 2 == 0, "champions", "prospects")
    df["rfm_code"] = np.where(df["segment"] == "champions", "555", None)
    df["rfm_recency_days"] = 3.0
    df["rfm_frequency"] = 4
    df["rfm_monetary"] = 900.0
    df["model_name"] = "ecom-propensity"
    df["model_version"] = 3
    df["model_family"] = "xgboost"
    df["cutoff"] = "2019-12-01"
    return df


@pytest.fixture
def client():
    col = mongomock.MongoClient(tz_aware=True).db.users
    ensure_indexes(col)
    col.insert_many(to_documents(_rows(), "load-1", datetime(2026, 1, 1, tzinfo=UTC)))
    model = api.ChampionModel(
        FakeModel(), api.ModelInfo(name="ecom-propensity", version=3, family="xgboost")
    )
    api.app.dependency_overrides[api.collection_dep] = lambda: col
    api.app.dependency_overrides[api.model_dep] = lambda: model
    original = api.model_dep
    api.model_dep = lambda: model  # used directly by ?live=true
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()
    api.model_dep = original


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["users"] == 30


def test_user_score_and_live_match(client):
    r = client.get("/users/120/score", params={"live": True})
    assert r.status_code == 200
    body = r.json()
    assert body["user_id"] == 120
    assert body["segment"] == "champions"
    assert body["model"]["version"] == 3
    assert body["live_matches_stored"] is True
    assert body["propensity"] == pytest.approx(body["live_propensity"])
    assert body["propensity_tier"] == tier(body["decile"])


def test_unknown_user_404(client):
    assert client.get("/users/1/score").status_code == 404


def test_segment_top_sorted(client):
    r = client.get("/segments/champions/top", params={"k": 5})
    assert r.status_code == 200
    users = r.json()["users"]
    assert len(users) == 5
    props = [u["propensity"] for u in users]
    assert props == sorted(props, reverse=True)
    assert all(u["user_id"] % 2 == 0 for u in users)
    assert client.get("/segments/nope/top").status_code == 404
    assert client.get("/segments/champions/top", params={"k": 0}).status_code == 422


def test_segments_counts(client):
    rows = {r["segment"]: r["users"] for r in client.get("/segments").json()}
    assert rows == {"champions": 15, "prospects": 15}


def test_live_score_endpoint(client):
    values = dict.fromkeys(NUMERIC_FEATURES, 1.0) | dict.fromkeys(CATEGORICAL_FEATURES, "kids")
    values["n_views"] = 30.0
    r = client.post("/score", json={"values": values})
    assert r.status_code == 200
    assert r.json()["propensity"] == pytest.approx(0.5)
    bad = client.post("/score", json={"values": {"bogus": 1}})
    assert bad.status_code == 422


def test_documents_shape():
    doc = to_documents(_rows(2), "x", datetime(2026, 1, 1, tzinfo=UTC))[0]
    assert set(doc["features"]) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    assert doc["_id"] == doc["user_id"]
    assert doc["rfm"]["code"] == "555"
