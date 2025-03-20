"""FastAPI scoring service backed by MongoDB and the MLflow champion model."""

from __future__ import annotations

import logging
import math
import threading
import time
from datetime import datetime
from functools import lru_cache
from typing import Annotated, Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from ecom.config import get_config
from ecom.serving.load_mongo import get_collection, tier
from ecom.transform.build_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

log = logging.getLogger(__name__)

app = FastAPI(
    title="E-commerce propensity API",
    version="1.0.0",
    description="Next-7-day purchase propensity scores, RFM segments and live scoring.",
)


class ModelInfo(BaseModel):
    name: str
    version: int
    family: str


class UserScore(BaseModel):
    user_id: int
    propensity: float = Field(description="P(purchase in the next 7 days)")
    decile: int = Field(description="1 = highest propensity")
    propensity_tier: str
    segment: str
    model: ModelInfo
    snapshot_cutoff: str
    updated_at: datetime
    live_propensity: float | None = Field(
        None, description="Score recomputed now from the stored features (?live=true)"
    )
    live_matches_stored: bool | None = None


class SegmentUser(BaseModel):
    user_id: int
    propensity: float
    decile: int


class SegmentTop(BaseModel):
    segment: str
    k: int
    users: list[SegmentUser]


class SegmentCount(BaseModel):
    segment: str
    users: int
    avg_propensity: float


class Features(BaseModel):
    """Raw behavioural features; missing numeric values are imputed by the model pipeline."""

    model_config = {"extra": "forbid"}
    values: dict[str, float | str | None] = Field(
        description=f"Keys: {', '.join(NUMERIC_FEATURES + CATEGORICAL_FEATURES)}"
    )


class LiveScore(BaseModel):
    propensity: float
    propensity_tier_estimate: str
    model: ModelInfo


# --- dependencies -------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _collection() -> Collection:
    return get_collection(get_config())


def collection_dep() -> Collection:
    return _collection()


class ChampionModel:
    def __init__(self, model: Any, info: ModelInfo):
        self.model = model
        self.info = info

    def predict(self, rows: list[dict[str, Any]]) -> list[float]:
        X = pd.DataFrame(rows)[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].astype(float)
        X["top_category"] = X["top_category"].fillna("unknown").astype(str)
        return self.model.predict_proba(X)[:, 1].tolist()


CHAMPION_RECHECK_S = 60.0
_champion_cache: dict[str, Any] = {"model": None, "checked_at": 0.0}
_champion_lock = threading.Lock()


def _champion() -> ChampionModel:
    """The registered champion, re-checking the registry alias at most once a minute.

    A retrain moves the ``champion`` alias to a new version; reloading only when the
    version changes keeps requests fast while never serving a stale model for long.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    with _champion_lock:
        cached: ChampionModel | None = _champion_cache["model"]
        if cached is not None and time.monotonic() - _champion_cache["checked_at"] < (
            CHAMPION_RECHECK_S
        ):
            return cached
        cfg = get_config()
        mlflow.set_tracking_uri(cfg.mlflow_uri)
        name = cfg.get("ml.registered_model", "ecom-propensity")
        mv = MlflowClient().get_model_version_by_alias(name, "champion")
        if cached is None or cached.info.version != int(mv.version):
            model = mlflow.sklearn.load_model(f"models:/{name}/{mv.version}")
            family = mv.tags.get("model_family", "unknown")
            cached = ChampionModel(
                model, ModelInfo(name=name, version=int(mv.version), family=family)
            )
            log.info("Loaded champion %s v%s", name, mv.version)
        _champion_cache.update(model=cached, checked_at=time.monotonic())
        return cached


def model_dep() -> ChampionModel:
    try:
        return _champion()
    except Exception as exc:  # registry missing or model not trained yet
        raise HTTPException(503, f"champion model unavailable: {exc}") from exc


CollectionDep = Annotated[Collection, Depends(collection_dep)]
ModelDep = Annotated[ChampionModel, Depends(model_dep)]


# --- routes -------------------------------------------------------------------------------


@app.get("/health")
def health(col: CollectionDep) -> dict[str, Any]:
    try:
        col.database.client.admin.command("ping")
        users = col.estimated_document_count()
        mongo_ok = True
    except PyMongoError as exc:
        log.warning("mongo health check failed: %s", exc)
        users, mongo_ok = 0, False
    return {"status": "ok" if mongo_ok else "degraded", "mongo": mongo_ok, "users": users}


@app.get("/users/{user_id}/score", response_model=UserScore)
def user_score(
    user_id: int,
    col: CollectionDep,
    live: Annotated[bool, Query(description="also rescore with the champion model")] = False,
) -> UserScore:
    doc = col.find_one({"_id": user_id})
    if doc is None:
        raise HTTPException(404, f"user {user_id} not found")
    out = UserScore(**{k: doc[k] for k in UserScore.model_fields if k in doc})
    if live:
        model = model_dep()
        p = model.predict([doc["features"]])[0]
        out.live_propensity = p
        out.live_matches_stored = math.isclose(p, doc["propensity"], rel_tol=1e-6, abs_tol=1e-9)
    return out


@app.get("/segments", response_model=list[SegmentCount])
def segments(col: CollectionDep) -> list[SegmentCount]:
    rows = col.aggregate(
        [
            {
                "$group": {
                    "_id": "$segment",
                    "users": {"$sum": 1},
                    "avg_propensity": {"$avg": "$propensity"},
                }
            },
            {"$sort": {"users": -1}},
        ]
    )
    return [
        SegmentCount(segment=r["_id"], users=r["users"], avg_propensity=r["avg_propensity"])
        for r in rows
    ]


@app.get("/segments/{name}/top", response_model=SegmentTop)
def segment_top(
    name: str, col: CollectionDep, k: Annotated[int, Query(ge=1, le=1000)] = 10
) -> SegmentTop:
    cursor = (
        col.find({"segment": name}, {"user_id": 1, "propensity": 1, "decile": 1})
        .sort("propensity", -1)
        .limit(k)
    )
    users = [
        SegmentUser(user_id=d["user_id"], propensity=d["propensity"], decile=d["decile"])
        for d in cursor
    ]
    if not users and col.count_documents({"segment": name}, limit=1) == 0:
        raise HTTPException(404, f"segment '{name}' not found")
    return SegmentTop(segment=name, k=k, users=users)


@app.post("/score", response_model=LiveScore)
def score(features: Features, model: ModelDep) -> LiveScore:
    unknown = set(features.values) - set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    if unknown:
        raise HTTPException(422, f"unknown features: {sorted(unknown)}")
    row = {k: features.values.get(k) for k in NUMERIC_FEATURES + CATEGORICAL_FEATURES}
    p = model.predict([row])[0]
    # Tier from the propensity distribution is only available for loaded users; estimate
    # it from fixed thresholds instead.
    est = "high" if p >= 0.15 else "medium" if p >= 0.05 else "low"
    return LiveScore(propensity=p, propensity_tier_estimate=est, model=model.info)


__all__ = ["app", "collection_dep", "model_dep", "tier"]
