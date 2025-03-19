"""Upsert one document per scored user into MongoDB.

Document shape::

    {_id, user_id, features{...}, propensity, decile, propensity_tier, segment,
     rfm{recency_days, frequency, monetary, code}, model{name, version, family},
     snapshot_cutoff, load_id, updated_at}
"""

from __future__ import annotations

import logging
import math
import uuid
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from pymongo import ASCENDING, DESCENDING, MongoClient, UpdateOne
from pymongo.collection import Collection

from ecom.config import Config
from ecom.layout import sample_tag, table_path
from ecom.models.data import load_snapshot_features
from ecom.transform.build_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

log = logging.getLogger(__name__)

PROSPECT_SEGMENT = "prospects"  # users who have never purchased


def tier(decile: int) -> str:
    return "high" if decile <= 2 else "medium" if decile <= 5 else "low"


def get_collection(cfg: Config, client: MongoClient | None = None) -> Collection:
    client = client or MongoClient(cfg.mongo_uri, serverSelectionTimeoutMS=5000, tz_aware=True)
    return client[cfg.get("mongo.db", "ecom")][cfg.get("mongo.collection", "users")]


def ensure_indexes(col: Collection) -> None:
    col.create_index([("user_id", ASCENDING)], unique=True, name="user_id_unique")
    col.create_index(
        [("segment", ASCENDING), ("propensity", DESCENDING)], name="segment_propensity"
    )
    col.create_index([("decile", ASCENDING)], name="decile")


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def build_documents(cfg: Config) -> pd.DataFrame:
    feats = load_snapshot_features(cfg, "scoring")
    scores = pd.read_parquet(table_path(cfg, "scores"))
    scores = scores[scores["snapshot"] == "scoring"].drop(columns="snapshot")
    rfm_path = cfg.paths.gold / cfg.source / sample_tag(cfg.sample_rate) / "rfm_users"
    rfm = pd.read_parquet(rfm_path)[
        ["user_id", "recency_days", "frequency", "monetary", "rfm_code", "segment"]
    ]
    rfm = rfm.rename(
        columns={
            "monetary": "rfm_monetary",
            "recency_days": "rfm_recency_days",
            "frequency": "rfm_frequency",
        }
    )
    df = feats.merge(scores, on="user_id", how="inner").merge(rfm, on="user_id", how="left")
    df["segment"] = df["segment"].fillna(PROSPECT_SEGMENT)
    return df


def to_documents(df: pd.DataFrame, load_id: str, now: datetime) -> list[dict[str, Any]]:
    docs = []
    for row in df.to_dict(orient="records"):
        uid = int(row["user_id"])
        has_rfm = isinstance(row.get("rfm_code"), str)
        docs.append(
            {
                "_id": uid,
                "user_id": uid,
                "features": {k: _clean(row[k]) for k in NUMERIC_FEATURES + CATEGORICAL_FEATURES},
                "propensity": float(row["propensity"]),
                "decile": int(row["decile"]),
                "propensity_tier": tier(int(row["decile"])),
                "segment": row["segment"],
                "rfm": {
                    "recency_days": _clean(row["rfm_recency_days"]),
                    "frequency": _clean(row["rfm_frequency"]),
                    "monetary": _clean(row["rfm_monetary"]),
                    "code": row["rfm_code"],
                }
                if has_rfm
                else None,
                "model": {
                    "name": row["model_name"],
                    "version": int(row["model_version"]),
                    "family": row["model_family"],
                },
                "snapshot_cutoff": str(row["cutoff"]),
                "load_id": load_id,
                "updated_at": now,
            }
        )
    return docs


def load_users(cfg: Config, col: Collection | None = None, batch_size: int = 5000) -> dict:
    col = col if col is not None else get_collection(cfg)
    df = build_documents(cfg)
    load_id = uuid.uuid4().hex
    now = datetime.now(UTC)
    docs = to_documents(df, load_id, now)
    ensure_indexes(col)
    upserted = modified = 0
    for i in range(0, len(docs), batch_size):
        ops = [
            UpdateOne({"_id": d["_id"]}, {"$set": d}, upsert=True) for d in docs[i : i + batch_size]
        ]
        res = col.bulk_write(ops, ordered=False)
        upserted += res.upserted_count
        modified += res.modified_count
    removed = col.delete_many({"load_id": {"$ne": load_id}}).deleted_count
    stats = {
        "documents": len(docs),
        "upserted": upserted,
        "modified": modified,
        "removed_stale": removed,
        "load_id": load_id,
    }
    log.info("MongoDB load: %s", stats)
    return stats
