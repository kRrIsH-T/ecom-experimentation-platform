"""Baseline: class-weighted logistic regression on log-scaled, standardised features."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ecom.config import Config
from ecom.models.data import Splits, make_preprocessor


def train_baseline(cfg: Config, splits: Splits) -> tuple[Pipeline, dict]:
    X, y = splits.xy("train")
    params = {"C": 1.0, "class_weight": "balanced", "max_iter": 3000}
    model = Pipeline(
        [
            ("prep", make_preprocessor(cfg, scale=True)),
            ("model", LogisticRegression(**params, random_state=cfg.seed)),
        ]
    )
    model.fit(X, y)
    return model, params
