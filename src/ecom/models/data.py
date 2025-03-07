"""Model dataset loading and the shared feature preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ecom.config import Config
from ecom.layout import table_path
from ecom.transform.build_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "label"


@dataclass
class Splits:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    def xy(self, name: str) -> tuple[pd.DataFrame, np.ndarray]:
        df = getattr(self, name)
        return df[FEATURES], df[TARGET].to_numpy()


def _read_parquet(path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    """All numeric features as float64 (counts included) so the logged model schema accepts
    missing values, and the category as a plain string."""
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].astype("float64")
    df["top_category"] = df["top_category"].astype(str)
    return df


def load_splits(cfg: Config) -> Splits:
    df = _typed(_read_parquet(table_path(cfg, "model_dataset")))
    return Splits(
        *(df[df["split"] == s].reset_index(drop=True) for s in ("train", "valid", "test"))
    )


def load_snapshot_features(cfg: Config, snapshot: str) -> pd.DataFrame:
    df = _read_parquet(table_path(cfg, "user_features") / f"snapshot={snapshot}")
    df["snapshot"] = snapshot
    return _typed(df)


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """log1p that keeps the sign; tames the heavy right tails of count and money features."""
    return np.sign(x) * np.log1p(np.abs(x))


def make_preprocessor(cfg: Config, scale: bool) -> ColumnTransformer:
    """Numeric features (+ log/scale for linear models) and a one-hot top category."""
    numeric_steps: list = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps += [
            ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    max_categories=int(cfg.get("ml.top_categories", 12)),
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )
