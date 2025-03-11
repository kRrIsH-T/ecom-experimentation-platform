"""H2O AutoML benchmark leaderboard (sorted by AUCPR on the validation snapshot)."""

from __future__ import annotations

import logging
import time
import warnings

import pandas as pd

from ecom.config import Config
from ecom.env import local_dir
from ecom.models.data import FEATURES, TARGET, Splits
from ecom.tools import ensure_runtime

log = logging.getLogger(__name__)


def _upload(h2o, df: pd.DataFrame, name: str):
    """Stream a frame to the H2O server with upload_file.

    `H2OFrame(pandas_df)` makes the *server* read a temp CSV from disk, which H2O's default
    file deny-list rejects for paths under dot-directories such as `.cache/`. Uploading
    sends the bytes over HTTP, so the server never touches our filesystem.
    """
    path = local_dir("tmp") / f"h2o_{name}.csv"
    df.to_csv(path, index=False)
    try:
        types = {TARGET: "enum", "top_category": "enum"}
        return h2o.upload_file(str(path), col_types=types, destination_frame=f"ecom_{name}")
    finally:
        path.unlink(missing_ok=True)


def _download(frame) -> pd.DataFrame:
    """H2O frame -> pandas over HTTP. The multi-threaded path makes the server write a temp
    CSV, which the same deny-list blocks, and the frames here are small anyway."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Converting H2O frame to pandas")
        return frame.as_data_frame(use_multi_thread=False)


def train_h2o(cfg: Config, splits: Splits) -> dict:
    ensure_runtime(cfg)
    import h2o
    from h2o.automl import H2OAutoML

    h2o.init(
        nthreads=-1,
        max_mem_size=cfg.get("ml.h2o_max_mem", "4G"),
        ice_root=str(local_dir("h2o")),
        log_dir=str(local_dir("h2o") / "logs"),
        verbose=False,
    )
    h2o.no_progress()
    try:
        frames = {}
        for name in ("train", "valid", "test"):
            frames[name] = _upload(h2o, getattr(splits, name)[[*FEATURES, TARGET]], name)
        runtime = int(cfg.get("ml.h2o_max_runtime_s", 300))
        aml = H2OAutoML(
            max_runtime_secs=runtime,
            seed=cfg.seed,
            sort_metric="AUCPR",
            balance_classes=False,
            exclude_algos=["DeepLearning"],  # slow and rarely competitive on tabular data
            project_name="ecom_propensity",
        )
        t0 = time.perf_counter()
        aml.train(
            x=FEATURES, y=TARGET, training_frame=frames["train"], leaderboard_frame=frames["valid"]
        )
        train_s = time.perf_counter() - t0
        lb = _download(aml.leaderboard)
        pred = _download(aml.leader.predict(frames["test"]))
        p = pred["p1"].to_numpy(dtype=float)
        return {
            "leaderboard": lb,
            "leader": str(aml.leader.model_id),
            "y": splits.test[TARGET].to_numpy(),
            "p": p,
            "train_seconds": train_s,
            "params": {
                "max_runtime_secs": runtime,
                "sort_metric": "AUCPR",
                "models_trained": len(lb),
            },
        }
    finally:
        h2o.cluster().shutdown()
