"""Score a saved model on the active source's validation and test snapshots.

Answers "does a model trained elsewhere transfer?" - e.g. the champion trained on the
synthetic data, scored on the real Kaggle snapshots (`ecom kaggle push` runs this remotely).
The model directory is an MLflow sklearn model; an optional ``meta.json`` next to it
describes where the model came from.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlflow

from ecom.config import Config
from ecom.layout import outputs_dir
from ecom.models.data import load_splits
from ecom.models.evaluate import classification_metrics

log = logging.getLogger(__name__)

RESULT_FILE = "external_model_eval.json"


def evaluate_saved_model(cfg: Config, model_dir: Path, name: str) -> dict:
    model = mlflow.sklearn.load_model(str(model_dir))
    meta_path = model_dir.parent / "meta.json"
    splits = load_splits(cfg)
    result: dict = {
        "name": name,
        "model_dir": str(model_dir),
        "meta": json.loads(meta_path.read_text()) if meta_path.exists() else {},
        "evaluated_on": {"source": cfg.source, "sample_rate": cfg.sample_rate},
    }
    for split in ("valid", "test"):
        X, y = splits.xy(split)
        result[split] = classification_metrics(y, model.predict_proba(X)[:, 1])
    (outputs_dir(cfg) / RESULT_FILE).write_text(json.dumps(result, indent=2))
    log.info(
        "%s on %s: valid PR-AUC %.4f | test PR-AUC %.4f ROC-AUC %.4f (base rate %.4f)",
        name,
        cfg.source,
        result["valid"]["pr_auc"],
        result["test"]["pr_auc"],
        result["test"]["roc_auc"],
        result["test"]["base_rate"],
    )
    return result
