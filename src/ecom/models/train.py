"""Train, compare, register and batch-score the propensity models.

Models are fit on the October snapshot, tuned / early-stopped on early November and
evaluated once on late November. The champion is chosen on *validation* PR-AUC among the
servable sklearn-compatible models, registered in MLflow with the ``champion`` alias, and
used to score the held-out test users (for the A/B simulation) and the serving snapshot.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import time
from datetime import UTC, datetime
from importlib import metadata

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from ecom.config import Config
from ecom.layout import figures_dir, outputs_dir, table_path
from ecom.models.data import FEATURES, Splits, load_snapshot_features, load_splits
from ecom.models.evaluate import classification_metrics, curve_data, plot_gains, plot_model_curves
from ecom.models.explain import explain_tree_model
from ecom.models.train_baseline import train_baseline
from ecom.models.train_xgb import train_xgb

log = logging.getLogger(__name__)

SERVABLE = ("logreg", "xgboost")
DISPLAY = {
    "logreg": "Logistic regression",
    "xgboost": "XGBoost + Optuna",
    "spark_gbt": "Spark MLlib GBT",
    "h2o_automl": "H2O AutoML leader",
}


def _pip_requirements() -> list[str]:
    """Pin the libraries needed to unpickle the pipeline (explicit, so MLflow need not probe
    the environment for pip)."""
    pkgs = ("scikit-learn", "xgboost", "pandas", "numpy", "scipy", "mlflow")
    return [f"{p}=={metadata.version(p)}" for p in pkgs]


@contextlib.contextmanager
def _preserve_root_logging():
    """The first connection to a new sqlite store runs MLflow's Alembic migrations, whose
    ``fileConfig()`` swaps the root handlers and sets the root level to WARN - silently
    dropping every later INFO line of the process (all of train/abtest on a fresh run)."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    try:
        yield
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)


def setup_mlflow(cfg: Config) -> str:
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    name = cfg.get("mlflow.experiment", "ecom-propensity")
    with _preserve_root_logging():
        exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        artifacts = (cfg.root / "mlruns").as_uri()
        return mlflow.create_experiment(name, artifact_location=artifacts)
    return exp.experiment_id


def _log_metrics(prefix: str, metrics: dict) -> None:
    mlflow.log_metrics(
        {
            f"{prefix}_{k}": float(v)
            for k, v in metrics.items()
            if isinstance(v, int | float) and np.isfinite(v)
        }
    )


def _sample_tags(cfg: Config) -> dict:
    return {"data_source": cfg.source, "sample_rate": str(cfg.sample_rate)}


def _fit_sklearn(cfg: Config, splits: Splits, key: str, results: dict, curves: dict) -> None:
    Xv, yv = splits.xy("valid")
    Xte, yte = splits.xy("test")
    with mlflow.start_run(run_name=key, tags={"model_family": key, **_sample_tags(cfg)}) as run:
        t0 = time.perf_counter()
        if key == "logreg":
            raw, params = train_baseline(cfg, splits)
            study = None
        else:
            raw, params, study = train_xgb(cfg, splits)
        # Platt-scale on the validation snapshot so scores are usable as probabilities (the
        # A/B simulation and the API rely on that). Monotonic, so rankings / PR-AUC are kept.
        pipe = CalibratedClassifierCV(FrozenEstimator(raw), method="sigmoid").fit(Xv, yv)
        secs = time.perf_counter() - t0
        pv = pipe.predict_proba(Xv)[:, 1]
        pt = pipe.predict_proba(Xte)[:, 1]
        params = {**params, "calibration": "sigmoid on valid"}
        mv, mt = classification_metrics(yv, pv), classification_metrics(yte, pt)
        mlflow.log_params(params)
        _log_metrics("valid", mv)
        _log_metrics("test", mt)
        mlflow.log_metric("train_seconds", secs)
        if study is not None:
            trials = study.trials_dataframe()
            path = outputs_dir(cfg) / "optuna_trials.csv"
            trials.to_csv(path, index=False)
            mlflow.log_artifact(str(path))
        mlflow.sklearn.log_model(
            pipe,
            "model",
            signature=infer_signature(Xv, pv),
            input_example=Xv.head(3),
            pyfunc_predict_fn="predict_proba",
            pip_requirements=_pip_requirements(),
        )
        results[key] = {
            "run_id": run.info.run_id,
            "valid": mv,
            "test": mt,
            "train_seconds": secs,
            "params": params,
            "pipeline": raw,
        }
        curves[key] = curve_data(yte, pt)
        log.info(
            "%s: valid PR-AUC %.4f | test PR-AUC %.4f ROC-AUC %.4f (%.0fs)",
            key,
            mv["pr_auc"],
            mt["pr_auc"],
            mt["roc_auc"],
            secs,
        )


def _fit_spark_gbt(cfg: Config, results: dict, curves: dict) -> None:
    from ecom.models.train_spark_gbt import train_spark_gbt
    from ecom.spark import get_spark

    spark = get_spark(cfg, "ecom-pipeline")
    with mlflow.start_run(
        run_name="spark_gbt", tags={"model_family": "spark_gbt", **_sample_tags(cfg)}
    ) as run:
        res = train_spark_gbt(spark, cfg)
        mt = classification_metrics(res["y"], res["p"])
        mv = classification_metrics(res["y_valid"], res["p_valid"])
        mlflow.log_params(res["params"])
        _log_metrics("valid", mv)
        _log_metrics("test", mt)
        mlflow.log_metrics({"train_seconds": res["train_seconds"], "train_rows": res["train_rows"]})
        results["spark_gbt"] = {
            "run_id": run.info.run_id,
            "valid": mv,
            "test": mt,
            "train_seconds": res["train_seconds"],
            "params": res["params"],
        }
        curves["spark_gbt"] = curve_data(res["y"], res["p"])
        log.info(
            "spark_gbt: test PR-AUC %.4f (%.0fs on %d rows)",
            mt["pr_auc"],
            res["train_seconds"],
            res["train_rows"],
        )


def _fit_h2o(cfg: Config, splits: Splits, results: dict, curves: dict) -> None:
    from ecom.models.train_h2o import train_h2o

    with mlflow.start_run(
        run_name="h2o_automl", tags={"model_family": "h2o_automl", **_sample_tags(cfg)}
    ) as run:
        res = train_h2o(cfg, splits)
        mt = classification_metrics(res["y"], res["p"])
        lb = res["leaderboard"]
        path = outputs_dir(cfg) / "h2o_leaderboard.csv"
        lb.to_csv(path, index=False)
        mlflow.log_artifact(str(path))
        mlflow.log_params({**res["params"], "leader": res["leader"]})
        _log_metrics("test", mt)
        valid = {"pr_auc": float(lb.iloc[0]["aucpr"]), "roc_auc": float(lb.iloc[0]["auc"])}
        _log_metrics("valid", valid)
        mlflow.log_metric("train_seconds", res["train_seconds"])
        results["h2o_automl"] = {
            "run_id": run.info.run_id,
            "valid": valid,
            "test": mt,
            "train_seconds": res["train_seconds"],
            "params": {**res["params"], "leader": res["leader"]},
        }
        curves["h2o_automl"] = curve_data(res["y"], res["p"])
        log.info("h2o: leader %s test PR-AUC %.4f", res["leader"], mt["pr_auc"])


def register_champion(cfg: Config, results: dict) -> dict:
    servable = {k: v for k, v in results.items() if k in SERVABLE}
    champion = max(servable, key=lambda k: servable[k]["valid"]["pr_auc"])
    name = cfg.get("ml.registered_model", "ecom-propensity")
    mv = mlflow.register_model(f"runs:/{results[champion]['run_id']}/model", name)
    client = MlflowClient()
    client.set_registered_model_alias(name, "champion", mv.version)
    client.set_model_version_tag(name, mv.version, "model_family", champion)
    info = {
        "name": name,
        "version": int(mv.version),
        "model_family": champion,
        "run_id": results[champion]["run_id"],
        "uri": f"models:/{name}@champion",
        "registered_at": datetime.now(UTC).isoformat(),
    }
    log.info("Registered %s v%s (%s) as champion", name, mv.version, champion)
    return info


def score_snapshots(cfg: Config, champion: dict) -> pd.DataFrame:
    model = mlflow.sklearn.load_model(champion["uri"])
    frames = []
    for snapshot in ("test", "scoring"):
        feats = load_snapshot_features(cfg, snapshot)
        p = model.predict_proba(feats[FEATURES])[:, 1]
        rank = pd.Series(p).rank(ascending=False, method="first").to_numpy()
        frames.append(
            pd.DataFrame(
                {
                    "user_id": feats["user_id"].to_numpy(),
                    "snapshot": snapshot,
                    "propensity": p,
                    "decile": np.ceil(rank / len(p) * 10).astype(int),
                    "model_name": champion["name"],
                    "model_version": champion["version"],
                    "model_family": champion["model_family"],
                }
            )
        )
    scores = pd.concat(frames, ignore_index=True)
    path = table_path(cfg, "scores")
    path.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(path / "scores.parquet", index=False)
    return scores


def run_training(cfg: Config, skip: set[str] | None = None) -> dict:
    skip = skip or set()
    splits = load_splits(cfg)
    log.info(
        "train/valid/test rows: %d / %d / %d (positive rate %.3f / %.3f / %.3f)",
        len(splits.train),
        len(splits.valid),
        len(splits.test),
        splits.train.label.mean(),
        splits.valid.label.mean(),
        splits.test.label.mean(),
    )
    setup_mlflow(cfg)
    mlflow.set_experiment(cfg.get("mlflow.experiment", "ecom-propensity"))
    results: dict = {}
    curves: dict = {}
    for key in ("logreg", "xgboost"):
        _fit_sklearn(cfg, splits, key, results, curves)
    if "spark_gbt" not in skip:
        _fit_spark_gbt(cfg, results, curves)
    if "h2o" not in skip:
        _fit_h2o(cfg, splits, results, curves)

    champion = register_champion(cfg, results)
    out, figs = outputs_dir(cfg), figures_dir(cfg)
    comparison = pd.DataFrame(
        [
            {
                "model": DISPLAY[k],
                "key": k,
                "valid_pr_auc": r["valid"].get("pr_auc", np.nan),
                **{
                    f"test_{m}": r["test"][m]
                    for m in (
                        "pr_auc",
                        "roc_auc",
                        "brier",
                        "log_loss",
                        "top_decile_precision",
                        "top_decile_lift",
                        "top_decile_recall",
                        "base_rate",
                    )
                },
                "train_seconds": r["train_seconds"],
                "champion": k == champion["model_family"],
            }
            for k, r in results.items()
        ]
    ).sort_values("test_pr_auc", ascending=False)
    comparison.to_csv(out / "model_comparison.csv", index=False)
    (out / "model_curves.json").write_text(json.dumps(curves))
    (out / "champion.json").write_text(json.dumps(champion, indent=2))
    plot_model_curves(
        {DISPLAY[k]: c for k, c in curves.items()},
        splits.test.label.mean(),
        figs / "model_curves.png",
    )
    champ_gains = pd.DataFrame(curves[champion["model_family"]]["gains"])
    plot_gains(champ_gains, DISPLAY[champion["model_family"]], figs / "model_gains.png")

    expl = explain_tree_model(
        results["xgboost"]["pipeline"], splits.test[FEATURES], figs, seed=cfg.seed
    )
    expl["importance"].to_csv(out / "shap_importance.csv", index=False)
    (out / "model_drivers.json").write_text(json.dumps(expl["drivers"], indent=2))
    # reports/figures is shared by every run; keep this run's copy with its outputs
    if (shap_png := figs / "shap_summary.png").exists():
        shutil.copy2(shap_png, out / shap_png.name)
    with mlflow.start_run(run_id=results["xgboost"]["run_id"]):
        for fig in figs.glob("shap_*.png"):
            mlflow.log_artifact(str(fig), "shap")
        mlflow.log_artifact(str(figs / "model_curves.png"))

    scores = score_snapshots(cfg, champion)
    log.info("Scored %d users (%s)", len(scores), scores.groupby("snapshot").size().to_dict())
    return {"champion": champion, "comparison": comparison.to_dict(orient="records")}
