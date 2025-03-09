"""XGBoost tuned with Optuna (TPE) on validation PR-AUC, early-stopped on validation log-loss."""

from __future__ import annotations

import logging
import subprocess

import optuna
import xgboost
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from ecom.config import Config
from ecom.models.data import Splits, make_preprocessor

log = logging.getLogger(__name__)


def resolve_device(setting: str) -> str:
    """``cpu`` | ``cuda`` | ``auto`` (``cuda`` when this XGBoost build has CUDA and a GPU is
    visible to ``nvidia-smi``)."""
    if setting != "auto":
        return setting
    if not xgboost.build_info().get("USE_CUDA"):
        return "cpu"
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return "cpu"
    return "cuda" if out.returncode == 0 and "GPU" in out.stdout else "cpu"


def _model(
    params: dict, seed: int, n_estimators: int, early_stopping: int | None, device: str = "cpu"
):
    # Unweighted log-loss keeps probabilities on the true scale and gives early stopping a
    # smooth signal (validation PR-AUC is too noisy and stops after a handful of trees).
    return XGBClassifier(
        **params,
        n_estimators=n_estimators,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=device,
        early_stopping_rounds=early_stopping,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )


def train_xgb(cfg: Config, splits: Splits) -> tuple[Pipeline, dict, optuna.Study]:
    prep = make_preprocessor(cfg, scale=False)
    X_train, y_train = splits.xy("train")
    X_valid, y_valid = splits.xy("valid")
    Xt = prep.fit_transform(X_train)
    Xv = prep.transform(X_valid)
    device = resolve_device(str(cfg.get("ml.xgb_device", "cpu")))
    log.info("XGBoost device: %s", device)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "gamma": trial.suggest_float("gamma", 1e-3, 5, log=True),
        }
        model = _model(params, cfg.seed, 3000, 100, device)
        model.fit(Xt, y_train, eval_set=[(Xv, y_valid)], verbose=False)
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        model.set_params(device="cpu")  # host arrays: predict on CPU, no device-mismatch copy
        return float(average_precision_score(y_valid, model.predict_proba(Xv)[:, 1]))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.seed)
    )
    study.optimize(
        objective,
        n_trials=int(cfg.get("ml.optuna_trials", 30)),
        timeout=int(cfg.get("ml.optuna_timeout_s", 600)),
    )
    best = dict(study.best_params)
    n_estimators = int(study.best_trial.user_attrs["best_iteration"]) + 1
    log.info(
        "Optuna: %d trials, best valid PR-AUC %.4f with %s (n_estimators=%d)",
        len(study.trials),
        study.best_value,
        best,
        n_estimators,
    )
    final = _model(best, cfg.seed, n_estimators, None, device)
    final.fit(Xt, y_train, verbose=False)
    # The registered model must load and predict on CPU-only machines (API, dashboard).
    final.set_params(device="cpu")
    pipe = Pipeline([("prep", prep), ("model", final)])
    params = {
        **best,
        "n_estimators": n_estimators,
        "optuna_trials": len(study.trials),
        "train_device": device,
    }
    return pipe, params, study
