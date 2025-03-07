"""Classification metrics, calibration and lift/gains for imbalanced propensity models."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from ecom.viz import BLUE, MUTED


def gains_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Decile table sorted by score (decile 1 = highest propensity)."""
    df = pd.DataFrame({"y": y, "p": p}).sort_values("p", ascending=False, kind="stable")
    df["decile"] = (np.arange(len(df)) * n_bins // len(df)) + 1
    g = df.groupby("decile").agg(
        users=("y", "size"), positives=("y", "sum"), avg_score=("p", "mean")
    )
    base = df["y"].mean()
    g["positive_rate"] = g["positives"] / g["users"]
    g["lift"] = g["positive_rate"] / base if base > 0 else np.nan
    g["cumulative_gain"] = g["positives"].cumsum() / max(df["y"].sum(), 1)
    g["cumulative_users"] = g["users"].cumsum() / len(df)
    return g.reset_index()


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    g = gains_table(y, p)
    top = g.iloc[0]
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "base_rate": float(y.mean()),
        "top_decile_precision": float(top["positive_rate"]),
        "top_decile_lift": float(top["lift"]),
        "top_decile_recall": float(top["cumulative_gain"]),
        "n": len(y),
    }


def curve_data(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    """Down-sampled ROC / PR / calibration curves (JSON-friendly, for the dashboard)."""
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")

    def thin(a: np.ndarray, b: np.ndarray, k: int = 200):
        idx = np.unique(np.linspace(0, len(a) - 1, min(k, len(a))).astype(int))
        return a[idx].tolist(), b[idx].tolist()

    fpr_s, tpr_s = thin(fpr, tpr)
    rec_s, prec_s = thin(rec, prec)
    return {
        "roc": {"fpr": fpr_s, "tpr": tpr_s},
        "pr": {"recall": rec_s, "precision": prec_s},
        "calibration": {
            "mean_predicted": mean_pred.tolist(),
            "fraction_positive": frac_pos.tolist(),
        },
        "gains": gains_table(y, p).to_dict(orient="list"),
    }


def plot_model_curves(curves: dict[str, dict], base_rate: float, path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for name, c in curves.items():
        axes[0].plot(c["roc"]["fpr"], c["roc"]["tpr"], label=name)
        axes[1].plot(c["pr"]["recall"], c["pr"]["precision"], label=name)
        axes[2].plot(
            c["calibration"]["mean_predicted"],
            c["calibration"]["fraction_positive"],
            marker="o",
            label=name,
        )
    axes[0].plot([0, 1], [0, 1], "--", color=MUTED, lw=1)
    axes[0].set(title="ROC (test)", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].axhline(base_rate, color=MUTED, ls="--", lw=1, label=f"base rate {base_rate:.3f}")
    axes[1].set(title="Precision-recall (test)", xlabel="Recall", ylabel="Precision")
    lim = max(
        max(c["calibration"]["mean_predicted"] + c["calibration"]["fraction_positive"])
        for c in curves.values()
    )
    axes[2].plot([0, lim], [0, lim], "--", color=MUTED, lw=1)
    axes[2].set(
        title="Calibration (test, quantile bins)",
        xlabel="Mean predicted probability",
        ylabel="Observed purchase rate",
    )
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_gains(gains: pd.DataFrame, model_name: str, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(gains["decile"], gains["lift"], color=BLUE)
    axes[0].axhline(1, color=MUTED, ls="--", lw=1)
    axes[0].set(
        title=f"Lift by decile - {model_name}",
        xlabel="Score decile (1 = highest)",
        ylabel="Lift vs base rate",
        xticks=gains["decile"],
    )
    axes[1].plot(
        [0, *gains["cumulative_users"]],
        [0, *gains["cumulative_gain"]],
        marker="o",
        label=model_name,
    )
    axes[1].plot([0, 1], [0, 1], "--", color=MUTED, lw=1, label="random")
    axes[1].set(
        title="Cumulative gains",
        xlabel="Share of users contacted",
        ylabel="Share of next-week buyers captured",
    )
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
