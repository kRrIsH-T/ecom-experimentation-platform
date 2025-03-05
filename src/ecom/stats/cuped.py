"""CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction.

``y_cuped = y - theta * (x - mean(x))`` with ``theta = cov(y, x) / var(x)`` estimated on the
pooled sample. The adjusted metric keeps the same expected treatment effect (x is
pre-treatment, hence independent of assignment) while its variance shrinks by ~corr(x, y)^2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class CupedResult:
    theta: float
    correlation: float
    var_raw: float
    var_adjusted: float
    variance_reduction: float

    def to_dict(self) -> dict:
        return asdict(self)


def cuped_theta(y: np.ndarray, x: np.ndarray) -> float:
    var_x = np.var(x, ddof=1)
    return float(np.cov(y, x, ddof=1)[0, 1] / var_x) if var_x > 0 else 0.0


def cuped_adjust(
    y: np.ndarray, x: np.ndarray, theta: float | None = None
) -> tuple[np.ndarray, CupedResult]:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    theta = cuped_theta(y, x) if theta is None else theta
    adjusted = y - theta * (x - x.mean())
    var_raw = float(np.var(y, ddof=1))
    var_adj = float(np.var(adjusted, ddof=1))
    corr = float(np.corrcoef(y, x)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0
    return adjusted, CupedResult(
        theta=theta,
        correlation=corr,
        var_raw=var_raw,
        var_adjusted=var_adj,
        variance_reduction=1 - var_adj / var_raw if var_raw > 0 else 0.0,
    )
