"""Frequentist, CUPED, Bayesian and sequential (peeking) analysis of the A/B test."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ecom.stats.core import benjamini_hochberg, two_proportion_ztest, welch_ttest
from ecom.stats.cuped import cuped_adjust


def _arms(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        df.loc[df.group == "control", col].to_numpy(dtype=float),
        df.loc[df.group == "treatment", col].to_numpy(dtype=float),
    )


def conversion_test(df: pd.DataFrame, alpha: float) -> dict[str, Any]:
    c, t = _arms(df, "converted")
    res = two_proportion_ztest(
        int(c.sum()), len(c), int(t.sum()), len(t), alpha, name="Conversion (primary)"
    )
    return res.to_dict()


def revenue_tests(df: pd.DataFrame, alpha: float) -> dict[str, Any]:
    """Revenue per user: raw Welch t-test vs CUPED-adjusted with pre-period spend."""
    raw = welch_ttest(*_arms(df, "revenue"), alpha=alpha, name="Revenue per user (raw)")
    adjusted, cuped = cuped_adjust(
        df["revenue"].to_numpy(dtype=float), df["pre_spend"].to_numpy(dtype=float)
    )
    tmp = df.assign(revenue_cuped=adjusted)
    adj = welch_ttest(
        *_arms(tmp, "revenue_cuped"), alpha=alpha, name="Revenue per user (CUPED, pre-period spend)"
    )
    raw_w, adj_w = raw.ci_high - raw.ci_low, adj.ci_high - adj.ci_low
    return {
        "raw": raw.to_dict(),
        "cuped": adj.to_dict(),
        "cuped_stats": cuped.to_dict(),
        "ci_width_raw": raw_w,
        "ci_width_cuped": adj_w,
        "ci_width_reduction": 1 - adj_w / raw_w if raw_w > 0 else 0.0,
    }


def conversion_cupac(df: pd.DataFrame, alpha: float) -> dict[str, Any]:
    """CUPED on conversion using the pre-period model score as covariate (a.k.a. CUPAC)."""
    adjusted, cuped = cuped_adjust(
        df["converted"].to_numpy(dtype=float), df["propensity"].to_numpy(dtype=float)
    )
    tmp = df.assign(conv_adj=adjusted)
    res = welch_ttest(
        *_arms(tmp, "conv_adj"), alpha=alpha, name="Conversion (CUPAC, propensity covariate)"
    )
    return {"test": res.to_dict(), "cuped_stats": cuped.to_dict()}


def aov_guardrail(df: pd.DataFrame, alpha: float) -> dict[str, Any]:
    buyers = df[df.converted == 1].assign(aov=lambda d: d.revenue / d.orders.clip(lower=1))
    c, t = _arms(buyers, "aov")
    res = welch_ttest(c, t, alpha=alpha, name="Average order value of buyers (guardrail)")
    mw = stats.mannwhitneyu(t, c, alternative="two-sided")
    out = res.to_dict()
    out["mann_whitney_p"] = float(mw.pvalue)
    return out


def segment_breakdown(df: pd.DataFrame, alpha: float) -> list[dict[str, Any]]:
    segs = {
        "top decile (targeted)": df.decile == 1,
        "deciles 2-5": df.decile.between(2, 5),
        "deciles 6-10": df.decile >= 6,
    }
    for cat in df["top_category"].value_counts().head(4).index:
        segs[f"category: {cat}"] = df.top_category == cat
    rows = []
    for name, mask in segs.items():
        sub = df[mask]
        c, t = _arms(sub, "converted")
        if len(c) < 30 or len(t) < 30 or c.sum() == 0 or t.sum() == 0:
            continue
        r = two_proportion_ztest(int(c.sum()), len(c), int(t.sum()), len(t), alpha, name=name)
        rows.append(
            {
                "segment": name,
                "n_control": len(c),
                "n_treatment": len(t),
                "conv_control": r.details["p_a"],
                "conv_treatment": r.details["p_b"],
                "diff": r.details["diff"],
                "ci_low": r.ci_low,
                "ci_high": r.ci_high,
                "relative_lift": r.details["relative_lift"],
                "p_value": r.p_value,
            }
        )
    adj = benjamini_hochberg([r["p_value"] for r in rows])
    for r, p in zip(rows, adj, strict=True):
        r["p_bh"] = float(p)
    return rows


def bayesian_conversion(
    df: pd.DataFrame, n_draws: int = 200_000, seed: int = 0, prior: tuple[float, float] = (1.0, 1.0)
) -> dict[str, Any]:
    """Beta-Binomial posterior for each arm: P(T > C), lift credible interval, expected loss."""
    rng = np.random.default_rng(seed)
    c, t = _arms(df, "converted")
    a0, b0 = prior
    pc = rng.beta(a0 + c.sum(), b0 + len(c) - c.sum(), n_draws)
    pt = rng.beta(a0 + t.sum(), b0 + len(t) - t.sum(), n_draws)
    lift = pt / pc - 1
    return {
        "prior": list(prior),
        "posterior_mean_control": float(pc.mean()),
        "posterior_mean_treatment": float(pt.mean()),
        "prob_treatment_better": float((pt > pc).mean()),
        "lift_mean": float(lift.mean()),
        "lift_credible_95": [float(np.quantile(lift, 0.025)), float(np.quantile(lift, 0.975))],
        "expected_loss_ship": float(np.maximum(pc - pt, 0).mean()),
        "expected_loss_keep": float(np.maximum(pt - pc, 0).mean()),
        "lift_histogram": np.histogram(lift, bins=60)[0].tolist(),
        "lift_bin_edges": np.histogram(lift, bins=60)[1].tolist(),
    }


def sequential_pvalues(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    """Cumulative conversion and p-value at each daily look of the 7-day window."""
    rows = []
    for d in range(1, days + 1):
        conv = (df.converted == 1) & (df.purchase_day <= d)
        c = conv[df.group == "control"]
        t = conv[df.group == "treatment"]
        r = two_proportion_ztest(int(c.sum()), len(c), int(t.sum()), len(t))
        rows.append(
            {
                "day": d,
                "conv_control": c.mean(),
                "conv_treatment": t.mean(),
                "p_value": r.p_value,
                "z": r.statistic,
            }
        )
    return pd.DataFrame(rows)


def peeking_simulation(
    control_like: pd.DataFrame, n_sims: int, seed: int, days: int = 7, alpha: float = 0.05
) -> dict[str, Any]:
    """A/A tests on untreated outcomes: how often does *any* daily look reach p < alpha?

    Each simulation re-randomises users 50/50 (no true effect) and runs the z-test after
    every day. Peeking inflates the false-positive rate well above alpha; the empirical
    per-look threshold that restores a 5% family-wise rate is reported as the fix.
    """
    rng = np.random.default_rng(seed)
    day = control_like["purchase_day"].to_numpy()
    conv = control_like["converted"].to_numpy() == 1
    n = len(day)
    cum = np.stack([conv & (day <= d) & (day > 0) for d in range(1, days + 1)])  # (days, n)
    min_p = np.empty(n_sims)
    final_p = np.empty(n_sims)
    for s in range(n_sims):
        treat = rng.random(n) < 0.5
        nt, nc = treat.sum(), (~treat).sum()
        xt = cum[:, treat].sum(axis=1)
        xc = cum[:, ~treat].sum(axis=1)
        pooled = (xt + xc) / n
        se = np.sqrt(np.maximum(pooled * (1 - pooled), 1e-12) * (1 / nt + 1 / nc))
        z = (xt / nt - xc / nc) / se
        p = 2 * stats.norm.sf(np.abs(z))
        min_p[s] = p.min()
        final_p[s] = p[-1]
    return {
        "n_simulations": n_sims,
        "looks": days,
        "fpr_final_look_only": float((final_p < alpha).mean()),
        "fpr_any_look": float((min_p < alpha).mean()),
        "empirical_pocock_threshold": float(np.quantile(min_p, alpha)),
        "min_p_values": min_p.tolist(),
    }
