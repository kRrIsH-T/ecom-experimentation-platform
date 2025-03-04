"""Pure statistical tests used by the analysis and the A/B experiment.

Every test returns a :class:`TestResult` carrying the hypotheses, assumption checks, an
effect size and a confidence interval, so results can be reported uniformly.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import stats

ArrayLike = Sequence[float] | np.ndarray


@dataclass
class TestResult:
    name: str
    test: str
    h0: str
    h1: str
    statistic: float
    p_value: float
    effect_size: float
    effect_size_name: str
    ci_low: float
    ci_high: float
    ci_of: str
    n: int
    assumptions: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    p_adjusted: float | None = None
    alpha: float = 0.05

    @property
    def significant(self) -> bool:
        p = self.p_adjusted if self.p_adjusted is not None else self.p_value
        return bool(p < self.alpha)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["significant"] = self.significant
        return d


# --- proportions --------------------------------------------------------------------------


def cohens_h(p1: float, p2: float) -> float:
    return float(2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1)))


def two_proportion_ztest(
    x_a: int, n_a: int, x_b: int, n_b: int, alpha: float = 0.05, name: str = "two-proportion"
) -> TestResult:
    """Pooled two-sided z-test for p_b - p_a with an unpooled Wald CI on the difference."""
    p_a, p_b = x_a / n_a, x_b / n_b
    pooled = (x_a + x_b) / (n_a + n_b)
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se_pooled if se_pooled > 0 else 0.0
    p = 2 * stats.norm.sf(abs(z))
    se = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    zc = stats.norm.ppf(1 - alpha / 2)
    diff = p_b - p_a
    # Relative lift CI via the delta method on log(p_b / p_a)
    rel = p_b / p_a - 1 if p_a > 0 else np.nan
    se_log = np.sqrt((1 - p_a) / x_a + (1 - p_b) / x_b) if x_a > 0 and x_b > 0 else np.nan
    min_cell = min(x_a, n_a - x_a, x_b, n_b - x_b)
    return TestResult(
        name=name,
        test="two-proportion z-test",
        h0="p_A = p_B",
        h1="p_A != p_B",
        statistic=float(z),
        p_value=float(p),
        effect_size=cohens_h(p_a, p_b),
        effect_size_name="Cohen's h",
        ci_low=float(diff - zc * se),
        ci_high=float(diff + zc * se),
        ci_of="p_B - p_A",
        n=int(n_a + n_b),
        assumptions={
            "independent_units": True,
            "min_expected_cell_>=10": bool(min_cell >= 10),
            "min_cell": int(min_cell),
        },
        details={
            "p_a": p_a,
            "p_b": p_b,
            "diff": diff,
            "relative_lift": float(rel),
            "relative_lift_ci": [
                float(np.exp(np.log(p_b / p_a) - zc * se_log) - 1),
                float(np.exp(np.log(p_b / p_a) + zc * se_log) - 1),
            ]
            if np.isfinite(se_log)
            else [np.nan, np.nan],
            "n_a": n_a,
            "n_b": n_b,
        },
        alpha=alpha,
    )


def chi_square_2x2(x_a: int, n_a: int, x_b: int, n_b: int, alpha: float = 0.05) -> TestResult:
    table = np.array([[x_a, n_a - x_a], [x_b, n_b - x_b]])
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    phi = np.sqrt(chi2 / table.sum())
    return TestResult(
        name="chi-square 2x2",
        test="Pearson chi-square test of independence",
        h0="outcome independent of group",
        h1="outcome depends on group",
        statistic=float(chi2),
        p_value=float(p),
        effect_size=float(phi),
        effect_size_name="phi",
        ci_low=np.nan,
        ci_high=np.nan,
        ci_of="",
        n=int(table.sum()),
        assumptions={"min_expected_>=5": bool(expected.min() >= 5), "dof": int(dof)},
        alpha=alpha,
    )


def srm_test(observed: Sequence[int], expected_shares: Sequence[float]) -> TestResult:
    """Sample-ratio-mismatch chi-square goodness-of-fit test."""
    obs = np.asarray(observed, dtype=float)
    exp = np.asarray(expected_shares, dtype=float) * obs.sum()
    chi2, p = stats.chisquare(obs, exp)
    return TestResult(
        name="SRM check",
        test="chi-square goodness of fit",
        h0="assignment matches the designed split",
        h1="sample ratio mismatch",
        statistic=float(chi2),
        p_value=float(p),
        effect_size=float(np.max(np.abs(obs / obs.sum() - np.asarray(expected_shares)))),
        effect_size_name="max share deviation",
        ci_low=np.nan,
        ci_high=np.nan,
        ci_of="",
        n=int(obs.sum()),
        details={"observed": obs.tolist(), "expected": exp.tolist()},
        alpha=0.01,
    )


# --- continuous ---------------------------------------------------------------------------


def _normality(x: np.ndarray, max_n: int = 5000, seed: int = 0) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    if len(x) > max_n:
        x = np.random.default_rng(seed).choice(x, max_n, replace=False)
    if len(x) < 3:
        return {"shapiro_p": np.nan, "normal": False}
    w, p = stats.shapiro(x)
    return {
        "shapiro_W": float(w),
        "shapiro_p": float(p),
        "normal": bool(p > 0.05),
        "skew": float(stats.skew(x)),
    }


def welch_ttest(
    a: ArrayLike, b: ArrayLike, alpha: float = 0.05, name: str = "Welch t-test"
) -> TestResult:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    t, p = stats.ttest_ind(b, a, equal_var=False)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    tc = stats.t.ppf(1 - alpha / 2, df)
    diff = b.mean() - a.mean()
    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return TestResult(
        name=name,
        test="Welch two-sample t-test",
        h0="mean_A = mean_B",
        h1="mean_A != mean_B",
        statistic=float(t),
        p_value=float(p),
        effect_size=float(diff / pooled_sd) if pooled_sd > 0 else 0.0,
        effect_size_name="Cohen's d",
        ci_low=float(diff - tc * se),
        ci_high=float(diff + tc * se),
        ci_of="mean_B - mean_A",
        n=len(a) + len(b),
        assumptions={
            "large_sample_CLT": bool(min(len(a), len(b)) >= 30),
            "levene_p": float(stats.levene(a, b).pvalue),
        },
        details={
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
            "diff": float(diff),
            "relative_lift": float(diff / a.mean()) if a.mean() else np.nan,
            "df": float(df),
        },
        alpha=alpha,
    )


def mann_whitney(
    a: ArrayLike, b: ArrayLike, alpha: float = 0.05, name: str = "Mann-Whitney U", seed: int = 0
) -> TestResult:
    """Two-sided Mann-Whitney U with rank-biserial effect and bootstrap CI of the median gap."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    u, p = stats.mannwhitneyu(b, a, alternative="two-sided")
    rbc = 2 * u / (len(a) * len(b)) - 1  # P(B > A) - P(B < A)
    lo, hi = bootstrap_ci_diff(a, b, np.median, alpha=alpha, seed=seed)
    return TestResult(
        name=name,
        test="Mann-Whitney U (two-sided)",
        h0="distributions of A and B are equal (no stochastic dominance)",
        h1="one group tends to have larger values",
        statistic=float(u),
        p_value=float(p),
        effect_size=float(rbc),
        effect_size_name="rank-biserial r",
        ci_low=lo,
        ci_high=hi,
        ci_of="median_B - median_A (bootstrap)",
        n=len(a) + len(b),
        assumptions={
            "normal_A": _normality(a)["normal"],
            "normal_B": _normality(b)["normal"],
            "independent_samples": True,
        },
        details={"median_a": float(np.median(a)), "median_b": float(np.median(b))},
        alpha=alpha,
    )


def anova_or_kruskal(
    groups: dict[str, ArrayLike], alpha: float = 0.05, name: str = "k-group comparison"
) -> TestResult:
    """One-way ANOVA when normal with equal variances, otherwise Kruskal-Wallis + Dunn.

    Normality is checked per group with Shapiro-Wilk at a Bonferroni-corrected level so the
    chance of wrongly rejecting normality does not grow with the number of groups.
    """
    labels = list(groups)
    arrays = [np.asarray(groups[k], dtype=float) for k in labels]
    level = 0.05 / len(arrays)
    normal = {
        k: bool(_normality(x)["shapiro_p"] > level) for k, x in zip(labels, arrays, strict=True)
    }
    levene_p = float(stats.levene(*arrays).pvalue)
    f, p_anova = stats.f_oneway(*arrays)
    h, p_kw = stats.kruskal(*arrays)
    n = sum(len(x) for x in arrays)
    k = len(arrays)
    use_anova = all(normal.values()) and levene_p > 0.05
    grand = np.concatenate(arrays).mean()
    ss_between = sum(len(x) * (x.mean() - grand) ** 2 for x in arrays)
    ss_total = sum(((x - grand) ** 2).sum() for x in arrays)
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
    eps_sq = (h - k + 1) / (n - k)  # epsilon-squared for Kruskal-Wallis
    posthoc = dunn_test(groups) if not use_anova else []
    return TestResult(
        name=name,
        test="one-way ANOVA" if use_anova else "Kruskal-Wallis H (ANOVA assumptions failed)",
        h0="all groups share the same mean/distribution",
        h1="at least one group differs",
        statistic=float(f if use_anova else h),
        p_value=float(p_anova if use_anova else p_kw),
        effect_size=float(eta_sq if use_anova else eps_sq),
        effect_size_name="eta-squared" if use_anova else "epsilon-squared",
        ci_low=np.nan,
        ci_high=np.nan,
        ci_of="",
        n=int(n),
        assumptions={
            "normal_by_group": normal,
            "levene_p": levene_p,
            "equal_variances": levene_p > 0.05,
        },
        details={
            "anova_F": float(f),
            "anova_p": float(p_anova),
            "kruskal_H": float(h),
            "kruskal_p": float(p_kw),
            "group_medians": {
                k_: float(np.median(x)) for k_, x in zip(labels, arrays, strict=True)
            },
            "group_means": {k_: float(x.mean()) for k_, x in zip(labels, arrays, strict=True)},
            "group_n": {k_: len(x) for k_, x in zip(labels, arrays, strict=True)},
            "posthoc_dunn_bh": posthoc,
        },
        alpha=alpha,
    )


def dunn_test(groups: dict[str, ArrayLike]) -> list[dict[str, Any]]:
    """Dunn's pairwise post-hoc test (tie-corrected) with Benjamini-Hochberg adjustment."""
    labels = list(groups)
    arrays = [np.asarray(groups[k], dtype=float) for k in labels]
    allx = np.concatenate(arrays)
    ranks = stats.rankdata(allx)
    n = len(allx)
    _, counts = np.unique(allx, return_counts=True)
    tie = (counts**3 - counts).sum() / (12 * (n - 1))
    bounds = np.cumsum([0] + [len(x) for x in arrays])
    mean_rank = [ranks[bounds[i] : bounds[i + 1]].mean() for i in range(len(arrays))]
    rows = []
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            ni, nj = len(arrays[i]), len(arrays[j])
            se = np.sqrt((n * (n + 1) / 12 - tie) * (1 / ni + 1 / nj))
            z = (mean_rank[i] - mean_rank[j]) / se
            rows.append(
                {
                    "a": labels[i],
                    "b": labels[j],
                    "z": float(z),
                    "p": float(2 * stats.norm.sf(abs(z))),
                }
            )
    adj = benjamini_hochberg([r["p"] for r in rows])
    for r, pa in zip(rows, adj, strict=True):
        r["p_adj"] = float(pa)
        r["significant"] = bool(pa < 0.05)
    return rows


def spearman(x: ArrayLike, y: ArrayLike, alpha: float = 0.05, name: str = "Spearman") -> TestResult:
    """Spearman rank correlation with a Fisher-z confidence interval."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rho, p = stats.spearmanr(x, y)
    n = len(x)
    zc = stats.norm.ppf(1 - alpha / 2)
    # Bonett-Wright standard error for Spearman's rho
    se = np.sqrt((1 + rho**2 / 2) / (n - 3))
    z = np.arctanh(np.clip(rho, -0.999999, 0.999999))
    return TestResult(
        name=name,
        test="Spearman rank correlation",
        h0="rho = 0 (no monotonic association)",
        h1="rho != 0",
        statistic=float(rho),
        p_value=float(p),
        effect_size=float(rho),
        effect_size_name="Spearman rho",
        ci_low=float(np.tanh(z - zc * se)),
        ci_high=float(np.tanh(z + zc * se)),
        ci_of="rho (Fisher z, Bonett-Wright SE)",
        n=n,
        assumptions={"monotonic_relationship_assumed": True, "n_>=10": n >= 10},
        alpha=alpha,
    )


# --- resampling and multiplicity ----------------------------------------------------------


def bootstrap_ci(
    x: ArrayLike,
    statistic: Callable[[np.ndarray], float] = np.median,
    alpha: float = 0.05,
    n_resamples: int = 2000,
    seed: int = 0,
    method: str = "BCa",
) -> tuple[float, float, float]:
    """(estimate, low, high) using scipy's bootstrap (BCa by default).

    BCa is undefined when the jackknife distribution is degenerate (e.g. a median sitting on
    a heavily tied value); the percentile interval is used in that case.
    """
    x = np.asarray(x, dtype=float)

    def run(m: str):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", stats.DegenerateDataWarning)
            warnings.simplefilter("ignore", RuntimeWarning)  # BCa's 0/0 on tied data
            return stats.bootstrap(
                (x,),
                lambda s, axis=-1: np.apply_along_axis(statistic, axis, s),
                confidence_level=1 - alpha,
                n_resamples=n_resamples,
                method=m,
                random_state=np.random.default_rng(seed),
            ).confidence_interval

    ci = run(method)
    if not (np.isfinite(ci.low) and np.isfinite(ci.high)):
        ci = run("percentile")
    return float(statistic(x)), float(ci.low), float(ci.high)


def bootstrap_ci_diff(
    a: ArrayLike,
    b: ArrayLike,
    statistic: Callable[[np.ndarray], float] = np.median,
    alpha: float = 0.05,
    n_resamples: int = 2000,
    seed: int = 0,
    max_n: int = 20_000,
) -> tuple[float, float]:
    """Percentile bootstrap CI for statistic(b) - statistic(a) (subsampled for speed)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a = rng.choice(a, max_n, replace=False) if len(a) > max_n else a
    b = rng.choice(b, max_n, replace=False) if len(b) > max_n else b
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        diffs[i] = statistic(rng.choice(b, len(b))) - statistic(rng.choice(a, len(a)))
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """BH step-up adjusted p-values (FDR control), returned in the input order."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return out
