"""Statistical routines checked against reference implementations and known answers."""

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

from ecom.stats.core import (
    anova_or_kruskal,
    benjamini_hochberg,
    bootstrap_ci,
    chi_square_2x2,
    cohens_h,
    dunn_test,
    mann_whitney,
    spearman,
    srm_test,
    two_proportion_ztest,
    welch_ttest,
)
from ecom.stats.cuped import cuped_adjust
from ecom.stats.power import (
    achieved_power,
    minimum_detectable_effect,
    sample_size_two_proportions,
)

rng = np.random.default_rng(7)


def test_two_proportion_matches_statsmodels():
    res = two_proportion_ztest(120, 2000, 160, 2000)
    z, p = proportions_ztest([160, 120], [2000, 2000])
    assert res.statistic == pytest.approx(z)
    assert res.p_value == pytest.approx(p)
    assert res.details["diff"] == pytest.approx(0.02)
    assert res.ci_low < 0.02 < res.ci_high
    assert res.details["relative_lift"] == pytest.approx(1 / 3)
    lo, hi = res.details["relative_lift_ci"]
    assert lo < 1 / 3 < hi
    assert res.significant


def test_chi_square_equals_z_squared():
    z = two_proportion_ztest(120, 2000, 160, 2000)
    chi = chi_square_2x2(120, 2000, 160, 2000)
    assert chi.statistic == pytest.approx(z.statistic**2)
    assert chi.p_value == pytest.approx(z.p_value)


def test_cohens_h_zero_for_equal_rates():
    assert cohens_h(0.3, 0.3) == 0.0


def test_welch_matches_scipy():
    a, b = rng.normal(10, 2, 300), rng.normal(10.5, 4, 200)
    res = welch_ttest(a, b)
    ref = stats.ttest_ind(b, a, equal_var=False)
    assert res.statistic == pytest.approx(ref.statistic)
    assert res.p_value == pytest.approx(ref.pvalue)
    assert res.ci_low < res.details["diff"] < res.ci_high


def test_mann_whitney_rank_biserial_extremes():
    res = mann_whitney(np.arange(50), np.arange(100, 150), seed=1)
    assert res.effect_size == pytest.approx(1.0)
    assert res.p_value < 1e-10
    assert res.ci_low > 0


def test_anova_vs_kruskal_selection():
    rng = np.random.default_rng(11)
    normal = {k: rng.normal(m, 1, 400) for k, m in [("a", 0), ("b", 0), ("c", 0.3)]}
    res = anova_or_kruskal(normal)
    assert res.test == "one-way ANOVA"
    skewed = {k: rng.lognormal(m, 1, 400) for k, m in [("a", 0), ("b", 0.5), ("c", 1)]}
    res = anova_or_kruskal(skewed)
    assert res.test.startswith("Kruskal-Wallis")
    assert res.p_value < 1e-6
    assert any(r["significant"] for r in res.details["posthoc_dunn_bh"])


def test_dunn_no_difference_between_identical_groups():
    x = rng.normal(0, 1, 500)
    rows = dunn_test({"a": x, "b": x.copy()})
    assert rows[0]["z"] == pytest.approx(0.0)
    assert not rows[0]["significant"]


def test_spearman_perfect_monotonic():
    x = np.arange(100.0)
    res = spearman(x, np.exp(x / 20))
    assert res.statistic == pytest.approx(1.0)
    noisy = spearman(x, x + rng.normal(0, 30, 100))
    assert noisy.ci_low < noisy.statistic < noisy.ci_high


def test_bootstrap_ci_covers_median():
    x = rng.lognormal(3, 0.5, 2000)
    est, lo, hi = bootstrap_ci(x, np.median, seed=3)
    assert lo < est < hi
    assert lo < np.exp(3) < hi


def test_benjamini_hochberg_matches_statsmodels():
    p = rng.uniform(0, 0.2, 25)
    ours = benjamini_hochberg(p)
    ref = multipletests(p, method="fdr_bh")[1]
    np.testing.assert_allclose(ours, ref)
    assert len(benjamini_hochberg([])) == 0


def test_srm_detects_mismatch():
    assert srm_test([5000, 5000], [0.5, 0.5]).p_value == pytest.approx(1.0)
    assert srm_test([5000, 5300], [0.5, 0.5]).p_value < 0.01


def test_sample_size_known_value():
    # 10% -> 11% (10% relative) at alpha 0.05, power 0.8 needs ~14.7k users per arm
    plan = sample_size_two_proportions(0.10, 0.10, daily_eligible_users=1000)
    assert 14_500 < plan.n_per_arm < 15_000
    assert plan.duration_days == pytest.approx(np.ceil(plan.n_per_arm / 500))
    assert achieved_power(0.10, 0.10, plan.n_per_arm) == pytest.approx(0.8, abs=0.01)


def test_mde_inverts_sample_size():
    plan = sample_size_two_proportions(0.05, 0.2)
    assert minimum_detectable_effect(0.05, plan.n_per_arm) == pytest.approx(0.2, rel=0.02)


def test_cuped_reduces_variance_and_keeps_effect():
    n = 20_000
    x = rng.gamma(2, 50, n)
    treat = rng.random(n) < 0.5
    y = 0.8 * x + rng.normal(0, 30, n) + 5 * treat
    adjusted, res = cuped_adjust(y, x)
    rho = np.corrcoef(x, y)[0, 1]
    assert res.variance_reduction == pytest.approx(rho**2, rel=0.01)
    raw_effect = y[treat].mean() - y[~treat].mean()
    adj_effect = adjusted[treat].mean() - adjusted[~treat].mean()
    assert adj_effect == pytest.approx(5, abs=1.0)
    assert abs(adj_effect - 5) < abs(raw_effect - 5) + 1.0


def test_bootstrap_falls_back_to_percentile_on_ties():
    rng = np.random.default_rng(5)
    x = np.r_[np.full(900, 168.0), rng.uniform(10, 900, 100)]
    est, lo, hi = bootstrap_ci(x, np.median)
    assert est == 168.0
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= est <= hi
