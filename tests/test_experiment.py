import numpy as np
import pandas as pd
import pytest

from ecom.experiment.analyze_ab import (
    bayesian_conversion,
    peeking_simulation,
    revenue_tests,
    sequential_pvalues,
)
from ecom.experiment.run import decide
from ecom.experiment.simulate_ab import (
    EffectModel,
    assign,
    choose_salt,
    hash_unit,
    naive_utc,
    simulate,
)
from ecom.experiment.srm_check import check_srm


def population(n: int = 20_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = rng.beta(1, 20, n)
    label = (rng.random(n) < p).astype(int)
    cutoff = pd.Timestamp("2019-11-24")
    return pd.DataFrame(
        {
            "user_id": np.arange(1_000, 1_000 + n),
            "propensity": p,
            "decile": np.ceil(pd.Series(p).rank(ascending=False) / n * 10).astype(int),
            "price_sensitivity": rng.uniform(-0.5, 1, n),
            "avg_price_viewed": rng.uniform(20, 500, n),
            "monetary": rng.gamma(1, 50, n),
            "label": label,
            "label_revenue": label * rng.uniform(20, 500, n),
            "label_purchases": label,
            "first_purchase_time": np.where(
                label == 1, cutoff + pd.to_timedelta(rng.uniform(0, 7 * 86400, n), unit="s"), pd.NaT
            ),
            "cutoff": cutoff.date(),
            "top_category": rng.choice(["electronics", "appliances", "kids"], n),
        }
    )


def test_hash_assignment_is_deterministic_and_balanced():
    ids = np.arange(100_000)
    a1, a2 = assign(ids, "salt"), assign(ids, "salt")
    assert (a1 == a2).all()
    assert abs((a1 == "treatment").mean() - 0.5) < 0.01
    assert not (assign(ids, "other-salt") == a1).all()
    u = hash_unit(ids[:5000], "s")
    assert u.min() >= 0 and u.max() < 1


def test_naive_utc_handles_aware_and_naive():
    aware = pd.Series(pd.to_datetime(["2019-11-24 05:30:00+05:30"]))
    naive = pd.Series(pd.to_datetime(["2019-11-24 00:00:00"]))
    assert naive_utc(aware).iloc[0] == naive.iloc[0]
    assert naive_utc(naive).iloc[0] == naive.iloc[0]


def test_simulation_only_changes_targeted_treatment_users():
    pop = population()
    df = simulate(pop, EffectModel(0.1, 0.5, 1), "salt", 0.5, seed=1)
    untouched = ~df["treated"]
    assert (df.loc[untouched, "converted"] == pop.loc[untouched, "label"]).all()
    assert np.allclose(df.loc[untouched, "revenue"], pop.loc[untouched, "label_revenue"])
    treated_buyers = df["treated"] & (pop["label"] == 1)
    assert np.allclose(
        df.loc[treated_buyers, "revenue"], pop.loc[treated_buyers, "label_revenue"] * 0.9
    )
    assert df.loc[df["treated"], "simulated_conversion"].sum() > 0
    assert (df.loc[df.converted == 1, "purchase_day"].between(1, 7)).all()
    assert (df.loc[df.converted == 0, "purchase_day"] == 0).all()
    srm = check_srm(df)
    assert sum(srm.details["observed"]) == len(df)
    assert srm.alpha == 0.01


def test_sequential_and_peeking():
    pop = population(30_000)
    df = simulate(pop, EffectModel(0.1, 0.0, 1), "salt", 0.5, seed=2)  # no true effect
    seq = sequential_pvalues(df)
    assert list(seq["day"]) == list(range(1, 8))
    assert (seq["conv_control"].diff().dropna() >= 0).all()
    aa = peeking_simulation(df, n_sims=300, seed=3)
    assert aa["fpr_any_look"] >= aa["fpr_final_look_only"]
    assert aa["fpr_final_look_only"] < 0.1
    assert aa["empirical_pocock_threshold"] < 0.05


def test_cuped_revenue_shrinks_ci():
    rng = np.random.default_rng(4)
    n = 10_000
    pre = rng.gamma(1, 50, n)
    df = pd.DataFrame(
        {
            "group": np.where(rng.random(n) < 0.5, "treatment", "control"),
            "pre_spend": pre,
            "revenue": 0.7 * pre + rng.gamma(1, 10, n),
        }
    )
    res = revenue_tests(df, 0.05)
    assert res["cuped_stats"]["variance_reduction"] > 0.5
    assert res["ci_width_cuped"] < res["ci_width_raw"]


def test_bayesian_prefers_better_arm():
    df = pd.DataFrame(
        {
            "group": ["control"] * 5000 + ["treatment"] * 5000,
            "converted": [1] * 250 + [0] * 4750 + [1] * 350 + [0] * 4650,
        }
    )
    res = bayesian_conversion(df, n_draws=50_000)
    assert res["prob_treatment_better"] > 0.99
    lo, hi = res["lift_credible_95"]
    assert lo < 0.4 < hi


def _result(p, diff, ci=(0.0, 0.0), lift=0.0, alpha=0.05):
    return {
        "p_value": p,
        "alpha": alpha,
        "ci_low": ci[0],
        "ci_high": ci[1],
        "details": {
            "diff": diff,
            "relative_lift": lift,
            "p_a": 0.05,
            "p_b": 0.05 + diff,
            "passed": True,
        },
    }


@pytest.mark.parametrize(
    ("conv_p", "rev_p", "rev_diff", "aov_lift", "aov_p", "verdict"),
    [
        (0.001, 0.5, 0.1, -0.05, 0.01, "SHIP"),
        (0.4, 0.5, 0.1, -0.05, 0.01, "DON'T SHIP - no significant lift on the primary metric"),
        (
            0.001,
            0.01,
            -2.0,
            -0.05,
            0.01,
            "DON'T SHIP - primary metric wins but a guardrail is breached",
        ),
        (
            0.001,
            0.5,
            0.1,
            -0.2,
            0.001,
            "DON'T SHIP - primary metric wins but a guardrail is breached",
        ),
    ],
)
def test_decision_rules(conv_p, rev_p, rev_diff, aov_lift, aov_p, verdict):
    srm = {"p_value": 0.5, "details": {"passed": True}}
    conv = _result(conv_p, 0.01, lift=0.2)
    rev = {"cuped": _result(rev_p, rev_diff, (-1, 1))}
    aov = _result(aov_p, -1.0, lift=aov_lift)
    assert decide(srm, conv, rev, aov, 0.10)["verdict"] == verdict
    srm_bad = {"p_value": 0.0001, "details": {"passed": False}}
    assert decide(srm_bad, conv, rev, aov, 0.10)["verdict"].startswith("INVALID")


def test_salt_search_is_outcome_blind_and_balanced():
    pop = population(20_000).assign(n_sessions=lambda d: d["monetary"] // 10)
    res = choose_salt(pop, "salt", 0.5)
    assert res["history"][-1]["balanced"] and res["history"][-1]["salt"] == res["salt"]
    assert all(not h["balanced"] for h in res["history"][:-1])
    assert res["history"][-1]["srm_p"] >= 0.2
    # only pre-treatment columns are consulted: changing outcomes changes nothing
    flipped = pop.assign(label=1 - pop["label"], label_revenue=0.0)
    assert choose_salt(flipped, "salt", 0.5)["salt"] == res["salt"]
    strict = choose_salt(pop, "salt", 0.5, min_srm_p=0.9, max_tries=200)
    assert strict["candidates_tried"] >= 1 and strict["history"][-1]["srm_p"] >= 0.9
