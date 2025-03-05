"""Experiment sizing: required sample size, achievable MDE and test duration."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


@dataclass
class PowerPlan:
    baseline_rate: float
    mde_relative: float
    target_rate: float
    alpha: float
    power: float
    effect_size_h: float
    n_per_arm: int
    n_total: int
    daily_eligible_users: float | None = None
    duration_days: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def sample_size_two_proportions(
    baseline_rate: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    daily_eligible_users: float | None = None,
    treatment_share: float = 0.5,
) -> PowerPlan:
    """Users per arm to detect a relative lift ``mde_relative`` with a two-sided z-test."""
    target = baseline_rate * (1 + mde_relative)
    h = abs(proportion_effectsize(target, baseline_rate))
    n1 = NormalIndPower().solve_power(
        effect_size=h, alpha=alpha, power=power, ratio=ratio, alternative="two-sided"
    )
    n_control = math.ceil(n1)
    n_treat = math.ceil(n1 * ratio)
    duration = None
    if daily_eligible_users:
        # the smaller arm is the bottleneck when the split is uneven
        share = min(treatment_share, 1 - treatment_share)
        duration = math.ceil(max(n_control, n_treat) / (daily_eligible_users * share))
    return PowerPlan(
        baseline_rate=baseline_rate,
        mde_relative=mde_relative,
        target_rate=target,
        alpha=alpha,
        power=power,
        effect_size_h=h,
        n_per_arm=n_control,
        n_total=n_control + n_treat,
        daily_eligible_users=daily_eligible_users,
        duration_days=duration,
    )


def minimum_detectable_effect(
    baseline_rate: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Smallest relative lift detectable with ``n_per_arm`` users per arm."""
    h = NormalIndPower().solve_power(
        nobs1=n_per_arm, alpha=alpha, power=power, ratio=1.0, alternative="two-sided"
    )
    # invert Cohen's h: h = 2 asin(sqrt(p2)) - 2 asin(sqrt(p1))
    p2 = math.sin(math.asin(math.sqrt(baseline_rate)) + h / 2) ** 2
    return p2 / baseline_rate - 1


def achieved_power(
    baseline_rate: float, mde_relative: float, n_per_arm: int, alpha: float = 0.05
) -> float:
    h = abs(proportion_effectsize(baseline_rate * (1 + mde_relative), baseline_rate))
    return float(NormalIndPower().power(effect_size=h, nobs1=n_per_arm, alpha=alpha, ratio=1.0))
