"""Simulate a targeted-discount A/B test on held-out late-November users.

Design
------
* Population: every user active in the 21 days before the test cutoff (2019-11-24); the
  outcome window is the following 7 days, which no model has seen.
* Assignment: deterministic SHA-256 hash of ``salt:user_id`` -> 50/50 control/treatment.
* Control: business as usual (observed outcomes).
* Treatment: users in the top propensity decile(s) receive a ``discount`` coupon.

Effect model (stated assumptions)
---------------------------------
1. A targeted user's baseline purchase probability is the champion model's propensity p.
2. The coupon multiplies purchase *odds* by ``1 + base_uplift * s`` where ``s`` in [0, 1] is
   the user's percentile of observed price sensitivity (users who cart cheaper items than
   they view respond more to discounts).
3. A targeted non-buyer converts with probability (p' - p) / (1 - p), which lifts the
   marginal purchase probability from p to p'. Their order value is their average viewed
   price, minus the discount.
4. Targeted users who would have bought anyway still redeem the coupon, so their revenue is
   reduced by the discount: the cost side that the AOV / revenue guardrails must catch.
5. Simulated purchases happen uniformly at random inside the 7-day window.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

WINDOW_DAYS = 7


def naive_utc(values) -> pd.Series:
    """Timestamps as tz-naive UTC whether Parquet stored them tz-aware or naive."""
    s = pd.to_datetime(pd.Series(values))
    return s.dt.tz_convert("UTC").dt.tz_localize(None) if s.dt.tz is not None else s


def hash_unit(user_ids: np.ndarray, salt: str) -> np.ndarray:
    """Deterministic uniform [0, 1) value per user from SHA-256(salt:user_id)."""
    out = np.empty(len(user_ids))
    for i, uid in enumerate(user_ids.tolist()):
        digest = hashlib.sha256(f"{salt}:{uid}".encode()).digest()
        out[i] = int.from_bytes(digest[:8], "big") / 2**64
    return out


def assign(user_ids: np.ndarray, salt: str, treatment_share: float = 0.5) -> np.ndarray:
    return np.where(hash_unit(user_ids, salt) < treatment_share, "treatment", "control")


BALANCE_COVARIATES = ("monetary", "propensity", "n_sessions")


def standardized_mean_diff(x: np.ndarray, treat: np.ndarray) -> float:
    a, b = x[~treat], x[treat]
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((b.mean() - a.mean()) / pooled) if pooled > 0 else 0.0


def choose_salt(
    users: pd.DataFrame,
    base_salt: str,
    treatment_share: float,
    min_srm_p: float = 0.2,
    max_abs_smd: float = 0.02,
    max_tries: int = 50,
) -> dict:
    """Pre-launch salt search ("seedfinder"): first candidate salt whose split is balanced.

    Uses only pre-treatment information (assignment counts and pre-period covariates), so it
    cannot bias the treatment estimate; it just avoids launching on an unlucky hash draw.
    Candidates are ``base_salt``, ``base_salt#1``, ``base_salt#2``, ...
    """
    from ecom.stats.core import srm_test

    ids = users["user_id"].to_numpy()
    tried = []
    for i in range(max_tries):
        salt = base_salt if i == 0 else f"{base_salt}#{i}"
        treat = assign(ids, salt, treatment_share) == "treatment"
        p = srm_test(
            [int((~treat).sum()), int(treat.sum())], [1 - treatment_share, treatment_share]
        ).p_value
        smd = {
            c: standardized_mean_diff(users[c].to_numpy(dtype=float), treat)
            for c in BALANCE_COVARIATES
            if c in users
        }
        ok = p >= min_srm_p and all(abs(v) <= max_abs_smd for v in smd.values())
        tried.append({"salt": salt, "srm_p": float(p), "smd": smd, "balanced": ok})
        if ok:
            return {
                "salt": salt,
                "candidates_tried": i + 1,
                "criteria": {"min_srm_p": min_srm_p, "max_abs_smd": max_abs_smd},
                "history": tried,
            }
    raise RuntimeError(f"no balanced salt in {max_tries} candidates: {tried[-1]}")


@dataclass
class EffectModel:
    discount: float = 0.10
    base_uplift: float = 0.25
    top_deciles: int = 1


def simulate(
    users: pd.DataFrame, effect: EffectModel, salt: str, treatment_share: float, seed: int
) -> pd.DataFrame:
    """Return one row per user with group, outcome, revenue and (simulated) purchase time.

    ``users`` needs: user_id, propensity, decile, price_sensitivity, avg_price_viewed,
    monetary (pre-period spend), label, label_revenue, first_purchase_time, cutoff.
    """
    rng = np.random.default_rng(seed)
    df = users.reset_index(drop=True).copy()
    df["group"] = assign(df["user_id"].to_numpy(), salt, treatment_share)
    targeted = (df["decile"] <= effect.top_deciles).to_numpy()
    treated = targeted & (df["group"] == "treatment").to_numpy()
    bought = (df["label"] == 1).to_numpy()

    sens = df["price_sensitivity"].rank(pct=True).to_numpy()
    p = df["propensity"].clip(1e-6, 1 - 1e-6).to_numpy()
    odds = p / (1 - p) * (1 + effect.base_uplift * sens)
    p_new = odds / (1 + odds)
    flip_prob = np.where(treated & ~bought, (p_new - p) / (1 - p), 0.0)
    flipped = rng.random(len(df)) < flip_prob
    converted = bought | flipped

    revenue = df["label_revenue"].to_numpy(dtype=float)
    revenue = np.where(treated & bought, revenue * (1 - effect.discount), revenue)
    new_order = df["avg_price_viewed"].to_numpy(dtype=float) * (1 - effect.discount)
    revenue = np.where(flipped, new_order, revenue)

    cutoff = naive_utc(df["cutoff"].to_numpy())
    sim_time = cutoff + pd.to_timedelta(rng.uniform(0, WINDOW_DAYS * 86400, len(df)), unit="s")
    purchase_time = sim_time.where(flipped, naive_utc(df["first_purchase_time"]))
    day = ((purchase_time - cutoff).dt.total_seconds() // 86400).clip(0, WINDOW_DAYS - 1) + 1

    df["targeted"] = targeted
    df["treated"] = treated
    df["purchase_time"] = purchase_time.where(converted)
    df["purchase_day"] = np.where(converted, day.fillna(0), 0).astype(int)
    df["converted"] = converted.astype(int)
    df["revenue"] = revenue
    df["orders"] = np.where(flipped, 1, df["label_purchases"]).astype(int)
    df["simulated_conversion"] = flipped
    df["pre_spend"] = df["monetary"].astype(float)
    return df
