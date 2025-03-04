"""Inferential statistics on the silver tables (DuckDB over Parquet).

Each test states its hypotheses, checks assumptions, and reports an effect size and CI.
The family of primary tests is corrected for multiplicity with Benjamini-Hochberg.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from ecom.config import Config
from ecom.layout import outputs_dir, table_path
from ecom.stats.core import (
    TestResult,
    anova_or_kruskal,
    benjamini_hochberg,
    bootstrap_ci,
    chi_square_2x2,
    mann_whitney,
    spearman,
    two_proportion_ztest,
)

log = logging.getLogger(__name__)


def parquet_glob(cfg: Config, table: str) -> str:
    return (table_path(cfg, table) / "**" / "*.parquet").as_posix()


def connect(cfg: Config) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    tmp = (cfg.root / ".cache" / "tmp" / "duckdb").as_posix()
    con.execute(f"SET temp_directory = '{tmp}'")
    for name in ("sessions", "events_clean"):
        con.execute(
            f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{parquet_glob(cfg, name)}', "
            "hive_partitioning = true)"
        )
    return con


def weekend_conversion(con, alpha: float) -> list[TestResult]:
    df = (
        con.execute(
            """SELECT is_weekend, COUNT(*) AS n, SUM(CASE WHEN purchased THEN 1 ELSE 0 END) AS x
           FROM sessions GROUP BY is_weekend"""
        )
        .df()
        .set_index("is_weekend")
    )
    wd, we = df.loc[False], df.loc[True]
    z = two_proportion_ztest(
        int(wd.x),
        int(wd.n),
        int(we.x),
        int(we.n),
        alpha,
        name="Session conversion: weekend vs weekday",
    )
    z.h0, z.h1 = (
        "conversion(weekend) = conversion(weekday)",
        "conversion(weekend) != conversion(weekday)",
    )
    z.assumptions["independent_units"] = (
        "approx. - sessions of one user are correlated; effect is large relative to clustering"
    )
    chi = chi_square_2x2(int(wd.x), int(wd.n), int(we.x), int(we.n), alpha)
    chi.name = "Session conversion x weekend (chi-square)"
    return [z, chi]


def aov_by_category(con, alpha: float, top_k: int = 5, max_per_group: int = 20_000, seed: int = 0):
    orders = con.execute(
        """SELECT main_category, revenue FROM sessions
           WHERE purchased AND revenue > 0 AND main_category <> 'unknown'"""
    ).df()
    top = orders["main_category"].value_counts().head(top_k).index.tolist()
    rng = np.random.default_rng(seed)
    groups = {}
    for c in top:
        vals = orders.loc[orders["main_category"] == c, "revenue"].to_numpy()
        groups[c] = (
            rng.choice(vals, max_per_group, replace=False) if len(vals) > max_per_group else vals
        )
    res = anova_or_kruskal(groups, alpha, name=f"Order value across top {top_k} categories")
    return res, orders


def duration_buyers(con, alpha: float, max_n: int = 50_000, seed: int = 0) -> TestResult:
    df = con.execute("SELECT purchased, duration_s FROM sessions WHERE n_events >= 2").df()
    rng = np.random.default_rng(seed)
    a = df.loc[~df.purchased, "duration_s"].to_numpy(dtype=float)
    b = df.loc[df.purchased, "duration_s"].to_numpy(dtype=float)
    a = rng.choice(a, max_n, replace=False) if len(a) > max_n else a
    b = rng.choice(b, max_n, replace=False) if len(b) > max_n else b
    res = mann_whitney(a, b, alpha, name="Session duration: buyers vs non-buyers", seed=seed)
    res.details["subsampled_to"] = max_n
    return res


def price_vs_conversion(con, alpha: float, min_views: int = 30) -> TestResult:
    df = con.execute(
        f"""SELECT product_id, AVG(price) AS price,
                   SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END)::DOUBLE
                     / SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS conversion
            FROM events_clean GROUP BY product_id
            HAVING SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) >= {min_views}"""
    ).df()
    res = spearman(
        df["price"],
        df["conversion"],
        alpha,
        name=f"Product price vs view->purchase conversion (products with >= {min_views} views)",
    )
    res.details["n_products"] = len(df)
    return res


def run_all_tests(cfg: Config) -> list[dict[str, Any]]:
    alpha = float(cfg.get("experiment.alpha", 0.05))
    con = connect(cfg)
    family: list[TestResult] = []
    family += weekend_conversion(con, alpha)
    aov_res, orders = aov_by_category(con, alpha, seed=cfg.seed)
    family.append(aov_res)
    family.append(duration_buyers(con, alpha, seed=cfg.seed))
    family.append(price_vs_conversion(con, alpha))

    adjusted = benjamini_hochberg([t.p_value for t in family])
    for t, p in zip(family, adjusted, strict=True):
        t.p_adjusted = float(p)

    # Bootstrap CIs for median AOV (estimation, not part of the hypothesis family)
    rng = np.random.default_rng(cfg.seed)
    vals = orders["revenue"].to_numpy()
    sample = rng.choice(vals, 20_000, replace=False) if len(vals) > 20_000 else vals
    med, lo, hi = bootstrap_ci(sample, np.median, alpha, seed=cfg.seed)
    bootstrap = [
        {"segment": "all orders", "median": med, "ci_low": lo, "ci_high": hi, "n": len(vals)}
    ]
    for c in orders["main_category"].value_counts().head(5).index:
        v = orders.loc[orders["main_category"] == c, "revenue"].to_numpy()
        n = len(v)  # orders in the segment (the CI uses a subsample of at most 20k)
        v = rng.choice(v, 20_000, replace=False) if n > 20_000 else v
        m, l_, h_ = bootstrap_ci(v, np.median, alpha, seed=cfg.seed)
        bootstrap.append({"segment": c, "median": m, "ci_low": l_, "ci_high": h_, "n": n})

    results = [t.to_dict() for t in family]
    out = outputs_dir(cfg)
    payload = {"tests": results, "bootstrap_median_aov": bootstrap, "alpha": alpha}
    (out / "stats_tests.json").write_text(json.dumps(payload, indent=2, default=_json_default))
    pd.DataFrame(
        [
            {
                "test": t.name,
                "method": t.test,
                "statistic": t.statistic,
                "p_value": t.p_value,
                "p_bh": t.p_adjusted,
                "effect_size": t.effect_size,
                "effect": t.effect_size_name,
                "ci_low": t.ci_low,
                "ci_high": t.ci_high,
                "ci_of": t.ci_of,
                "n": t.n,
                "significant_bh": t.significant,
            }
            for t in family
        ]
    ).to_csv(out / "stats_tests.csv", index=False)
    pd.DataFrame(bootstrap).to_csv(out / "bootstrap_median_aov.csv", index=False)
    for t in family:
        log.info(
            "%-60s p=%.3g p_bh=%.3g %s=%.4f",
            t.name,
            t.p_value,
            t.p_adjusted,
            t.effect_size_name,
            t.effect_size,
        )
    return results


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
