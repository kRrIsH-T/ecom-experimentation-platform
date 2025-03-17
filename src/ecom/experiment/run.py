"""End-to-end experiment step: design (power) -> simulate -> SRM -> analyse -> decide."""

from __future__ import annotations

import json
import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecom.config import Config
from ecom.experiment.analyze_ab import (
    aov_guardrail,
    bayesian_conversion,
    conversion_cupac,
    conversion_test,
    peeking_simulation,
    revenue_tests,
    segment_breakdown,
    sequential_pvalues,
)
from ecom.experiment.simulate_ab import EffectModel, choose_salt, naive_utc, simulate
from ecom.experiment.srm_check import check_srm
from ecom.layout import figures_dir, outputs_dir, table_path
from ecom.stats.power import achieved_power, minimum_detectable_effect, sample_size_two_proportions
from ecom.viz import BLUE, INK_2, MUTED, ORANGE

log = logging.getLogger(__name__)


def load_population(cfg: Config) -> pd.DataFrame:
    ds = pd.read_parquet(table_path(cfg, "model_dataset"))
    test = ds[ds["split"] == "test"].copy()
    scores = pd.read_parquet(table_path(cfg, "scores"))
    scores = scores[scores["snapshot"] == "test"][["user_id", "propensity", "decile"]]
    return test.merge(scores, on="user_id", how="inner")


def decide(srm: dict, conv: dict, rev: dict, aov: dict, max_aov_drop: float) -> dict[str, Any]:
    reasons = []
    srm_ok = srm["details"]["passed"]
    reasons.append(f"SRM check {'passed' if srm_ok else 'FAILED'} (p = {srm['p_value']:.3f}).")
    lift = conv["details"]["relative_lift"]
    primary_win = conv["p_value"] < conv["alpha"] and conv["details"]["diff"] > 0
    reasons.append(
        f"Conversion {conv['details']['p_a']:.2%} -> {conv['details']['p_b']:.2%} "
        f"({lift:+.1%} relative, p = {conv['p_value']:.3g}): "
        + ("significant improvement." if primary_win else "not a significant improvement.")
    )
    cuped = rev["cuped"]
    rev_harm = cuped["p_value"] < cuped["alpha"] and cuped["details"]["diff"] < 0
    reasons.append(
        f"Revenue/user (CUPED) diff {cuped['details']['diff']:+.2f} "
        f"[{cuped['ci_low']:+.2f}, {cuped['ci_high']:+.2f}]: "
        + ("significant harm." if rev_harm else "no significant harm.")
    )
    aov_change = aov["details"]["relative_lift"]
    aov_ok = not (aov_change < -max_aov_drop and aov["p_value"] < aov["alpha"])
    reasons.append(
        f"Buyer AOV {aov_change:+.1%} (p = {aov['p_value']:.3g}); guardrail allows at most "
        f"-{max_aov_drop:.0%}: " + ("within guardrail." if aov_ok else "guardrail breached.")
    )
    if not srm_ok:
        verdict = "INVALID - fix assignment before reading results"
    elif primary_win and not rev_harm and aov_ok:
        verdict = "SHIP"
    elif primary_win:
        verdict = "DON'T SHIP - primary metric wins but a guardrail is breached"
    else:
        verdict = "DON'T SHIP - no significant lift on the primary metric"
    return {
        "verdict": verdict,
        "reasons": reasons,
        "srm_ok": srm_ok,
        "primary_win": primary_win,
        "revenue_harm": rev_harm,
        "aov_ok": aov_ok,
    }


def _plots(fig_dir, conv, rev, seq, aa, bayes, segments) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    ax = axes[0, 0]
    seg = pd.DataFrame(segments)
    labels = ["overall", *seg["segment"].tolist()] if len(seg) else ["overall"]
    diffs = [conv["details"]["diff"], *seg.get("diff", pd.Series(dtype=float)).tolist()]
    lo = [conv["ci_low"], *seg.get("ci_low", pd.Series(dtype=float)).tolist()]
    hi = [conv["ci_high"], *seg.get("ci_high", pd.Series(dtype=float)).tolist()]
    y = np.arange(len(labels))[::-1]
    ax.errorbar(
        diffs,
        y,
        xerr=[np.subtract(diffs, lo), np.subtract(hi, diffs)],
        fmt="o",
        capsize=4,
        color=BLUE,
    )
    ax.axvline(0, color=MUTED, lw=1)
    ax.set(
        yticks=y,
        yticklabels=labels,
        title="Conversion difference (T - C), 95% CI",
        xlabel="percentage points",
    )
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

    ax = axes[0, 1]
    names = ["raw", "CUPED"]
    width = [rev["ci_width_raw"], rev["ci_width_cuped"]]
    ax.bar(names, width, color=[MUTED, BLUE])
    ax.set(
        title=f"Revenue/user CI width (variance -{rev['cuped_stats']['variance_reduction']:.0%})",
        ylabel="95% CI width",
    )

    ax = axes[0, 2]
    ax.plot(seq["day"], seq["p_value"], marker="o")
    ax.axhline(0.05, color=INK_2, ls="--", lw=1, label="alpha = 0.05")
    ax.set(
        title="Sequential p-value by daily look",
        xlabel="day of test",
        ylabel="p-value",
        yscale="log",
    )
    ax.legend()

    ax = axes[1, 0]
    ax.hist(aa["min_p_values"], bins=40, color=MUTED)
    ax.axvline(0.05, color=INK_2, ls="--", lw=1)
    ax.set(
        title=f"A/A min p over {aa['looks']} looks: FPR {aa['fpr_any_look']:.0%} "
        f"(vs {aa['fpr_final_look_only']:.0%} single look)",
        xlabel="smallest p-value seen while peeking",
        ylabel="simulations",
    )

    ax = axes[1, 1]
    edges = np.array(bayes["lift_bin_edges"])
    ax.bar((edges[:-1] + edges[1:]) / 2, bayes["lift_histogram"], width=np.diff(edges), color=BLUE)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set(
        title=f"Posterior relative lift, P(T > C) = {bayes['prob_treatment_better']:.1%}",
        xlabel="relative lift in conversion",
    )
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

    ax = axes[1, 2]
    ax.plot(seq["day"], seq["conv_control"], marker="o", color=BLUE, label="control")
    ax.plot(seq["day"], seq["conv_treatment"], marker="o", color=ORANGE, label="treatment")
    ax.set(title="Cumulative conversion", xlabel="day of test", ylabel="converted users")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend()
    for a in axes.flat:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "experiment_summary.png", dpi=130)
    plt.close(fig)


def run_experiment(cfg: Config) -> dict[str, Any]:
    ex = cfg["experiment"]
    alpha = float(ex["alpha"])
    pop = load_population(cfg)
    out, figs = outputs_dir(cfg), figures_dir(cfg)

    # Design: size the test from the observed baseline before looking at any outcome.
    baseline = float(pop["label"].mean())
    kpis_path = out / "daily_kpis.csv"
    daily_users = float(pd.read_csv(kpis_path)["users"].mean()) if kpis_path.exists() else None
    plan = sample_size_two_proportions(
        baseline,
        float(ex["mde_relative"]),
        alpha,
        float(ex["power"]),
        daily_eligible_users=daily_users,
        treatment_share=float(ex["treatment_share"]),
    )
    n_arm_actual = int(len(pop) * min(ex["treatment_share"], 1 - ex["treatment_share"]))
    design = {
        **plan.to_dict(),
        "population": len(pop),
        "n_per_arm_available": n_arm_actual,
        "mde_at_available_n": minimum_detectable_effect(
            baseline, n_arm_actual, alpha, float(ex["power"])
        ),
        "power_at_available_n": achieved_power(
            baseline, float(ex["mde_relative"]), n_arm_actual, alpha
        ),
    }

    effect = EffectModel(
        float(ex["discount"]), float(ex["base_uplift"]), int(ex.get("target_top_deciles", 1))
    )
    # Pick the hash salt before looking at outcomes (balance on counts + pre-period data).
    salt_search = choose_salt(pop, str(ex["salt"]), float(ex["treatment_share"]))
    design["salt"] = salt_search["salt"]
    design["salt_search"] = salt_search
    df = simulate(pop, effect, salt_search["salt"], float(ex["treatment_share"]), cfg.seed)

    srm = check_srm(df, float(ex["treatment_share"])).to_dict()
    conv = conversion_test(df, alpha)
    rev = revenue_tests(df, alpha)
    cupac = conversion_cupac(df, alpha)
    aov = aov_guardrail(df, alpha)
    segments = segment_breakdown(df, alpha)
    bayes = bayesian_conversion(df, seed=cfg.seed)
    seq = sequential_pvalues(df)
    untreated = pop.assign(
        converted=pop["label"],
        purchase_day=np.where(
            pop["label"] == 1,
            (
                (
                    naive_utc(pop["first_purchase_time"]) - naive_utc(pop["cutoff"].to_numpy())
                ).dt.total_seconds()
                // 86400
            ).clip(0, 6)
            + 1,
            0,
        ).astype(int),
    )
    aa = peeking_simulation(untreated, int(ex["n_aa_simulations"]), cfg.seed, alpha=alpha)
    decision = decide(srm, conv, rev, aov, float(ex.get("max_aov_drop", 0.10)))

    _plots(figs, conv, rev, seq, aa, bayes, segments)
    arm_summary = (
        df.groupby("group")
        .agg(
            users=("user_id", "size"),
            converted=("converted", "sum"),
            conversion=("converted", "mean"),
            revenue=("revenue", "sum"),
            revenue_per_user=("revenue", "mean"),
            treated=("treated", "sum"),
            simulated_conversions=("simulated_conversion", "sum"),
        )
        .reset_index()
    )
    arm_summary.to_csv(out / "ab_arms.csv", index=False)
    pd.DataFrame(segments).to_csv(out / "ab_segments.csv", index=False)
    seq.to_csv(out / "ab_sequential.csv", index=False)
    result = {
        "design": design,
        "effect_model": effect.__dict__,
        "srm": srm,
        "conversion": conv,
        "conversion_cupac": cupac,
        "revenue": rev,
        "aov_guardrail": aov,
        "segments": segments,
        "bayesian": bayes,
        "peeking": {k: v for k, v in aa.items() if k != "min_p_values"},
        "decision": decision,
    }
    (out / "ab_results.json").write_text(json.dumps(result, indent=2, default=_default))
    log.info(
        "A/B: n=%d, conv %s, verdict: %s",
        len(df),
        f"{conv['details']['p_a']:.4f}->{conv['details']['p_b']:.4f}",
        decision["verdict"],
    )
    return result


def _default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    return str(o)
