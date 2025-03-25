"""Build (and execute) the four analysis notebooks from code, so they always match the package.

Each notebook works live against the pipeline outputs and the package functions rather than
duplicating logic: notebooks are for narrative and exploration, ``src/ecom`` for the logic.
"""

from __future__ import annotations

import logging

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from ecom.config import Config

log = logging.getLogger(__name__)

SETUP = """\
import json
import pandas as pd
from IPython.display import Image, Markdown, display
from ecom.config import get_config
from ecom.layout import outputs_dir, table_path
cfg = get_config()
OUT = outputs_dir(cfg)
FIG = cfg.paths.figures
pd.set_option("display.float_format", "{:,.4f}".format)
print(f"source={cfg.source} sample={cfg.sample_rate:.0%} outputs={OUT}")"""


def _nb(title: str, intro: str, cells: list[tuple[str, str]]) -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = [new_markdown_cell(f"# {title}\n\n{intro}"), new_code_cell(SETUP)]
    for kind, src in cells:
        nb.cells.append(new_markdown_cell(src) if kind == "md" else new_code_cell(src))
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    return nb


def eda_notebook() -> nbformat.NotebookNode:
    return _nb(
        "01 - Exploratory data analysis",
        "Distributions, time patterns, funnel, cohorts and revenue concentration. "
        "Figures are produced by `ecom eda` (`src/ecom/analysis/eda.py`).",
        [
            ("md", "## Headline KPIs (Spark SQL `kpis.sql`)"),
            ("code", 'pd.read_csv(OUT / "overview.csv").T'),
            ("md", "## Ad-hoc SQL with DuckDB straight on the silver Parquet"),
            (
                "code",
                "from ecom.stats.hypothesis_tests import connect\ncon = connect(cfg)\n"
                "con.execute('''\n"
                "    SELECT brand, COUNT(*) AS purchases, ROUND(SUM(price)) AS revenue\n"
                "    FROM events_clean WHERE event_type = 'purchase' AND brand <> 'unknown'\n"
                "    GROUP BY brand ORDER BY revenue DESC LIMIT 10\n''').df()",
            ),
            ("md", "## Distributions (log scale)"),
            ("code", 'display(Image(FIG / "eda_distributions.png"))'),
            ("md", "## When do people shop?"),
            (
                "code",
                'display(Image(FIG / "eda_hour_dow.png"))\ndisplay(Image(FIG / "eda_daily.png"))',
            ),
            ("md", "## Funnel and cohorts"),
            (
                "code",
                'display(pd.read_csv(OUT / "funnel_overall.csv"))\n'
                'display(Image(FIG / "eda_funnel.png"))\ndisplay(Image(FIG / "eda_cohorts.png"))',
            ),
            ("md", "## Revenue concentration"),
            (
                "code",
                'display(pd.read_csv(OUT / "revenue_concentration.csv"))\n'
                'display(Image(FIG / "eda_pareto.png"))',
            ),
            ("md", "## Insights"),
            (
                "code",
                'ins = json.loads((OUT / "eda_insights.json").read_text())["insights"]\n'
                'display(Markdown("\\n".join(f"{i}. {s}" for i, s in enumerate(ins, 1))))',
            ),
        ],
    )


def stats_notebook() -> nbformat.NotebookNode:
    return _nb(
        "02 - Inferential statistics",
        "Hypotheses, assumption checks, effect sizes, CIs and BH correction "
        "(`src/ecom/stats`). The first cells re-run one test by hand to show the mechanics.",
        [
            ("md", "## Weekend vs weekday conversion, step by step"),
            (
                "code",
                "from ecom.stats.core import two_proportion_ztest\n"
                "from ecom.stats.hypothesis_tests import connect\ncon = connect(cfg)\n"
                "t = con.execute('SELECT is_weekend, COUNT(*) n, SUM(purchased::INT) x "
                "FROM sessions GROUP BY 1').df().set_index('is_weekend')\nt",
            ),
            (
                "code",
                "res = two_proportion_ztest(int(t.loc[False, 'x']), int(t.loc[False, 'n']),\n"
                "                          int(t.loc[True, 'x']), int(t.loc[True, 'n']))\n"
                "print(res.h0, '| z =', round(res.statistic, 2), '| p =', f'{res.p_value:.2g}')\n"
                "print('diff CI:', round(res.ci_low, 4), round(res.ci_high, 4), "
                "'| relative lift:', f\"{res.details['relative_lift']:+.1%}\")",
            ),
            ("md", "## All tests (BH-adjusted)"),
            ("code", 'pd.read_csv(OUT / "stats_tests.csv")'),
            ("md", "## Assumption checks and post-hoc tests"),
            (
                "code",
                'tests = json.loads((OUT / "stats_tests.json").read_text())["tests"]\n'
                "for t in tests:\n    print(t['name'], '->', t['test'])\n"
                "    print('   assumptions:', t['assumptions'])",
            ),
            (
                "code",
                "aov = next(t for t in tests if t['name'].startswith('Order value'))\n"
                "pd.DataFrame(aov['details']['posthoc_dunn_bh'])",
            ),
            ("md", "## Bootstrap CIs for the median order value"),
            ("code", 'pd.read_csv(OUT / "bootstrap_median_aov.csv")'),
        ],
    )


def modeling_notebook() -> nbformat.NotebookNode:
    return _nb(
        "03 - Propensity modelling",
        "Time-based snapshots, leakage-safe labels, model comparison, calibration and "
        "SHAP (`src/ecom/models`).",
        [
            (
                "code",
                "from ecom.models.data import load_splits\nsplits = load_splits(cfg)\n"
                "pd.DataFrame({s: {'users': len(getattr(splits, s)), "
                "'positive_rate': getattr(splits, s).label.mean()} "
                "for s in ('train', 'valid', 'test')}).T",
            ),
            ("md", "## Load the registered champion from MLflow and re-score the test snapshot"),
            (
                "code",
                "import mlflow\nfrom ecom.models.evaluate import classification_metrics\n"
                "champ = json.loads((OUT / 'champion.json').read_text())\n"
                "mlflow.set_tracking_uri(cfg.mlflow_uri)\n"
                "model = mlflow.sklearn.load_model(champ['uri'])\n"
                "X, y = splits.xy('test')\n"
                "pd.Series(classification_metrics(y, model.predict_proba(X)[:, 1]))",
            ),
            ("md", "## Model comparison"),
            ("code", 'pd.read_csv(OUT / "model_comparison.csv")'),
            (
                "code",
                'display(Image(FIG / "model_curves.png"))\ndisplay(Image(FIG / "model_gains.png"))',
            ),
            ("md", "## Drivers (SHAP)"),
            (
                "code",
                'display(Image(FIG / "shap_summary.png"))\n'
                'for d in json.loads((OUT / "model_drivers.json").read_text()):\n'
                "    print('-', d['text'])",
            ),
            ("md", "## H2O AutoML leaderboard"),
            ("code", 'pd.read_csv(OUT / "h2o_leaderboard.csv").head(10)'),
        ],
    )


def abtest_notebook() -> nbformat.NotebookNode:
    return _nb(
        "04 - A/B experiment",
        "Design (power), simulated targeted-coupon test, SRM, CUPED, Bayesian analysis "
        "and the peeking pitfall (`src/ecom/experiment`).",
        [
            ("md", "## How many users do we need? (power curve)"),
            (
                "code",
                "import matplotlib.pyplot as plt\nimport ecom.viz  # house style\n"
                "from ecom.stats.power import sample_size_two_proportions\n"
                "ab = json.loads((OUT / 'ab_results.json').read_text())\n"
                "base = ab['design']['baseline_rate']\nmdes = [0.05, 0.075, 0.1, 0.15, 0.2, 0.3]\n"
                "n = [sample_size_two_proportions(base, m).n_per_arm for m in mdes]\n"
                "fig, ax = plt.subplots(figsize=(7, 3.5))\nax.plot([m * 100 for m in mdes], n, "
                "marker='o')\nax.set(xlabel='relative MDE (%)', ylabel='users per arm', "
                "yscale='log', title=f'Sample size at baseline {base:.2%}')\nplt.show()",
            ),
            ("code", "pd.Series(ab['design'])"),
            ("md", "## Results"),
            (
                "code",
                'display(Image(FIG / "experiment_summary.png"))\n'
                'display(pd.read_csv(OUT / "ab_arms.csv"))\n'
                'display(pd.read_csv(OUT / "ab_segments.csv"))',
            ),
            (
                "code",
                "print('SRM p =', round(ab['srm']['p_value'], 4))\n"
                "print('CUPED variance reduction:', "
                "f\"{ab['revenue']['cuped_stats']['variance_reduction']:.1%}\")\n"
                "print('P(treatment better):', "
                "f\"{ab['bayesian']['prob_treatment_better']:.1%}\")\n"
                "print('Peeking FPR:', f\"{ab['peeking']['fpr_any_look']:.1%}\", 'vs', "
                "f\"{ab['peeking']['fpr_final_look_only']:.1%}\")",
            ),
            ("md", "## Decision"),
            (
                "code",
                "display(Markdown(f\"**{ab['decision']['verdict']}**\\n\\n\" + "
                "'\\n'.join('- ' + r for r in ab['decision']['reasons'])))",
            ),
        ],
    )


NOTEBOOKS = {
    "01_eda.ipynb": eda_notebook,
    "02_stats.ipynb": stats_notebook,
    "03_modeling.ipynb": modeling_notebook,
    "04_abtest.ipynb": abtest_notebook,
}


def build_notebooks(cfg: Config, execute: bool = True) -> None:
    from nbconvert.preprocessors import ExecutePreprocessor

    out_dir = cfg.root / "notebooks"
    out_dir.mkdir(exist_ok=True)
    for name, factory in NOTEBOOKS.items():
        nb = factory()
        if execute:
            ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(
                nb, {"metadata": {"path": str(cfg.root)}}
            )
        nbformat.write(nb, out_dir / name)
        log.info("Wrote notebooks/%s%s", name, " (executed)" if execute else "")
