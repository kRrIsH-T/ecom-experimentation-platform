"""Command-line entry point: ``ecom <step>`` (or ``python -m ecom <step>``).

Every pipeline stage is a sub-command; ``ecom all`` runs them in order in one process so
the Spark session and its Hive metastore are created once.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from collections.abc import Callable

from ecom.config import Config, get_config

log = logging.getLogger("ecom")

PIPELINE = [
    "ingest",
    "transform",
    "features",
    "sql",
    "eda",
    "stats",
    "train",
    "abtest",
    "load-mongo",
    "report",
]


# --- step implementations (imports are lazy so `ecom --help` stays instant) -----------------


def _spark(cfg: Config):
    from ecom.spark import get_spark

    return get_spark(cfg, "ecom-pipeline")


def cmd_setup(cfg: Config, args) -> None:
    from ecom.serving.mongo_local import ensure_mongo_binary
    from ecom.tools import ensure_runtime

    cfg.paths.ensure()
    ensure_runtime(cfg)
    log.info("Java and Hadoop helpers ready under %s", cfg.paths.tools)
    if not args.skip_mongo:
        log.info("MongoDB binary: %s", ensure_mongo_binary(cfg))


def cmd_generate(cfg: Config, args) -> None:
    from ecom.ingest.synthetic import generate

    summary = generate(cfg, n_users=args.n_users)
    log.info("Synthetic data: %s", json.dumps(summary.__dict__))


def cmd_download(cfg: Config, args) -> None:
    from ecom.ingest.load_raw import download_kaggle

    log.info("Downloaded: %s", download_kaggle(cfg))


def cmd_ingest(cfg: Config, args) -> None:
    from ecom.pipeline import step_ingest

    step_ingest(_spark(cfg), cfg, force=args.force)


def cmd_transform(cfg: Config, args) -> None:
    from ecom.pipeline import register_events, step_transform

    spark = _spark(cfg)
    register_events(spark, cfg)
    step_transform(spark, cfg)


def cmd_features(cfg: Config, args) -> None:
    from ecom.pipeline import ensure_tables, step_features

    spark = _spark(cfg)
    ensure_tables(spark, cfg)
    step_features(spark, cfg)


def cmd_sql(cfg: Config, args) -> None:
    from ecom.pipeline import ensure_tables, step_sql

    spark = _spark(cfg)
    ensure_tables(spark, cfg)
    step_sql(spark, cfg)


def cmd_benchmark(cfg: Config, args) -> None:
    from ecom.analysis.benchmark import run_benchmark

    run_benchmark(cfg)


def cmd_eda(cfg: Config, args) -> None:
    from ecom.analysis.eda import run_eda

    run_eda(cfg)


def cmd_stats(cfg: Config, args) -> None:
    from ecom.stats.hypothesis_tests import run_all_tests

    run_all_tests(cfg)


def cmd_train(cfg: Config, args) -> None:
    from ecom.models.train import run_training

    run_training(cfg, skip=set(args.skip or []))


def cmd_abtest(cfg: Config, args) -> None:
    from ecom.experiment.run import run_experiment

    run_experiment(cfg)


def cmd_evaluate_model(cfg: Config, args) -> None:
    from pathlib import Path

    from ecom.models.external_eval import evaluate_saved_model

    evaluate_saved_model(cfg, Path(args.model), args.name or Path(args.model).name)


def cmd_kaggle(cfg: Config, args) -> None:
    from ecom import kaggle_job

    {
        "push": kaggle_job.push,
        "status": kaggle_job.status,
        "logs": kaggle_job.logs,
        "pull": kaggle_job.pull,
    }[args.action](cfg)


def cmd_load_mongo(cfg: Config, args) -> None:
    from ecom.serving.load_mongo import load_users

    load_users(cfg)


def cmd_mongo(cfg: Config, args) -> None:
    from ecom.serving.mongo_local import start_mongo, status_mongo, stop_mongo

    {"start": start_mongo, "stop": stop_mongo, "status": status_mongo}[args.action](cfg)


def cmd_serve(cfg: Config, args) -> None:
    import uvicorn

    uvicorn.run("ecom.serving.api:app", host=args.host, port=args.port, reload=False)


def cmd_dashboard(cfg: Config, args) -> None:
    app = cfg.root / "dashboard" / "app.py"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app),
            "--server.headless",
            "true",
            "--server.port",
            str(args.port),
            "--browser.gatherUsageStats",
            "false",
        ],
        check=True,
    )


def cmd_report(cfg: Config, args) -> None:
    from ecom.reporting import render_reports

    render_reports(cfg)


def cmd_notebooks(cfg: Config, args) -> None:
    from ecom.notebooks import build_notebooks

    build_notebooks(cfg, execute=not args.no_execute)


def cmd_all(cfg: Config, args) -> None:
    steps = [s for s in PIPELINE if s not in set(args.skip or [])]
    timings = {}
    for step in steps:
        t0 = time.perf_counter()
        log.info("=== %s ===", step)
        COMMANDS[step][0](cfg, args)
        timings[step] = round(time.perf_counter() - t0, 1)
    log.info("Pipeline finished: %s (total %.0fs)", timings, sum(timings.values()))
    from ecom.layout import outputs_dir

    # Merge so a partial re-run (--skip) keeps the timings of the steps it did not run.
    path = outputs_dir(cfg) / "pipeline_timings.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    merged = {s: timings.get(s, previous.get(s)) for s in PIPELINE if s in timings or s in previous}
    path.write_text(json.dumps(merged, indent=2))


COMMANDS: dict[str, tuple[Callable, str]] = {
    "setup": (cmd_setup, "download JDK 17 (+winutils on Windows) and MongoDB into .tools/"),
    "generate": (cmd_generate, "generate the synthetic Kaggle-format CSVs"),
    "download": (cmd_download, "download the real Kaggle dataset (needs credentials)"),
    "ingest": (cmd_ingest, "raw CSV -> bronze Parquet + data quality + user sample + Hive"),
    "transform": (cmd_transform, "bronze -> silver events_clean and sessions"),
    "features": (cmd_features, "gold user_features, labels and model_dataset"),
    "sql": (cmd_sql, "run the SQL analytics (funnel, cohorts, RFM, categories, KPIs)"),
    "benchmark": (cmd_benchmark, "pandas vs DuckDB vs Spark aggregation benchmark"),
    "eda": (cmd_eda, "EDA figures and insights"),
    "stats": (cmd_stats, "hypothesis tests with effect sizes, CIs and BH correction"),
    "train": (cmd_train, "baseline, XGBoost+Optuna, Spark GBT, H2O AutoML, SHAP, MLflow"),
    "abtest": (cmd_abtest, "power analysis, simulated A/B test, SRM, CUPED, Bayesian, peeking"),
    "evaluate-model": (cmd_evaluate_model, "score a saved MLflow model on this source's snapshots"),
    "kaggle": (cmd_kaggle, "push|status|logs|pull a GPU run on the real data in a Kaggle kernel"),
    "load-mongo": (cmd_load_mongo, "upsert user profiles + scores into MongoDB"),
    "mongo": (cmd_mongo, "start/stop/status of the local MongoDB server"),
    "serve": (cmd_serve, "run the FastAPI scoring service"),
    "dashboard": (cmd_dashboard, "run the Streamlit dashboard"),
    "report": (cmd_report, "render reports/*.md from templates with the latest results"),
    "notebooks": (cmd_notebooks, "build (and execute) the analysis notebooks"),
    "all": (cmd_all, "run the whole pipeline end to end"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ecom", description=__doc__)
    parser.add_argument("--sample", type=float, help="user sample rate (default from config)")
    parser.add_argument("--source", choices=["synthetic", "kaggle"], help="data source")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (_, help_text) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text)
        if name in {"ingest", "all"}:
            p.add_argument("--force", action="store_true", help="rebuild bronze from raw CSV")
        if name in {"train", "all"}:
            p.add_argument("--skip", nargs="*", help="steps/models to skip (e.g. h2o spark_gbt)")
        if name == "generate":
            p.add_argument("--n-users", type=int)
        if name == "setup":
            p.add_argument("--skip-mongo", action="store_true")
        if name == "evaluate-model":
            p.add_argument("--model", required=True, help="MLflow model directory")
            p.add_argument("--name", help="label used in the results")
        if name == "kaggle":
            p.add_argument("action", choices=["push", "status", "logs", "pull"])
        if name == "mongo":
            p.add_argument("action", choices=["start", "stop", "status"])
        if name == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8000)
        if name == "dashboard":
            p.add_argument("--port", type=int, default=8501)
        if name == "notebooks":
            p.add_argument("--no-execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("py4j", "urllib3", "matplotlib", "h2o", "numba", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = get_config().with_overrides(
        **{"data.sample_rate": args.sample, "data.source": args.source}
    )
    cfg.paths.ensure()
    COMMANDS[args.command][0](cfg, args)


if __name__ == "__main__":
    main()
