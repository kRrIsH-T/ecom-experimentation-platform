# Handoff notes (for the next agent or contributor)

Read this first, then `docs/PLAN.md` (the original project plan) and
`graphify-out/GRAPH_REPORT.md` (knowledge graph of the codebase; query it with
`graphify query "<question>"`, see "Knowledge graph" below).

## What this project is
E-commerce experimentation & propensity platform (college project covering Python, SQL,
stats, ML, Spark/Hive/H2O, MongoDB, A/B testing, dashboard). One CLI drives everything:
`ecom <step>` (= `python -m ecom <step>`), defined in `src/ecom/cli.py`.

Pipeline order (`ecom all`): ingest -> transform -> features -> sql -> eda -> stats -> train ->
abtest -> load-mongo -> report. Extra commands: `setup`, `generate`, `download`, `benchmark`,
`mongo start|stop|status`, `serve`, `dashboard`, `notebooks`.

## Environment facts (this Windows machine)
- Python 3.11 venv at `.venv` (created with `uv venv --python 3.11`; install with
  `uv pip install --python .venv\Scripts\python.exe -e ".[dev]"`). System Python is 3.13 -
  PySpark 3.5 does not support it, so always use the venv.
- No system Java: `ecom setup` (or any Spark/H2O step) downloads Temurin JDK 17 and, on
  Windows, `winutils.exe` + `hadoop.dll` into `.tools/` (`src/ecom/tools.py`).
- **User requirement: nothing on C:.** All temp/spill/cache dirs are redirected to
  `.cache/` by `src/ecom/env.py`, which runs on `import ecom` (TMP/TEMP, SPARK_LOCAL_DIRS,
  java.io.tmpdir, Hive scratch dirs, H2O ice_root, matplotlib, numba, Jupyter). Keep it
  that way for any new tool you add.
- Docker Desktop's engine was broken (HTTP 500) and its disk image lives on C:, so the
  pipeline does not need Docker locally: MongoDB runs from a portable binary
  (`ecom mongo start`, `src/ecom/serving/mongo_local.py`) and MLflow uses
  `sqlite:///mlflow.db`. `docker-compose.yml` is an optional route for other machines.
- Kaggle CLI credentials are set up (user `krrisht`, `~/.kaggle`, access-token auth). The
  real 14 GB data is NOT downloaded locally (D: space, 16 GB RAM); real-data runs go through
  `ecom kaggle push|status|logs|pull` (`src/ecom/kaggle_job.py` local side,
  `src/ecom/kaggle_entry.py` Kaggle side) on the private kernel `krrisht/ecom-kaggle-run`
  with GPU T4 x2. `data.source: synthetic` stays the local default.
- `make` is not installed on Windows; the Makefile only wraps the CLI.

## Layout of generated data
`data/<layer>/<source>/<sample_tag>/<table>` where sample_tag is `s0500` for 5% or `full`
(`src/ecom/layout.py`). Hive tables live in database `ecom` (embedded Derby metastore in
`data/metastore`). Small results for the dashboard/reports: `data/outputs/<source>/<tag>/`.
Figures: `reports/figures/`.

## Status
See the checklist at the bottom; keep it current when you finish something.

## Gotchas found so far
- Kaggle image (Nov 2026): Python 3.13, PySpark 4, Java 21, xgboost 3 - all outside the
  project's supported range, so the kernel builds its own Python 3.11 venv with uv pinned to
  the pushing venv's versions (`constraints.txt` in the payload) and `ecom` fetches JDK 17.
  4 CPUs, 31 GB RAM, ~1 TB scratch (the run works in `/tmp/ecom-work`), dataset mounted at
  `/kaggle/input/datasets/mkechinov/...` (the runner searches `/kaggle/input` for the files).
- `ECOM_KAGGLE_INPUT/_WORK/_OUT/_PYTHON` let you dry-run the generated kernel script locally
  against fake Kaggle-format files before spending GPU quota.
- XGBoost trains on `ml.xgb_device` (default `cpu`; the Kaggle run uses `auto`) but the saved
  model is always switched to CPU so the API/dashboard can load it anywhere.
- PySpark `collect()` converts timestamps to the machine's local zone (IST here). Format
  timestamps inside Spark (`date_format`) or use `toPandas()` (respects session UTC).
- Parquet is written as `TIMESTAMP_MICROS` (UTC-adjusted). Older files used INT96; use
  `naive_utc()` in `experiment/simulate_ab.py` when mixing.
- SQLAlchemy 2.1 breaks MLflow 2.x -> pinned `<2.1`.
- PowerShell 5.1 `Set-Content -Encoding utf8` writes a BOM; config loader reads `utf-8-sig`.
- H2O's XGBoost is unavailable on Windows; AutoML skips it (expected).
- Probabilities must be calibrated: class weights inflate them and the A/B simulation uses
  propensity as a baseline probability -> sigmoid calibration on the valid snapshot.
- Never enable `spark.sql.sources.partitionOverwriteMode=dynamic` session-wide: the snapshot
  tables are written "overwrite first, append the rest", and dynamic mode kept the previous
  run's partitions (duplicated users, doubled positive rate). `step_features` now raises if
  `model_dataset` has duplicate (split, user_id) rows.
- H2O's server refuses to read files under dot-directories (default `-file_deny_glob`
  includes `**/.**`), so `H2OFrame(pandas_df)` fails with temp files in `.cache/`. Frames are
  sent with `h2o.upload_file` instead (`models/train_h2o.py`).
- Disk was nearly full on this machine at one point (D: 0.3 GB free). The MongoDB zip ships
  1.5 GB of `.pdb` debug symbols, so only `mongod.exe` is extracted; synthetic CSVs are written
  gzip-compressed in chunks (232 MB instead of 1.5 GB).
- The A/B result at a 5% sample is expected to be underpowered (the report says so); the full
  data run (`--sample 1.0`) has ~20x the users.

## Graphify notes
- Interpreter: `graphify-out/.graphify_python` (uv tool install on C:, pre-existing).
- `.graphifyignore` excludes generated figures, rendered reports and notebooks.
- `tree_sitter_sql` is not installed in that tool env, so `.sql` files are covered by the
  semantic (LLM) pass, not AST. Last update (incremental, after the full-data run): 649 nodes,
  1257 edges, 38 communities; health check clean (0 dangling / missing / self-loop edges).

## Knowledge graph
Built with the graphify skill into `graphify-out/` (`graph.json`, `GRAPH_REPORT.md`,
`graph.html`). Refresh it with `/graphify . --update` (AST + semantic for changed files).
Plain `graphify update .` (AST-only) refuses to write here because it would drop the
LLM-extracted SQL nodes (no `tree_sitter_sql` in the tool env) - do not pass `--force`.

## Checklist
- [x] Config, env/temp redirection, JDK/winutils bootstrap, Spark+Hive session
- [x] Synthetic generator (Kaggle format), ingest to bronze, data quality, user sampling
- [x] Silver (clean, sessions), gold (features, labels, model_dataset), SQL analytics
- [x] Stats tests, power, CUPED
- [x] Models: LR, XGB+Optuna, Spark GBT, H2O AutoML, SHAP, MLflow registry, scoring
- [x] A/B simulation + analysis (SRM, CUPED/CUPAC, Bayesian, peeking)
- [x] Mongo loader, FastAPI service, local Mongo manager
- [x] EDA figures + insights
- [x] Benchmark code (pandas vs DuckDB vs Spark) - run it: `ecom benchmark`
- [x] Report templates (`reports/templates`), notebook builder, dashboard code
- [x] Makefile, docker-compose, Dockerfile, CI, pre-commit, README (compose untested: Docker
      engine broken on this machine)
- [x] Knowledge graph built (graphify)
- [x] Full pipeline at 5% end to end; benchmark run (pandas 17.5 s, DuckDB 0.7 s, Spark 4.5 s)
- [x] Full-data run (`ecom --sample 1.0 all`) end to end; README/report numbers come from it
- [x] Verified: pytest 51/51 (incl. Spark integration), ruff clean, API curl incl. live score
      == MLflow champion, dashboard all tabs at 1440px and 420px (Playwright)
- [x] `docker compose config` validates (image build untested: Docker engine broken here)
- [x] Notebooks executed on full data (`ECOM__DATA__SAMPLE_RATE=1.0 ecom --sample 1.0 notebooks`)
- [ ] Optional stretch: uplift (T-learner) model; Prefect/Airflow DAG

## How to resume
1. `ecom mongo start` (MongoDB is not a service; it stops on reboot).
2. Full-data runs on this machine need less memory than the defaults because other apps hold
   most of the 16 GB RAM and the page file cannot grow (C: nearly full):
   `ECOM__SPARK__DRIVER_MEMORY=3g ECOM__SPARK__MASTER=local[4] ECOM__ML__H2O_MAX_MEM=3G`.
   Without them Spark dies with "The paging file is too small".
3. `ecom serve` / `ecom dashboard` for the API (:8000) and dashboard (:8501). The API re-checks
   the MLflow `champion` alias every 60 s, so a retrain is picked up without a restart.

## Experiment design note
At full data the configured salt `promo-2019-11` gave a 2.8-sigma unbalanced split (SRM
p = 0.005) - a genuine unlucky hash draw (other salts are balanced). Instead of hand-picking a
salt, `choose_salt()` (experiment/simulate_ab.py) does an outcome-blind pre-launch "seedfinder"
search on counts + pre-period covariates; the chosen salt is recorded in `ab_results.json`.
