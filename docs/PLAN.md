# E-commerce Experimentation & Propensity Platform - E2E Execution Plan

## Context
Build a college project (fits Data Science / Big Data / ML / Stats courses) that demonstrates every skill in the JD: Python, SQL, descriptive + inferential stats, ML, large-scale analysis + visualization, Hadoop/Hive/Spark/H2O, NoSQL, A/B testing, prototype models, and communication. Outcome: a public GitHub repo that runs end-to-end with one command, a dashboard, a scoring API, and a written report with quantified results.

Assumptions: Windows laptop with 16 GB RAM (8 GB works with a sampled dataset), Docker Desktop with WSL2, Python 3.11. Timeline: about 8 weeks at 8-10 hrs/week.

## Tech stack
- Storage: raw CSV -> Parquet (partitioned by month); HDFS via Docker (optional, for the "Hadoop" box)
- Processing: PySpark 3.5 (local mode, `local[*]`), Hive metastore via Spark SQL `enableHiveSupport()`
- SQL: Spark SQL + DuckDB for fast local analysis
- Stats: scipy, statsmodels
- ML: scikit-learn, XGBoost, H2O AutoML, SHAP, MLflow (experiment tracking)
- NoSQL: MongoDB (user profiles + scores)
- Serving: FastAPI
- Dashboard: Streamlit + Plotly
- Infra/quality: Docker Compose, Makefile, pytest, ruff, pre-commit, GitHub Actions CI

## Dataset
Kaggle "eCommerce behavior data from multi category store" (Oct-Nov 2019, ~110M events per month, ~14 GB raw).
Columns: event_time, event_type (view/cart/purchase), product_id, category_id, category_code, brand, price, user_id, user_session.
Dev workflow: build everything on a 5% user-level sample (`sample_users.py`), then run the full data once at the end for headline numbers.

## Repo structure
```
ecom-experimentation/
  docker-compose.yml        # spark, hive-metastore(postgres), hdfs (optional), mongo, mlflow
  Makefile                  # make setup | ingest | features | eda | train | abtest | serve | dashboard | all
  pyproject.toml
  configs/config.yaml       # paths, sample rate, seeds, model params
  data/{raw,bronze,silver,gold}/   # gitignored
  src/ecom/
    ingest/        load_raw.py, to_parquet.py, sample_users.py
    transform/     sessionize.py, build_features.py, labels.py
    sql/           funnel.sql, cohorts.sql, category_stats.sql, rfm.sql
    stats/         hypothesis_tests.py, power.py, cuped.py
    models/        train_baseline.py, train_xgb.py, train_h2o.py, evaluate.py, explain.py
    experiment/    simulate_ab.py, analyze_ab.py, srm_check.py
    serving/       load_mongo.py, api.py
  dashboard/app.py
  notebooks/       01_eda.ipynb, 02_stats.ipynb, 03_modeling.ipynb, 04_abtest.ipynb
  tests/           test_sessionize.py, test_features.py, test_stats.py, test_api.py
  reports/         report.md, business_memo.md, figures/
  README.md        # architecture diagram, results, how to run
```

## Phases

### Week 1 - Setup + ingestion (Python, Hadoop/Hive)
1. Create repo, pyproject, ruff, pre-commit, Makefile, GitHub Actions running ruff + pytest.
2. docker-compose: `bitnami/spark`, `mongo:7`, `postgres` (Hive metastore), `mlflow`; optional `bde2020/hadoop-namenode/datanode`.
3. `to_parquet.py`: read raw CSV with an explicit schema, cast types, parse timestamps, write Parquet partitioned by `event_date` (bronze).
4. Data quality checks: null rates, duplicate events, negative prices, event_type distribution. Log results to `reports/data_quality.md`.
5. Register external Hive tables over bronze/silver/gold so all later SQL works against `ecom.events`.
Deliverable: `make ingest` turns raw CSV into queryable Hive tables.

### Week 2 - Transformations + SQL analytics (Spark, SQL)
1. `sessionize.py`: per user_session, compute start/end, duration, #views, #carts, purchased flag, revenue (silver `sessions`).
2. `build_features.py` (gold `user_features`): RFM, views-to-cart ratio, cart abandonment rate, distinct categories, top category, avg price viewed, weekend/hour share, days active, price sensitivity.
3. SQL files (window functions, CTEs, joins):
   - `funnel.sql`: view -> cart -> purchase conversion overall and per category/brand
   - `cohorts.sql`: weekly first-seen cohorts x retention week matrix
   - `rfm.sql`: NTILE-based RFM segments
   - `category_stats.sql`: revenue, AOV, conversion by category with rank()
4. Benchmark: same aggregation in pandas vs Spark on full data (record time/memory for the report).
Deliverable: gold tables + SQL outputs saved as CSVs for the dashboard.

### Week 3 - EDA + descriptive stats (visualization, insights)
1. Notebook 01: distributions (price, session length - log scale), hourly/day-of-week heatmaps, funnel chart, cohort heatmap, Pareto of revenue by brand.
2. Write 5-8 concrete insights with numbers (e.g. "top 1% of users drive X% of revenue").
3. Save figures to `reports/figures/` for reuse in the dashboard and report.

### Week 4 - Inferential stats (hypothesis testing)
`hypothesis_tests.py` + notebook 02, each test with stated hypotheses, assumption checks, effect size, CI:
- Conversion weekend vs weekday: two-proportion z-test / chi-square
- AOV across top categories: ANOVA -> Kruskal-Wallis if non-normal, post-hoc Dunn
- Session duration buyers vs non-buyers: Mann-Whitney U
- Price vs conversion: Spearman correlation
- Bootstrap CIs for median AOV
- Multiple-comparison correction (Benjamini-Hochberg)
Unit tests in `test_stats.py` against known synthetic inputs.

### Weeks 5-6 - ML prototype (ML, H2O)
Task: predict whether a user purchases in the next 7 days given behavior in the prior 21 days (time-based split to avoid leakage: train on Oct, validate early Nov, test late Nov).
1. `labels.py`: label window construction, with a leakage test.
2. Baseline: Logistic Regression + standard scaling, class weights.
3. XGBoost with Optuna tuning; Spark MLlib GBT on full data as a scale comparison.
4. H2O AutoML (`h2o.automl`, max_runtime 20 min) as a benchmark leaderboard.
5. `evaluate.py`: ROC-AUC, PR-AUC (primary, since data is imbalanced), calibration curve, lift/gains chart, top-decile precision.
6. `explain.py`: SHAP summary + dependence plots; translate the top 5 drivers into plain English.
7. Track every run in MLflow; register the best model.
Deliverable: model comparison table + chosen model artifact.

### Week 7 - A/B experiment (experimentation)
Simulate a realistic experiment from held-out late-Nov users:
1. Design doc in `reports/report.md`: hypothesis, primary metric (conversion), guardrail (AOV, revenue/user), MDE, alpha=0.05, power=0.8.
2. `power.py`: required sample size (statsmodels `NormalIndPower`), test duration.
3. `simulate_ab.py`: randomize users 50/50 with a hashed user_id; control = no targeting; treatment = top-propensity-decile users get a simulated discount, with the effect modeled from observed price sensitivity (state assumptions explicitly).
4. `srm_check.py`: chi-square sample ratio mismatch check.
5. `analyze_ab.py`: two-proportion z-test, CI on lift, Welch t-test for revenue/user, CUPED using pre-period spend (show the variance reduction), segment breakdown, peeking pitfall demo (sequential p-value plot).
6. Decision: ship / don't ship with reasoning.

### Week 8 - Serving, dashboard, write-up (NoSQL, prototype, communication)
1. `load_mongo.py`: upsert user documents `{user_id, features, propensity, segment, updated_at}`, index on user_id and segment.
2. `api.py` (FastAPI): `GET /users/{id}/score`, `GET /segments/{name}/top?k=`, `GET /health`; pytest with TestClient.
3. `dashboard/app.py` (Streamlit) tabs: Overview KPIs, Funnel & Cohorts, Stats Tests, Model (metrics, SHAP), Experiment (lift + CI, CUPED before/after), User lookup (calls the API).
4. `README.md`: architecture diagram, results table, screenshots, one-command run.
5. `business_memo.md` (2 pages, non-technical) + 10-12 slide deck for the course presentation.
6. Final full-data run, freeze the headline numbers into the README and resume bullet.

## Stretch goals (pick 1-2)
- Airflow/Prefect DAG orchestrating the pipeline
- Bayesian A/B analysis (Beta-Binomial) next to the frequentist one
- Uplift modeling (T-learner) instead of plain propensity
- Deploy the dashboard on Streamlit Community Cloud using the sample data

## Verification (end to end)
1. `make setup && docker compose up -d`, check all containers are healthy.
2. `make all SAMPLE=0.05` runs ingest -> features -> train -> abtest -> load_mongo with no errors in under ~20 min.
3. `pytest` passes (sessionize, features, leakage, stats, API); `ruff check` clean; CI green on GitHub.
4. `curl localhost:8000/users/<id>/score` returns a JSON score matching MLflow's registered model output.
5. `streamlit run dashboard/app.py`: every tab renders, checked at desktop and narrow width, no broken charts.
6. Sanity checks: funnel rates monotonic, SRM p > 0.01, test PR-AUC above the baseline positive rate, CUPED variance lower than raw.
7. Fresh clone on another machine following only the README reproduces the results.
