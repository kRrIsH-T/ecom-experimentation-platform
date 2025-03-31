# Graph Report - smaiproj  (2026-11-01)

## Corpus Check
- 17 files · ~27,676 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 649 nodes · 1257 edges · 38 communities (32 shown, 6 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 135 edges (avg confidence: 0.89)
- Token cost: 71,588 input · 0 output

## Community Hubs (Navigation)
- Configuration Loading
- Benchmark and Memory Profiling
- Statistical Tests Core
- CLI Commands
- Features, Labels and Spark Tests
- Experiment Orchestration
- Synthetic Data and Raw Loading
- FastAPI Scoring Service
- Streamlit Dashboard
- MongoDB Loader
- A/B Analysis Methods
- Plan and SQL Analytics
- EDA Figures
- Modeling Decisions and Gotchas
- Central Config Settings
- Data Platform (Spark, Hive)
- A/B Design and Validity
- Container Services
- Local Environment Workarounds
- Notebook Builder
- Power Analysis
- Serving and Communication
- CI and Tooling Bootstrap
- Pipeline Conventions
- Handoff and Knowledge Graph
- Path type
- Python 3.11 Requirement
- Config BOM Handling
- Project Package
- date type
- pytest fixture

## God Nodes (most connected - your core abstractions)
1. `Config` - 64 edges
2. `TestResult` - 17 edges
3. `generate()` - 15 edges
4. `outputs_dir()` - 13 edges
5. `table_path()` - 13 edges
6. `run_all_tests()` - 13 edges
7. `run_experiment()` - 13 edges
8. `two_proportion_ztest()` - 12 edges
9. `make_events()` - 11 edges
10. `run_eda()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `rfm_users (table)` --shares_data_with--> `MongoDB user profile documents`  [INFERRED]
  src/ecom/sql/rfm.sql → README.md
- `Hive database ecom with embedded Derby metastore` --semantically_similar_to--> `Hive metastore (ecom.* tables)`  [INFERRED] [semantically similar]
  HANDOFF.md → README.md
- `Sigmoid probability calibration on valid snapshot` --semantically_similar_to--> `Platt-calibrated scores`  [INFERRED] [semantically similar]
  HANDOFF.md → README.md
- `End-to-end verification checklist` --references--> `funnel_overall`  [INFERRED]
  docs/PLAN.md → src/ecom/sql/funnel.sql
- `Week 4 - Inferential stats (hypothesis tests)` --conceptually_related_to--> `hour_dow`  [INFERRED]
  docs/PLAN.md → src/ecom/sql/kpis.sql

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **End-to-end data -> models -> experiment -> serving flow** — readme_medallion_architecture, readme_hive_metastore, readme_spark_sql_analytics, readme_propensity_models, readme_mlflow_registry, readme_ab_test_simulation, readme_mongodb_user_profiles, readme_fastapi_scoring_api, readme_streamlit_dashboard [EXTRACTED 1.00]
- **Workarounds driven by the nothing-on-C: constraint** — handoff_no_c_drive_requirement, handoff_cache_redirection, handoff_docker_free_local_setup, handoff_low_memory_spark_settings, handoff_disk_space_mitigation, handoff_h2o_upload_file_workaround [INFERRED 0.85]
- **A/B experiment validity safeguards** — readme_ab_test_simulation, readme_srm_check, readme_cuped, readme_hash_based_sampling_assignment, handoff_choose_salt_seedfinder, readme_peeking_simulation [INFERRED 0.85]
- **Compose services built from the shared x-app template** — docker_compose_x_app, docker_compose_pipeline, docker_compose_api, docker_compose_dashboard, docker_compose_mlflow [EXTRACTED 1.00]
- **ecom sql analytics suite over Hive silver tables** — src_ecom_sql_funnel, src_ecom_sql_cohorts, src_ecom_sql_rfm, src_ecom_sql_category_stats, src_ecom_sql_kpis [EXTRACTED 1.00]
- **Revenue Pareto / concentration analyses** — src_ecom_sql_category_stats_category_stats, src_ecom_sql_category_stats_brand_pareto, src_ecom_sql_kpis_revenue_concentration [INFERRED 0.85]

## Communities (38 total, 6 thin omitted)

### Community 0 - "Configuration Loading"
Cohesion: 0.06
Nodes (50): PathLike, _apply_env_overrides(), Config, find_root(), get_config(), load_config(), Paths, Any (+42 more)

### Community 1 - "Benchmark and Memory Profiling"
Cohesion: 0.06
Nodes (50): _duckdb(), _pandas(), PeakRSS, Same aggregation in pandas, DuckDB and Spark on the full bronze events. Query:…, Context manager sampling the RSS of this process and all its children., run_benchmark(), _spark_runner(), DataFrame (+42 more)

### Community 2 - "Statistical Tests Core"
Cohesion: 0.07
Nodes (53): ArrayLike, DuckDBPyConnection, check_srm(), DataFrame, Sample-ratio-mismatch check: is the observed split consistent with the design?, anova_or_kruskal(), benjamini_hochberg(), bootstrap_ci() (+45 more)

### Community 3 - "CLI Commands"
Cohesion: 0.09
Nodes (43): ArgumentParser, date, build_parser(), cmd_abtest(), cmd_all(), cmd_benchmark(), cmd_dashboard(), cmd_download() (+35 more)

### Community 4 - "Features, Labels and Spark Tests"
Cohesion: 0.07
Nodes (37): datetime, fixture, build_user_features(), feature_window(), _is(), Column, DataFrame, date (+29 more)

### Community 5 - "Experiment Orchestration"
Cohesion: 0.10
Nodes (36): Parquet TIMESTAMP_MICROS vs legacy INT96, Gotcha: PySpark collect() converts timestamps to local zone, parametrize, Series, decide(), _default(), load_population(), _plots() (+28 more)

### Community 6 - "Synthetic Data and Raw Loading"
Cohesion: 0.09
Nodes (36): datetime64, DatetimeIndex, Disk space mitigations (mongod-only extraction, gzip chunked CSVs), Synthetic data source (data.source: synthetic), Seeded synthetic generator (Kaggle schema), slice, _csv_files(), download_kaggle() (+28 more)

### Community 7 - "FastAPI Scoring Service"
Cohesion: 0.11
Nodes (30): BaseModel, CollectionDep, description, ge, get, le, ModelDep, post (+22 more)

### Community 8 - "Streamlit Dashboard"
Cohesion: 0.09
Nodes (25): cache_data, available_runs(), fig_layout(), DataFrame, Streamlit dashboard: KPIs, funnel & cohorts, statistics, model, experiment,…, read_csv(), read_json(), Figure (+17 more)

### Community 9 - "MongoDB Loader"
Cohesion: 0.12
Nodes (20): Collection, MongoClient, build_documents(), _clean(), ensure_indexes(), get_collection(), load_users(), Any (+12 more)

### Community 10 - "A/B Analysis Methods"
Cohesion: 0.16
Nodes (24): aov_guardrail(), _arms(), bayesian_conversion(), conversion_cupac(), conversion_test(), peeking_simulation(), Any, DataFrame (+16 more)

### Community 11 - "Plan and SQL Analytics"
Cohesion: 0.12
Nodes (17): PLAN.md (E2E execution plan), Stretch goals (Airflow, Bayesian A/B, uplift, Streamlit Cloud), Tech stack (PySpark, Hive, DuckDB, H2O, MongoDB, FastAPI, Streamlit), End-to-end verification checklist, Week 2 - Transformations + SQL analytics, Week 4 - Inferential stats (hypothesis tests), brand_pareto, category_stats (+9 more)

### Community 12 - "EDA Figures"
Cohesion: 0.24
Nodes (14): plot_cohorts(), plot_distributions(), plot_funnel(), plot_pareto(), plot_time_patterns(), DataFrame, Exploratory analysis: distribution, time-pattern, funnel, cohort and Pareto…, _read() (+6 more)

### Community 13 - "Modeling Decisions and Gotchas"
Cohesion: 0.18
Nodes (12): labels (feature_days, label_days, splits, scoring_cutoff), ml settings (optuna, h2o, spark_gbt, registered_model), Weeks 5-6 - ML propensity prototype, H2O XGBoost unavailable on Windows, model_dataset duplicate (split, user_id) guard in step_features, Gotcha: dynamic partitionOverwriteMode duplicates snapshots, Stretch goal: uplift T-learner model, Prefect/Airflow DAG, Time-based splits and point-in-time features (+4 more)

### Community 14 - "Central Config Settings"
Cohesion: 0.18
Nodes (11): config.yaml (central configuration), api url, data (source, sample_rate, kaggle_dataset), experiment settings (salt, alpha, power, mde, discount), mongo settings (uri, db, collection), synthetic generator settings, Week 7 - Simulated A/B experiment, Sigmoid probability calibration on valid snapshot (+3 more)

### Community 15 - "Data Platform (Spark, Hive)"
Cohesion: 0.20
Nodes (10): spark settings (local[*], hive), Hive database ecom with embedded Derby metastore, Benchmark: pandas vs DuckDB vs Spark, Benjamini-Hochberg correction, Hive metastore (ecom.* tables), Inferential stats tests, Medallion Parquet layers (bronze/silver/gold), Spark SQL analytics (funnel, cohorts, RFM, categories, KPIs) (+2 more)

### Community 16 - "A/B Design and Validity"
Cohesion: 0.22
Nodes (11): A/B underpowered at 5% sample, Outcome-blind seedfinder salt search (choose_salt), SRM from salt promo-2019-11 (2.8-sigma, p=0.005), Simulated targeted-coupon A/B test, Bayesian A/B analysis, CUPED/CUPAC variance reduction, User-level hash-based sampling and assignment, Peeking simulation (+3 more)

### Community 17 - "Container Services"
Cohesion: 0.39
Nodes (9): mlflow settings (tracking_uri, experiment), docker-compose.yml (optional container route), api service (ecom serve), dashboard service (ecom dashboard), HDFS datanode (hadoop profile), mlflow ui service, mongo service (mongo:7), HDFS namenode (hadoop profile) (+1 more)

### Community 18 - "Local Environment Workarounds"
Cohesion: 0.22
Nodes (9): Temp/cache redirection to .cache/ on import ecom, Docker-free local setup, H2O frames sent via h2o.upload_file, Low-memory full-data run settings (driver 3g, local[4], H2O 3G), MLflow sqlite backend (sqlite:///mlflow.db), User requirement: nothing on C:, Portable MongoDB binary (ecom mongo start), SQLAlchemy pinned <2.1 for MLflow 2.x (+1 more)

### Community 19 - "Notebook Builder"
Cohesion: 0.44
Nodes (8): NotebookNode, abtest_notebook(), build_notebooks(), eda_notebook(), modeling_notebook(), _nb(), Build (and execute) the four analysis notebooks from code, so they always match…, stats_notebook()

### Community 20 - "Power Analysis"
Cohesion: 0.25
Nodes (6): minimum_detectable_effect(), PowerPlan, Experiment sizing: required sample size, achievable MDE and test duration., Users per arm to detect a relative lift ``mde_relative`` with a two-sided…, Smallest relative lift detectable with ``n_per_arm`` users per arm., sample_size_two_proportions()

### Community 21 - "Serving and Communication"
Cohesion: 0.36
Nodes (8): Week 8 - Serving, dashboard, write-up, API re-checks MLflow champion alias every 60 s, README.md (E-commerce Experimentation & Propensity Platform), FastAPI scoring API (/health, /users/{id}/score, /segments, /score), MLflow runs + registry, MongoDB user profile documents, Streamlit dashboard, One visual system (colour-blind-safe palette, no dual axes)

### Community 22 - "CI and Tooling Bootstrap"
Cohesion: 0.33
Nodes (7): CI workflow, CI test job (ruff + pytest incl. Spark, Java 17), pre-commit config, pre-commit-hooks (whitespace, yaml, toml, large files), ruff + ruff-format pre-commit hooks, Week 1 - Setup + ingestion, Temurin JDK 17 + winutils bootstrap into .tools/

### Community 23 - "Pipeline Conventions"
Cohesion: 0.38
Nodes (7): paths (raw/bronze/silver/gold/outputs/warehouse/metastore), pipeline service (ecom --sample 0.05 all), HANDOFF.md (handoff notes), Data layout data/<layer>/<source>/<sample_tag>/<table>, ecom CLI (ecom <step>), Pipeline Order (ingest -> transform -> features -> sql -> eda -> stats -> train -> abtest -> load-mongo -> report), CI workflow (lint + full test suite)

### Community 24 - "Handoff and Knowledge Graph"
Cohesion: 0.40
Nodes (5): Graphify knowledge graph (graphify-out/), Handoff Notes, tree_sitter_sql missing - SQL covered by semantic pass, E-commerce Experimentation & Propensity Platform, Reports rendered from results

## Knowledge Gaps
- **20 isolated node(s):** `cohort_buyer_retention`, `daily_kpis`, `overview`, `Stretch goals (Airflow, Bayesian A/B, uplift, Streamlit Cloud)`, `Tech stack (PySpark, Hive, DuckDB, H2O, MongoDB, FastAPI, Streamlit)` (+15 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Config` connect `Configuration Loading` to `Benchmark and Memory Profiling`, `Statistical Tests Core`, `Synthetic Data and Raw Loading`, `MongoDB Loader`, `EDA Figures`, `Notebook Builder`?**
  _High betweenness centrality (0.214) - this node is a cross-community bridge._
- **Why does `Data layout data/<layer>/<source>/<sample_tag>/<table>` connect `Pipeline Conventions` to `Benchmark and Memory Profiling`, `Data Platform (Spark, Hive)`?**
  _High betweenness centrality (0.110) - this node is a cross-community bridge._
- **Why does `Medallion Parquet layers (bronze/silver/gold)` connect `Data Platform (Spark, Hive)` to `Modeling Decisions and Gotchas`, `Pipeline Conventions`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **Are the 34 inferred relationships involving `Config` (e.g. with `run_benchmark()` and `_spark_runner()`) actually correct?**
  _`Config` has 34 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `TestResult` (e.g. with `check_srm()` and `duration_buyers()`) actually correct?**
  _`TestResult` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `cohort_buyer_retention`, `daily_kpis`, `overview` to the rest of the system?**
  _20 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Configuration Loading` be split into smaller, more focused modules?**
  _Cohesion score 0.058823529411764705 - nodes in this community are weakly interconnected._