# Thin wrapper over the `ecom` CLI (the CLI is the source of truth and also works on Windows,
# where make is usually not installed: run `python -m ecom <step>` directly).
PY ?= python
SAMPLE ?= 0.05
SOURCE ?= synthetic
ECOM = $(PY) -m ecom --sample $(SAMPLE) --source $(SOURCE)

.PHONY: setup generate download ingest transform features sql benchmark eda stats train abtest \
        mongo load-mongo serve dashboard mlflow-ui report notebooks all test lint format clean

setup:            ## install deps, JDK/winutils/MongoDB into .tools, git hooks
	$(PY) -m pip install -e ".[dev]"
	$(ECOM) setup
	pre-commit install

generate:         ## synthetic Kaggle-format CSVs
	$(ECOM) generate
download:         ## real Kaggle dataset (needs ~/.kaggle/kaggle.json)
	$(ECOM) download
ingest:           ## raw CSV -> bronze Parquet + Hive + data quality + sample
	$(ECOM) ingest
transform:
	$(ECOM) transform
features: transform  ## silver + gold features/labels + SQL analytics
	$(ECOM) features
	$(ECOM) sql
sql:
	$(ECOM) sql
benchmark:
	$(ECOM) benchmark
eda:
	$(ECOM) eda
stats:
	$(ECOM) stats
train:
	$(ECOM) train
abtest:
	$(ECOM) abtest
mongo:
	$(ECOM) mongo start
load-mongo: mongo
	$(ECOM) load-mongo
serve:
	$(ECOM) serve
dashboard:
	$(ECOM) dashboard
mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
report:
	$(ECOM) report
notebooks:
	$(ECOM) notebooks

all: mongo        ## whole pipeline end to end
	$(ECOM) all

test:
	$(PY) -m pytest
lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
format:
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .
clean:            ## remove generated data (keeps raw CSVs and tools)
	rm -rf data/bronze data/silver data/gold data/outputs data/warehouse data/metastore mlruns mlflow.db
