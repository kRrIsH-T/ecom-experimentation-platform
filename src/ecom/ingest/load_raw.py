"""Locate (and optionally fetch) the raw CSV files for the configured data source."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

from ecom.config import Config

log = logging.getLogger(__name__)

RAW_SCHEMA_DDL = (
    "event_time STRING, event_type STRING, product_id BIGINT, category_id BIGINT, "
    "category_code STRING, brand STRING, price DOUBLE, user_id BIGINT, user_session STRING"
)


def _csv_files(raw_dir: Path) -> list[Path]:
    """Plain or gzip-compressed CSVs (Spark decompresses .csv.gz transparently)."""
    return sorted([*raw_dir.glob("*.csv"), *raw_dir.glob("*.csv.gz")])


def raw_files(cfg: Config) -> list[Path]:
    """Raw CSVs for the active source, generating synthetic data on first use."""
    raw_dir = cfg.paths.raw_source(cfg.source)
    files = _csv_files(raw_dir)
    if files:
        return files
    if cfg.source == "synthetic":
        from ecom.ingest.synthetic import generate

        log.info("No synthetic CSVs found - generating them into %s", raw_dir)
        generate(cfg, raw_dir)
        return _csv_files(raw_dir)
    raise FileNotFoundError(
        f"No CSV files in {raw_dir}. Run `ecom download` (needs Kaggle credentials) or copy "
        f"{cfg.get('data.kaggle_files')} there."
    )


def download_kaggle(cfg: Config) -> list[Path]:
    """Download the real dataset with the Kaggle API (needs ~/.kaggle/kaggle.json)."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    raw_dir = cfg.paths.raw_source("kaggle")
    raw_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    dataset = cfg.get("data.kaggle_dataset")
    for name in cfg.get("data.kaggle_files"):
        target = raw_dir / name
        if target.exists():
            log.info("%s already present", target)
            continue
        log.info("Downloading %s from %s", name, dataset)
        api.dataset_download_file(dataset, name, path=str(raw_dir))
        archive = raw_dir / f"{name}.zip"
        if archive.exists():
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(raw_dir)
            archive.unlink()
    return sorted(raw_dir.glob("*.csv"))
