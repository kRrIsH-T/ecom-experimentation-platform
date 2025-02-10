"""Logical table name -> storage location (medallion layout).

The dev workflow runs on a user-level sample, so every table derived from events lives
under a sample-specific prefix (``s0500`` = 5.00%). Switching ``data.sample_rate`` therefore
never mixes rows from two different samples.
"""

from __future__ import annotations

from pathlib import Path

from ecom.config import Config

BRONZE_FULL = "events_full"


def sample_tag(rate: float) -> str:
    if rate >= 1.0:
        return "full"
    return f"s{round(rate * 10_000):04d}"


# table name -> (layer, partition column or None)
TABLES: dict[str, tuple[str, str | None]] = {
    "events": ("bronze", "event_date"),
    "events_clean": ("silver", "event_date"),
    "sessions": ("silver", None),
    "user_features": ("gold", "snapshot"),
    "labels": ("gold", "snapshot"),
    "model_dataset": ("gold", None),
    "scores": ("gold", None),
}


def bronze_full_path(cfg: Config) -> Path:
    return cfg.paths.bronze / cfg.source / BRONZE_FULL


def table_path(cfg: Config, name: str) -> Path:
    if name == BRONZE_FULL:
        return bronze_full_path(cfg)
    if name == "events" and cfg.sample_rate >= 1.0:
        return bronze_full_path(cfg)
    layer, _ = TABLES[name]
    base = getattr(cfg.paths, layer)
    return base / cfg.source / sample_tag(cfg.sample_rate) / name


def outputs_dir(cfg: Config) -> Path:
    """Small result files (CSV/JSON) for the active source + sample."""
    path = cfg.paths.outputs / cfg.source / sample_tag(cfg.sample_rate)
    path.mkdir(parents=True, exist_ok=True)
    return path


def source_outputs_dir(cfg: Config) -> Path:
    """Results that describe the full source regardless of sample (e.g. data quality)."""
    path = cfg.paths.outputs / cfg.source
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir(cfg: Config) -> Path:
    path = cfg.paths.figures
    path.mkdir(parents=True, exist_ok=True)
    return path
