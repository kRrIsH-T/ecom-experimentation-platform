"""Render the written deliverables from templates so every number matches the latest run.

Templates live in ``reports/templates``; outputs are ``reports/report.md``,
``reports/business_memo.md``, ``reports/slides.md`` (Marp) and the results block between the
``RESULTS:START`` / ``RESULTS:END`` markers in ``README.md``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ecom.config import Config
from ecom.layout import outputs_dir, sample_tag, source_outputs_dir

log = logging.getLogger(__name__)

TEMPLATES = {
    "report.md.j2": "report.md",
    "business_memo.md.j2": "business_memo.md",
    "slides.md.j2": "slides.md",
}
README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _num(v: Any, digits: int = 0) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:,.{digits}f}"


def _pct(v: Any, digits: int = 1) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{digits}%}"


def _signed_pct(v: Any, digits: int = 1) -> str:
    return "n/a" if v is None else f"{v:+.{digits}%}"


def _pval(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return "< 1e-16" if v < 1e-16 else f"{v:.2g}" if v < 0.001 else f"{v:.3f}"


def _records(path: Path) -> list[dict]:
    return pd.read_csv(path).to_dict(orient="records") if path.exists() else []


def _json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def _hardware(cfg: Config) -> str:
    """Compute resources of this run, for the timings footnote."""
    from ecom.models.train_xgb import resolve_device

    master = str(cfg.get("spark.master", "local[*]"))
    if master == "local[*]":
        master = f"local[{os.cpu_count()}]"
    return (
        f"Spark {master}, {cfg.get('spark.driver_memory')} driver, "
        f"XGBoost on {resolve_device(str(cfg.get('ml.xgb_device', 'cpu')))}"
    )


def build_context(cfg: Config) -> dict[str, Any]:
    out, src = outputs_dir(cfg), source_outputs_dir(cfg)
    stats = _json(out / "stats_tests.json") or {"tests": [], "bootstrap_median_aov": []}
    ctx = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "source": cfg.source,
        "synthetic": cfg.source == "synthetic",
        "sample_rate": cfg.sample_rate,
        "sample_tag": sample_tag(cfg.sample_rate),
        "cfg": cfg.as_dict(),
        "dq": _json(src / "data_quality.json") or {},
        "benchmark": _json(src / "benchmark.json"),
        "overview": (_records(out / "overview.csv") or [{}])[0],
        "funnel": _records(out / "funnel_overall.csv"),
        "funnel_sessions": _records(out / "funnel_sessions.csv"),
        "rfm": _records(out / "rfm_summary.csv"),
        "categories": _records(out / "funnel_by_category.csv"),
        "eda": _json(out / "eda_insights.json") or {"facts": {}, "insights": []},
        "tests": stats["tests"],
        "bootstrap": stats["bootstrap_median_aov"],
        "dataset": _json(out / "dataset_stats.json") or {},
        "models": _records(out / "model_comparison.csv"),
        "champion": _json(out / "champion.json") or {},
        "drivers": _json(out / "model_drivers.json") or [],
        "h2o": _records(out / "h2o_leaderboard.csv")[:5],
        "ab": _json(out / "ab_results.json"),
        "timings": _json(out / "pipeline_timings.json"),
        "hardware": _hardware(cfg),
    }
    ctx["best_model"] = next((m for m in ctx["models"] if m.get("champion")), {})
    return ctx


def environment(template_dir: Path) -> Environment:
    env = Environment(
        loader=FileSystemLoader(template_dir),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.filters.update(num=_num, pct=_pct, spct=_signed_pct, pval=_pval)
    return env


def render_reports(cfg: Config) -> list[Path]:
    template_dir = cfg.paths.reports / "templates"
    env = environment(template_dir)
    ctx = build_context(cfg)
    written = []
    for tpl, name in TEMPLATES.items():
        path = cfg.paths.reports / name
        path.write_text(env.get_template(tpl).render(**ctx), encoding="utf-8")
        written.append(path)
    readme = cfg.root / "README.md"
    if readme.exists() and (template_dir / "readme_results.md.j2").exists():
        block = env.get_template("readme_results.md.j2").render(**ctx).strip()
        text = readme.read_text(encoding="utf-8")
        pattern = re.compile(re.escape(README_START) + ".*?" + re.escape(README_END), re.S)
        if pattern.search(text):
            text = pattern.sub(f"{README_START}\n{block}\n{README_END}", text)
            readme.write_text(text, encoding="utf-8")
            written.append(readme)
    log.info("Rendered %s", [p.name for p in written])
    return written
