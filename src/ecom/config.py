"""Configuration loading.

`configs/config.yaml` is the single source of truth. Any value can be overridden with an
environment variable named ``ECOM__<SECTION>__<KEY>`` (values are parsed as YAML, so
``ECOM__DATA__SAMPLE_RATE=1.0`` yields a float). ``ECOM_CONFIG`` points at an alternative
YAML file and ``ECOM_ROOT`` pins the project root.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

ENV_PREFIX = "ECOM__"


def find_root() -> Path:
    """Locate the project root (the directory holding configs/config.yaml)."""
    if env_root := os.environ.get("ECOM_ROOT"):
        return Path(env_root).resolve()
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "configs" / "config.yaml").is_file():
            return candidate
    # Editable install: src/ecom/config.py -> repo root is two levels above src/
    return Path(__file__).resolve().parents[2]


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(raw)
    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        parts = [p.lower() for p in key[len(ENV_PREFIX) :].split("__") if p]
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return out


@dataclass(frozen=True)
class Paths:
    root: Path
    raw: Path
    bronze: Path
    silver: Path
    gold: Path
    outputs: Path
    warehouse: Path
    metastore: Path
    reports: Path
    figures: Path
    models: Path
    tools: Path

    def raw_source(self, source: str) -> Path:
        return self.raw / source

    def ensure(self) -> None:
        for name in self.__dataclass_fields__:
            if name != "root":
                getattr(self, name).mkdir(parents=True, exist_ok=True)


class Config:
    """Thin read-only view over the merged YAML + environment configuration."""

    def __init__(self, data: dict[str, Any], root: Path):
        self._data = data
        self.root = root
        p = data["paths"]
        self.paths = Paths(root=root, **{k: (root / v).resolve() for k, v in p.items()})

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def with_overrides(self, **dotted: Any) -> Config:
        """Return a copy with ``section.key`` overrides (used by CLI flags and tests)."""
        data = copy.deepcopy(self._data)
        for key, value in dotted.items():
            if value is None:
                continue
            parts = key.split(".")
            node = data
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return Config(data, self.root)

    # Frequently used derived values -------------------------------------------------
    @property
    def seed(self) -> int:
        return int(self._data["seed"])

    @property
    def sample_rate(self) -> float:
        return float(self._data["data"]["sample_rate"])

    @property
    def source(self) -> str:
        return str(self._data["data"]["source"])

    @property
    def mlflow_uri(self) -> str:
        uri = str(self._data["mlflow"]["tracking_uri"])
        if uri.startswith("sqlite:///") and not Path(uri[len("sqlite:///") :]).is_absolute():
            return f"sqlite:///{(self.root / uri[len('sqlite:///') :]).as_posix()}"
        return uri

    @property
    def mongo_uri(self) -> str:
        return str(self._data["mongo"]["uri"])

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    root = find_root()
    cfg_path = Path(path or os.environ.get("ECOM_CONFIG") or root / "configs" / "config.yaml")
    with open(cfg_path, encoding="utf-8-sig") as fh:
        raw = yaml.safe_load(fh)
    return Config(_apply_env_overrides(raw), root)


@cache
def get_config() -> Config:
    """Process-wide cached config (CLI flags apply overrides via ``with_overrides``)."""
    return load_config()
