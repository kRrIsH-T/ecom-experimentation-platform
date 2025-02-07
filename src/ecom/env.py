"""Keep every temp, spill and cache directory inside the project (never the system drive).

Called from ``ecom/__init__.py`` so any entry point (CLI, notebooks, Streamlit, FastAPI,
pytest) is covered before matplotlib, numba, Spark or H2O pick their defaults. Child
processes (the Spark JVM, the H2O JVM, nbconvert kernels) inherit the environment.
Set ``ECOM_CACHE_DIR`` to relocate the whole tree.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Environment variable -> sub-directory of the cache root
_ENV_DIRS = {
    "TMP": "tmp",
    "TEMP": "tmp",
    "TMPDIR": "tmp",
    "SPARK_LOCAL_DIRS": "spark-local",  # shuffle + spill files
    "MPLCONFIGDIR": "matplotlib",
    "NUMBA_CACHE_DIR": "numba",
    "IPYTHONDIR": "ipython",
    "JUPYTER_RUNTIME_DIR": "jupyter/runtime",
    "JUPYTER_DATA_DIR": "jupyter/data",
    "JUPYTER_CONFIG_DIR": "jupyter/config",
    "XDG_CACHE_HOME": "xdg",
}
# KAGGLE_CONFIG_DIR is deliberately absent: it is where the user keeps credentials, not a cache.
_SUBDIRS = ("hive/scratch", "hive/local-scratch", "hive/resources", "h2o", "ivy2")

_configured: Path | None = None


def cache_root() -> Path:
    if env := os.environ.get("ECOM_CACHE_DIR"):
        return Path(env).resolve()
    from ecom.config import find_root

    return find_root() / ".cache"


def configure_local_dirs() -> Path:
    """Point temp/cache env vars at ``<root>/.cache`` (idempotent)."""
    global _configured
    if _configured is not None:
        return _configured
    root = cache_root()
    for var, sub in _ENV_DIRS.items():
        path = root / sub
        path.mkdir(parents=True, exist_ok=True)
        os.environ[var] = str(path)
    for sub in _SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = os.environ["TMP"]
    _configured = root
    return root


def local_dir(name: str) -> Path:
    """Path of a managed sub-directory (e.g. ``spark-local``, ``h2o``)."""
    return configure_local_dirs() / name
