"""Bootstrap the native tooling Spark and H2O need: a JDK 17 and, on Windows, winutils.

Everything is downloaded into ``.tools/`` (gitignored) so a fresh clone needs nothing but
Python. Existing installations are reused when they are new enough.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from ecom.config import Config

log = logging.getLogger(__name__)

ADOPTIUM_URL = (
    "https://api.adoptium.net/v3/binary/latest/17/ga/{os}/{arch}/jdk/hotspot/normal/eclipse"
)
# Hadoop 3.3.x native helpers for Windows (Spark 3.5 bundles the Hadoop 3.3 client).
WINUTILS_BASE = "https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-3.3.6/bin"
WINUTILS_FILES = ("winutils.exe", "hadoop.dll")
MIN_JAVA = 11
MAX_JAVA = 17  # Spark 3.5 is supported on Java 8/11/17


def _java_major(java_bin: Path | str) -> int | None:
    try:
        out = subprocess.run(
            [str(java_bin), "-version"], capture_output=True, text=True, timeout=30
        ).stderr
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r'version "(\d+)(?:\.(\d+))?', out)
    if not match:
        return None
    major = int(match.group(1))
    return int(match.group(2)) if major == 1 and match.group(2) else major


def _java_bin(home: Path) -> Path:
    return home / "bin" / ("java.exe" if os.name == "nt" else "java")


def _usable_java_home(home: Path) -> bool:
    major = _java_major(_java_bin(home))
    return major is not None and MIN_JAVA <= major <= MAX_JAVA


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "ecom-experimentation-setup"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    return dest


def _platform_ids() -> tuple[str, str]:
    system = {"Windows": "windows", "Linux": "linux", "Darwin": "mac"}[platform.system()]
    machine = platform.machine().lower()
    arch = "aarch64" if machine in {"arm64", "aarch64"} else "x64"
    return system, arch


def ensure_java(cfg: Config) -> Path:
    """Return a usable JAVA_HOME, downloading Temurin 17 into .tools/ if needed."""
    if (home := os.environ.get("JAVA_HOME")) and _usable_java_home(Path(home)):
        return Path(home)
    tools = cfg.paths.tools
    for candidate in sorted(tools.glob("jdk-17*"), reverse=True):
        home = candidate / "Contents" / "Home" if (candidate / "Contents").exists() else candidate
        if _usable_java_home(home):
            os.environ["JAVA_HOME"] = str(home)
            return home
    java = shutil.which("java")
    major = _java_major(java) if java else None
    if java and major and MIN_JAVA <= major <= MAX_JAVA:
        home = Path(java).resolve().parents[1]
        os.environ["JAVA_HOME"] = str(home)
        return home

    system, arch = _platform_ids()
    archive_name = "jdk17.zip" if system == "windows" else "jdk17.tar.gz"
    archive = _download(
        ADOPTIUM_URL.format(os=system, arch=arch), tools / "downloads" / archive_name
    )
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tools)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(tools, filter="data")
    archive.unlink()
    return ensure_java(cfg)


def ensure_hadoop_home(cfg: Config) -> Path | None:
    """On Windows, provide HADOOP_HOME with winutils.exe/hadoop.dll (no-op elsewhere)."""
    if os.name != "nt":
        return None
    hadoop_home = cfg.paths.tools / "hadoop"
    bin_dir = hadoop_home / "bin"
    for name in WINUTILS_FILES:
        if not (bin_dir / name).exists():
            _download(f"{WINUTILS_BASE}/{name}", bin_dir / name)
    os.environ["HADOOP_HOME"] = str(hadoop_home)
    if str(bin_dir) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    return hadoop_home


def ensure_runtime(cfg: Config) -> None:
    """Prepare JAVA_HOME / HADOOP_HOME / PYSPARK_PYTHON for Spark and H2O."""
    java_home = ensure_java(cfg)
    os.environ["JAVA_HOME"] = str(java_home)
    java_bin_dir = str(java_home / "bin")
    if java_bin_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{java_bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    ensure_hadoop_home(cfg)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
