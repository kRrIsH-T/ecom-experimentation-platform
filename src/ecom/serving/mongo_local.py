"""Run a local MongoDB server from a portable binary (no Docker or admin rights needed).

The binary goes to ``.tools/mongodb`` and the data, log and pid file to ``data/mongo``, so
nothing is written outside the project. With Docker available, ``docker compose up mongo``
is an equivalent alternative; both listen on the URI from ``mongo.uri``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import shutil
import socket
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse

from ecom.config import Config
from ecom.tools import _download

log = logging.getLogger(__name__)

MONGO_VERSION = "7.0.14"
_URLS = {
    (
        "Windows",
        "x64",
    ): f"https://fastdl.mongodb.org/windows/mongodb-windows-x86_64-{MONGO_VERSION}.zip",
    (
        "Linux",
        "x64",
    ): f"https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu2204-{MONGO_VERSION}.tgz",
    (
        "Linux",
        "aarch64",
    ): f"https://fastdl.mongodb.org/linux/mongodb-linux-aarch64-ubuntu2204-{MONGO_VERSION}.tgz",
    ("Darwin", "x64"): f"https://fastdl.mongodb.org/osx/mongodb-macos-x86_64-{MONGO_VERSION}.tgz",
    (
        "Darwin",
        "aarch64",
    ): f"https://fastdl.mongodb.org/osx/mongodb-macos-arm64-{MONGO_VERSION}.tgz",
}


def _host_port(cfg: Config) -> tuple[str, int]:
    parsed = urlparse(cfg.mongo_uri)
    return parsed.hostname or "localhost", parsed.port or 27017


def _data_dir(cfg: Config) -> Path:
    path = cfg.root / "data" / "mongo"
    (path / "db").mkdir(parents=True, exist_ok=True)
    return path


def is_listening(cfg: Config, timeout: float = 0.5) -> bool:
    host, port = _host_port(cfg)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ensure_mongo_binary(cfg: Config) -> Path:
    exe = "mongod.exe" if os.name == "nt" else "mongod"
    found = sorted(cfg.paths.tools.glob(f"mongodb*/bin/{exe}"))
    if found:
        return found[-1]
    if system_mongod := shutil.which("mongod"):
        return Path(system_mongod)
    machine = platform.machine().lower()
    arch = "aarch64" if machine in {"arm64", "aarch64"} else "x64"
    url = _URLS[(platform.system(), arch)]
    archive = _download(url, cfg.paths.tools / "downloads" / Path(url).name)
    # Only the server binary is needed; the archive also ships ~1.5 GB of debug symbols.
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(cfg.paths.tools, [n for n in zf.namelist() if n.endswith(f"/bin/{exe}")])
    else:
        with tarfile.open(archive) as tf:
            members = [m for m in tf.getmembers() if m.name.endswith(f"/bin/{exe}")]
            tf.extractall(cfg.paths.tools, members=members, filter="data")
    archive.unlink()
    return ensure_mongo_binary(cfg)


def start_mongo(cfg: Config, wait_s: float = 30.0) -> None:
    if is_listening(cfg):
        log.info("MongoDB already listening at %s", cfg.mongo_uri)
        return
    mongod = ensure_mongo_binary(cfg)
    data = _data_dir(cfg)
    _, port = _host_port(cfg)
    cmd = [
        str(mongod),
        "--dbpath",
        str(data / "db"),
        "--port",
        str(port),
        "--bind_ip",
        "127.0.0.1",
        "--logpath",
        str(data / "mongod.log"),
        "--logappend",
        # small footprint for a laptop: modest cache, no 100 MB journal preallocation
        "--wiredTigerCacheSizeGB",
        "0.5",
        "--wiredTigerEngineConfigString",
        "log=(prealloc=false,file_max=20MB)",
    ]
    kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **kwargs)
    (data / "mongod.pid").write_text(str(proc.pid))
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if is_listening(cfg):
            log.info("MongoDB started (pid %d) at %s, data in %s", proc.pid, cfg.mongo_uri, data)
            return
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    raise RuntimeError(f"mongod failed to start; see {data / 'mongod.log'}")


def stop_mongo(cfg: Config) -> None:
    from pymongo import MongoClient
    from pymongo.errors import AutoReconnect, ConnectionFailure

    if not is_listening(cfg):
        log.info("MongoDB is not running")
        return
    # the server closes the connection while shutting down, which raises on the client
    with contextlib.suppress(AutoReconnect, ConnectionFailure):
        MongoClient(cfg.mongo_uri, serverSelectionTimeoutMS=3000).admin.command("shutdown")
    (_data_dir(cfg) / "mongod.pid").unlink(missing_ok=True)
    log.info("MongoDB stopped")


def status_mongo(cfg: Config) -> None:
    log.info("MongoDB at %s is %s", cfg.mongo_uri, "UP" if is_listening(cfg) else "DOWN")
