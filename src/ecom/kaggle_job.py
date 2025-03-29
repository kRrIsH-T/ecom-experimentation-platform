"""Run the pipeline on the real Kaggle dataset in a private Kaggle kernel (GPU T4 x2).

    ecom kaggle push     # bundle source + pinned requirements + local champion; start a run
    ecom kaggle status   # status of the latest run
    ecom kaggle logs     # execution log of the latest run
    ecom kaggle pull     # results -> data/outputs/kaggle/, reports/kaggle/, models/kaggle/

The kernel is one generated script (`ecom.kaggle_entry` plus an embedded base64 tar.gz of
the project), so every kernel version records exactly the code it ran and no Kaggle
dataset has to be maintained for the source. Needs Kaggle CLI credentials (``kaggle
config view`` must show your username); the kernel is always private.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import shutil
import subprocess
import sys
import tarfile
from importlib import metadata
from pathlib import Path

from ecom.config import Config
from ecom.env import local_dir
from ecom.layout import sample_tag

log = logging.getLogger(__name__)

# Project files the remote run needs (tests, dashboard and notebooks stay local).
INCLUDE = ("pyproject.toml", "README.md", "configs", "src", "reports/templates")
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
# Never pinned: the project itself and Windows-only packages.
UNPINNED = {"ecom-experimentation", "pywin32", "pywinpty"}
BUNDLED_MODEL = "models/local_champion"
ENTRY = Path(__file__).with_name("kaggle_entry.py")
SCRIPT_NAME = "ecom_kaggle_run.py"


def _kaggle_cli() -> str:
    exe = Path(sys.executable).with_name("kaggle.exe" if sys.platform == "win32" else "kaggle")
    found = str(exe) if exe.exists() else shutil.which("kaggle")
    if not found:
        raise RuntimeError("Kaggle CLI not found - install the dev extras (`pip install kaggle`)")
    return found


def _kaggle(*args: str, capture: bool = False) -> str:
    proc = subprocess.run([_kaggle_cli(), *args], capture_output=capture, text=True, check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip() if capture else ""
        raise RuntimeError(f"kaggle {' '.join(args)} failed ({proc.returncode}) {detail}")
    return proc.stdout if capture else ""


def kaggle_username() -> str:
    out = _kaggle("config", "view", capture=True)
    match = re.search(r"^- username: (\S+)", out, re.M)
    if not match or match.group(1) == "None":
        raise RuntimeError("No Kaggle username configured - run `kaggle config view` to check")
    return match.group(1)


def kernel_ref(cfg: Config, username: str | None = None) -> str:
    return f"{username or kaggle_username()}/{cfg.get('kaggle.kernel_slug', 'ecom-kaggle-run')}"


def pinned_requirements() -> str:
    """``name==version`` for every distribution in this venv, as uv constraints for Kaggle."""
    pins = {}
    for dist in metadata.distributions():
        name = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
        if name not in UNPINNED:
            pins[name] = dist.version
    return "".join(f"{n}=={v}\n" for n, v in sorted(pins.items()))


def _project_files(root: Path) -> list[Path]:
    files = []
    for item in INCLUDE:
        path = root / item
        candidates = [path] if path.is_file() else sorted(path.rglob("*"))
        files += [
            p
            for p in candidates
            if p.is_file() and not EXCLUDED_PARTS.intersection(p.relative_to(root).parts)
        ]
    return files


def _local_champion(cfg: Config) -> tuple[Path, dict] | None:
    """The registered ``champion`` of this machine's MLflow registry (model dir + metadata)."""
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(cfg.mlflow_uri)
    name = cfg.get("ml.registered_model", "ecom-propensity")
    client = MlflowClient()
    try:
        version = client.get_model_version_by_alias(name, "champion")
    except MlflowException:
        return None
    run = client.get_run(version.run_id)
    model_dir = cfg.root / "mlruns" / version.run_id / "artifacts" / "model"
    if not model_dir.exists():
        return None
    meta = {
        "registered_model": name,
        "version": int(version.version),
        "run_id": version.run_id,
        "model_family": version.tags.get("model_family"),
        "trained_on": {
            "source": run.data.tags.get("data_source"),
            "sample_rate": run.data.tags.get("sample_rate"),
        },
        "metrics_on_training_source": {
            k: v for k, v in run.data.metrics.items() if k.startswith(("valid_", "test_"))
        },
    }
    return model_dir, meta


def build_payload(cfg: Config, include_champion: bool) -> tuple[str, dict | None]:
    """Base64 tar.gz of the project files, pinned requirements and (optionally) the champion."""
    root = cfg.root
    buf = io.BytesIO()
    meta = None
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tf:
        for path in _project_files(root):
            tf.add(path, arcname=path.relative_to(root).as_posix())
        data = pinned_requirements().encode()
        info = tarfile.TarInfo("constraints.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
        champion = _local_champion(cfg) if include_champion else None
        if champion is not None:
            model_dir, meta = champion
            tf.add(model_dir, arcname=f"{BUNDLED_MODEL}/model")
            blob = json.dumps(meta, indent=2).encode()
            info = tarfile.TarInfo(f"{BUNDLED_MODEL}/meta.json")
            info.size = len(blob)
            tf.addfile(info, io.BytesIO(blob))
    return base64.b64encode(buf.getvalue()).decode("ascii"), meta


def kernel_settings(cfg: Config, champion_meta: dict | None) -> dict:
    samples = [float(s) for s in cfg.get("kaggle.samples", [0.05, 1.0])]
    settings = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "kaggle_files": list(cfg.get("data.kaggle_files")),
        "runs": [{"sample": s, "tag": sample_tag(s)} for s in samples],
        "skip_steps": list(cfg.get("kaggle.skip_steps", ["load-mongo"])),
        "env": {k: str(v) for k, v in (cfg.get("kaggle.env") or {}).items()},
    }
    if champion_meta is not None:
        settings["eval_model"] = f"{BUNDLED_MODEL}/model"
        settings["eval_model_name"] = (
            f"local champion v{champion_meta['version']} "
            f"({champion_meta['model_family']}, trained on {champion_meta['trained_on']['source']})"
        )
    return settings


def render_script(settings: dict, payload: str) -> str:
    text = ENTRY.read_text(encoding="utf-8")
    for line, value in (("SETTINGS: dict = {}", repr(settings)), ('PAYLOAD = ""', repr(payload))):
        if text.count(f"\n{line}\n") != 1:
            raise RuntimeError(f"placeholder {line!r} not found exactly once in {ENTRY}")
        text = text.replace(f"\n{line}\n", f"\n{line.split(' =')[0]} = {value}\n", 1)
    return text


def build_kernel(cfg: Config, username: str, out_dir: Path | None = None) -> Path:
    """Write ``kernel-metadata.json`` and the self-contained script; return the folder."""
    out_dir = out_dir or local_dir("kaggle") / "kernel"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    payload, meta = build_payload(cfg, bool(cfg.get("kaggle.eval_local_champion", True)))
    settings = kernel_settings(cfg, meta)
    (out_dir / SCRIPT_NAME).write_text(render_script(settings, payload), encoding="utf-8")
    slug = cfg.get("kaggle.kernel_slug", "ecom-kaggle-run")
    kernel_meta = {
        "id": f"{username}/{slug}",
        "title": slug.replace("-", " "),
        "code_file": SCRIPT_NAME,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,  # uv fetches Python 3.11 and the pinned wheels
        "dataset_sources": [cfg.get("data.kaggle_dataset")],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(kernel_meta, indent=2))
    log.info(
        "Kernel %s built in %s (payload %.0f KB, champion bundled: %s, runs: %s)",
        kernel_meta["id"],
        out_dir,
        len(payload) / 1024,
        meta is not None,
        [r["tag"] for r in settings["runs"]],
    )
    return out_dir


def push(cfg: Config) -> str:
    username = kaggle_username()
    folder = build_kernel(cfg, username)
    accelerator = cfg.get("kaggle.accelerator", "NvidiaTeslaT4")
    _kaggle("kernels", "push", "-p", str(folder), "--accelerator", accelerator)
    ref = kernel_ref(cfg, username)
    log.info("Pushed %s on %s: https://www.kaggle.com/code/%s", ref, accelerator, ref)
    return ref


def status(cfg: Config) -> str:
    out = _kaggle("kernels", "status", kernel_ref(cfg), capture=True).strip()
    log.info(out)
    return out


def logs(cfg: Config) -> None:
    _kaggle("kernels", "logs", kernel_ref(cfg))


def pull(cfg: Config) -> Path:
    """Download the latest run's output and file it next to the local results."""
    dest = local_dir("kaggle") / "output"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    _kaggle("kernels", "output", kernel_ref(cfg), "-p", str(dest), "-o")
    file_results(cfg, dest)
    return dest


def file_results(cfg: Config, dest: Path) -> list[str]:
    """Copy a downloaded kernel output into the project layout; return the sample tags."""
    results = dest / "results"
    if not results.exists():
        raise RuntimeError(f"No results/ in the kernel output ({dest}) - see `ecom kaggle logs`")
    outputs = cfg.paths.outputs / "kaggle"
    outputs.mkdir(parents=True, exist_ok=True)
    if (results / "source").exists():
        shutil.copytree(results / "source", outputs, dirs_exist_ok=True)
    tags = []
    for run_dir in sorted(p for p in results.iterdir() if p.is_dir() and p.name != "source"):
        tag = run_dir.name
        pairs = {
            "outputs": outputs / tag,
            "reports": cfg.paths.reports / "kaggle" / tag,
            "champion_model": cfg.paths.models / "kaggle" / tag / "champion",
        }
        for sub, target in pairs.items():
            if (run_dir / sub).exists():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(run_dir / sub, target)
        # the dashboard reads the SHAP plot from the run's outputs (kernels pushed before
        # train.py kept that copy only have it under reports/figures)
        shap_png = run_dir / "reports" / "figures" / "shap_summary.png"
        if shap_png.exists() and not (outputs / tag / shap_png.name).exists():
            shutil.copy2(shap_png, outputs / tag / shap_png.name)
        tags.append(tag)
        log.info("Pulled %s -> %s", tag, ", ".join(str(t) for t in pairs.values()))
    if (summary := dest / "run_summary.json").exists():
        shutil.copy2(summary, outputs / "kaggle_run_summary.json")
    return tags
