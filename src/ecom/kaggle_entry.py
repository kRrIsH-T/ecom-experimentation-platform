"""Kaggle-side runner: the body of the private kernel that `ecom kaggle push` creates.

`ecom.kaggle_job.build_kernel` copies this file and fills in ``SETTINGS`` and ``PAYLOAD``
(the project source, pinned requirements and the local champion as a base64 tar.gz). On
Kaggle it:

1. unpacks the project into a scratch directory and links the dataset CSVs into
   ``data/raw/kaggle``;
2. builds a Python venv with uv, pinned to the exact package versions of the machine that
   pushed (the Kaggle image's own Python and Spark are newer than the project supports);
3. runs ``ecom --source kaggle --sample <s> all`` for each configured sample, then scores
   the bundled local champion on the same real snapshots;
4. copies outputs, rendered reports, figures, the new champion model and logs to the
   kernel output directory, where `ecom kaggle pull` downloads them.

Standard library only. ``ECOM_KAGGLE_INPUT`` / ``_WORK`` / ``_OUT`` / ``_PYTHON`` override
the Kaggle paths for a local dry run (``_PYTHON`` reuses an existing interpreter).
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

SETTINGS: dict = {}
PAYLOAD = ""

INPUT = Path(os.environ.get("ECOM_KAGGLE_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("ECOM_KAGGLE_WORK", "/tmp/ecom-work"))
OUT = Path(os.environ.get("ECOM_KAGGLE_OUT", "/kaggle/working"))


def log(msg: str) -> None:
    print(f"[ecom-kaggle {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list, env: dict | None = None, log_file: Path | None = None) -> int:
    """Run a command, streaming its output to the kernel log (and a file); return the code."""
    log("$ " + " ".join(str(c) for c in cmd))
    fh = open(log_file, "a", encoding="utf-8") if log_file else None  # noqa: SIM115
    try:
        with subprocess.Popen(
            [str(c) for c in cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            env=env,
        ) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()  # live progress in `kaggle kernels logs`
                if fh:
                    fh.write(line)
            return proc.wait()
    finally:
        if fh:
            fh.close()


def must(cmd: list, env: dict | None = None) -> None:
    if (code := run(cmd, env)) != 0:
        raise RuntimeError(f"command failed with exit code {code}: {cmd}")


def describe_machine() -> dict:
    info = {"python": sys.version.split()[0], "cpus": os.cpu_count()}
    try:
        gpus = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        info["gpus"] = [g.strip() for g in gpus.splitlines() if g.strip()]
    except (OSError, subprocess.SubprocessError):
        info["gpus"] = []
    try:
        meminfo = Path("/proc/meminfo").read_text().split()
        info["ram_gb"] = round(int(meminfo[meminfo.index("MemTotal:") + 1]) / 1024**2, 1)
    except (OSError, ValueError):
        pass
    log(f"machine: {info}")
    return info


def unpack(project: Path) -> None:
    if project.exists():
        shutil.rmtree(project)
    project.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz") as tf:
        tf.extractall(project, filter="data")
    log(f"project unpacked into {project}")


def link_raw(project: Path) -> list[Path]:
    """Expose the attached dataset's CSVs as data/raw/kaggle/* (symlinks: input is read-only)."""
    raw = project / "data" / "raw" / "kaggle"
    raw.mkdir(parents=True, exist_ok=True)
    linked = []
    for name in SETTINGS["kaggle_files"]:
        matches = sorted(INPUT.rglob(name))
        if not matches:
            raise FileNotFoundError(f"{name} not found under {INPUT} - is the dataset attached?")
        target = raw / name
        target.unlink(missing_ok=True)
        try:
            target.symlink_to(matches[0])
        except OSError:  # e.g. Windows without symlink rights (local dry run)
            shutil.copy2(matches[0], target)
        linked.append(matches[0])
        log(f"raw: {matches[0]} ({matches[0].stat().st_size / 1e9:.2f} GB)")
    return linked


def make_venv(project: Path) -> Path:
    if override := os.environ.get("ECOM_KAGGLE_PYTHON"):
        return Path(override)
    env = {
        **os.environ,
        "UV_CACHE_DIR": str(WORK / "uv-cache"),
        "UV_PYTHON_INSTALL_DIR": str(WORK / "uv-python"),
    }
    uv = shutil.which("uv")
    if uv is None:
        must([sys.executable, "-m", "pip", "install", "--quiet", "uv"])
        uv = shutil.which("uv") or str(Path(sys.executable).with_name("uv"))
    venv = WORK / "venv"
    # --seed adds pip, which MLflow probes when it writes the model's environment files
    must([uv, "venv", "--seed", "--python", SETTINGS["python"], venv], env)
    python = venv / "bin" / "python"
    pins = project / "constraints.txt"
    must([uv, "pip", "install", "--python", python, "-c", pins, "-e", project], env)
    return python


def pipeline_env(project: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("ECOM__")}
    env.update(SETTINGS["env"])
    env["ECOM_ROOT"] = str(project)
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("JAVA_HOME", None)  # the image ships Java 21; ecom fetches the JDK 17 Spark needs
    return env


def collect(project: Path, tag: str) -> None:
    """Copy one sample's results into the kernel output (outputs, reports, figures, model)."""
    dst = OUT / "results" / tag
    outputs = project / "data" / "outputs" / "kaggle"
    if (outputs / tag).exists():
        shutil.copytree(outputs / tag, dst / "outputs", dirs_exist_ok=True)
    for f in outputs.glob("*.json"):  # source-level results, e.g. data_quality.json
        (OUT / "results" / "source").mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, OUT / "results" / "source" / f.name)
    reports = project / "reports"
    (dst / "reports").mkdir(parents=True, exist_ok=True)
    for f in reports.glob("*.md"):
        shutil.copy2(f, dst / "reports" / f.name)
    if (reports / "figures").exists():
        shutil.copytree(reports / "figures", dst / "reports" / "figures", dirs_exist_ok=True)
    champion = outputs / tag / "champion.json"
    if champion.exists():
        run_id = json.loads(champion.read_text())["run_id"]
        model = project / "mlruns" / run_id / "artifacts" / "model"
        if model.exists():
            shutil.copytree(model, dst / "champion_model", dirs_exist_ok=True)
    log(f"collected {tag} -> {dst}")


def main() -> int:
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    summary: dict = {"settings": dict(SETTINGS), "runs": []}
    summary["machine"] = describe_machine()
    project = WORK / "smaiproj"
    code = 0
    try:
        unpack(project)
        summary["raw_files"] = [str(p) for p in link_raw(project)]
        t0 = time.time()
        python = make_venv(project)
        summary["setup_seconds"] = round(time.time() - t0, 1)
        env = pipeline_env(project)
        ecom = [python, "-m", "ecom", "--source", "kaggle"]
        for spec in SETTINGS["runs"]:
            sample, tag = spec["sample"], spec["tag"]
            entry: dict = {"sample": sample, "tag": tag}
            t0 = time.time()
            entry["pipeline_returncode"] = run(
                [*ecom, "--sample", sample, "all", "--skip", *SETTINGS["skip_steps"]],
                env,
                OUT / "logs" / f"pipeline_{tag}.log",
            )
            entry["pipeline_seconds"] = round(time.time() - t0, 1)
            if entry["pipeline_returncode"] == 0 and SETTINGS.get("eval_model"):
                entry["eval_returncode"] = run(
                    [
                        *ecom,
                        "--sample",
                        sample,
                        "evaluate-model",
                        "--model",
                        project / SETTINGS["eval_model"],
                        "--name",
                        SETTINGS["eval_model_name"],
                    ],
                    env,
                    OUT / "logs" / f"evaluate_{tag}.log",
                )
            collect(project, tag)
            summary["runs"].append(entry)
            if entry["pipeline_returncode"] != 0 or entry.get("eval_returncode", 0) != 0:
                code = 1
                log(f"sample {sample} failed - skipping the remaining samples")
                break
    except Exception as exc:
        log(f"runner failed: {exc!r}")
        summary["error"] = repr(exc)
        code = 1
    finally:
        summary["total_seconds"] = round(time.time() - t_start, 1)
        summary["status"] = "ok" if code == 0 else "failed"
        (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        log(f"done: {summary['status']} in {summary['total_seconds']:.0f}s")
    return code


if __name__ == "__main__":
    sys.exit(main())
