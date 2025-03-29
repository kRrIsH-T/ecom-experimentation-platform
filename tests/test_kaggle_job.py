"""The Kaggle kernel bundle: payload contents, script rendering and settings."""

import ast
import base64
import io
import json
import logging
import tarfile

import pytest

from ecom import kaggle_job
from ecom.config import load_config
from ecom.models import train_xgb


@pytest.fixture
def repo_cfg():
    return load_config()


def _names(payload: str) -> set[str]:
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload)), mode="r:gz") as tf:
        return set(tf.getnames())


def test_payload_has_project_and_pins_but_no_caches(repo_cfg):
    payload, meta = kaggle_job.build_payload(repo_cfg, include_champion=False)
    names = _names(payload)
    assert meta is None
    for required in (
        "pyproject.toml",
        "README.md",
        "configs/config.yaml",
        "constraints.txt",
        "src/ecom/cli.py",
        "src/ecom/sql/funnel.sql",
        "reports/templates/report.md.j2",
    ):
        assert required in names
    assert not any("__pycache__" in n or n.startswith(("tests/", "data/")) for n in names)


def test_pinned_requirements_exclude_project_and_windows_only():
    pins = kaggle_job.pinned_requirements().splitlines()
    names = {p.split("==")[0] for p in pins}
    assert "numpy" in names and "pyspark" in names
    assert names.isdisjoint(kaggle_job.UNPINNED)
    assert all("==" in p for p in pins)


def test_rendered_script_carries_settings_and_payload(repo_cfg):
    settings = kaggle_job.kernel_settings(repo_cfg, None)
    script = kaggle_job.render_script(settings, "UEFZTE9BRA==")
    tree = ast.parse(script)
    values = {
        node.targets[0].id if isinstance(node, ast.Assign) else node.target.id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign | ast.AnnAssign)
        and isinstance(node.targets[0] if isinstance(node, ast.Assign) else node.target, ast.Name)
    }
    assert ast.literal_eval(values["SETTINGS"]) == settings
    assert ast.literal_eval(values["PAYLOAD"]) == "UEFZTE9BRA=="
    assert "eval_model" not in settings
    assert [r["tag"] for r in settings["runs"]] == ["s0500", "full"]
    assert "load-mongo" in settings["skip_steps"]


def test_kernel_settings_name_the_bundled_champion(repo_cfg):
    meta = {"version": 5, "model_family": "xgboost", "trained_on": {"source": "synthetic"}}
    settings = kaggle_job.kernel_settings(repo_cfg, meta)
    assert settings["eval_model"] == f"{kaggle_job.BUNDLED_MODEL}/model"
    assert "v5" in settings["eval_model_name"] and "synthetic" in settings["eval_model_name"]


def test_resolve_device(monkeypatch):
    assert train_xgb.resolve_device("cpu") == "cpu"
    assert train_xgb.resolve_device("cuda") == "cuda"
    monkeypatch.setattr(train_xgb.xgboost, "build_info", lambda: {"USE_CUDA": False})
    assert train_xgb.resolve_device("auto") == "cpu"


def test_file_results_maps_kernel_output_into_project(cfg, tmp_path):
    dest = tmp_path / "download"
    run = dest / "results" / "full"
    (run / "outputs").mkdir(parents=True)
    (run / "outputs" / "champion.json").write_text("{}")
    (run / "reports" / "figures").mkdir(parents=True)
    (run / "reports" / "report.md").write_text("# r")
    (run / "reports" / "figures" / "shap_summary.png").write_bytes(b"png")
    (run / "champion_model").mkdir()
    (run / "champion_model" / "MLmodel").write_text("x")
    (dest / "results" / "source").mkdir()
    (dest / "results" / "source" / "data_quality.json").write_text("{}")
    (dest / "run_summary.json").write_text(json.dumps({"status": "ok"}))

    assert kaggle_job.file_results(cfg, dest) == ["full"]
    out = cfg.paths.outputs / "kaggle"
    assert (out / "full" / "champion.json").exists()
    assert (out / "data_quality.json").exists()
    assert (out / "kaggle_run_summary.json").exists()
    assert (out / "full" / "shap_summary.png").read_bytes() == b"png"
    assert (cfg.paths.reports / "kaggle" / "full" / "report.md").exists()
    assert (cfg.paths.models / "kaggle" / "full" / "champion" / "MLmodel").exists()


def test_setup_mlflow_keeps_root_logging_on_a_fresh_store(cfg):
    from ecom.models.train import setup_mlflow

    root = logging.getLogger()
    handler = logging.StreamHandler()
    old_handlers, old_level = root.handlers[:], root.level
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)
    try:
        setup_mlflow(cfg)  # new sqlite file under tmp_path -> Alembic migrations run
        assert root.handlers == [handler]
        assert root.level == logging.INFO
    finally:
        root.handlers[:] = old_handlers
        root.setLevel(old_level)
