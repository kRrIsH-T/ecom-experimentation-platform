import numpy as np
import pytest

from ecom.models.evaluate import classification_metrics, curve_data, gains_table


def test_gains_table_perfect_ranking():
    y = np.array([1] * 10 + [0] * 90)
    p = np.linspace(1, 0, 100)
    g = gains_table(y, p)
    assert len(g) == 10
    assert g.loc[0, "positive_rate"] == 1.0
    assert g.loc[0, "lift"] == pytest.approx(10.0)
    assert g["cumulative_gain"].iloc[-1] == pytest.approx(1.0)
    assert g["users"].sum() == 100


def test_metrics_random_vs_perfect():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.05).astype(int)
    perfect = classification_metrics(y, y * 0.9 + 0.05)
    random = classification_metrics(y, rng.random(5000))
    assert perfect["roc_auc"] == pytest.approx(1.0)
    assert perfect["pr_auc"] == pytest.approx(1.0)
    assert random["roc_auc"] == pytest.approx(0.5, abs=0.05)
    assert random["pr_auc"] == pytest.approx(y.mean(), abs=0.02)
    assert random["base_rate"] == pytest.approx(y.mean())


def test_curve_data_is_json_friendly():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.1).astype(int)
    c = curve_data(y, rng.random(2000))
    assert len(c["roc"]["fpr"]) <= 200
    assert set(c) == {"roc", "pr", "calibration", "gains"}
