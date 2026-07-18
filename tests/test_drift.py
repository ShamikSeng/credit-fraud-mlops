"""Phase 3b drift-monitoring tests.

Layered so the fast, deterministic parts run everywhere with no Evidently and no subprocess:

- buffer / window-trigger logic            (pure; no Evidently)
- Evidently `.dict()` -> summary parser     (pure; drift_worker._summarize has no top-level
                                             Evidently import, so it imports fine)
- Prometheus gauge update                   (pure; prometheus_client only)
- worker end-to-end                         (integration; spawns the isolated worker process,
                                             skipped automatically if the worker can't run)

The parser fixtures mirror the real evidently==0.7.21 `.dict()` shape, confirmed by probing the
installed stack (DriftedColumnsCount.value = {count, share}; ValueDrift.value = float score).
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from prometheus_client import REGISTRY

from prediction_model.config import config
from prediction_model.monitoring import drift, metrics
from prediction_model.monitoring.drift_worker import _summarize


# --------------------------------------------------------------------------------------
# buffer / window-trigger logic (no Evidently)
# --------------------------------------------------------------------------------------

def _reset_buffer():
    drift._BUFFER.clear()
    drift._COUNT = 0


def _row(value=0.0):
    return {name: value for name in config.FEATURES}   # full row incl. Time (filtered out)


def test_record_inference_fires_every_window(monkeypatch):
    monkeypatch.setattr(config, "DRIFT_WINDOW", 5)
    _reset_buffer()
    fires = [drift.record_inference(_row(i)) for i in range(12)]
    # Fires exactly on the window boundaries (5th and 10th record -> indices 4 and 9).
    assert [i for i, f in enumerate(fires) if f] == [4, 9]


def test_record_inference_keeps_only_monitored_features(monkeypatch):
    monkeypatch.setattr(config, "DRIFT_WINDOW", 3)
    _reset_buffer()
    drift.record_inference(_row(1.0))
    buffered = drift._BUFFER[-1]
    assert "Time" not in buffered          # Time is excluded from drift monitoring
    assert set(buffered.keys()) == set(config.DRIFT_FEATURES)


# --------------------------------------------------------------------------------------
# Evidently .dict() -> summary parser (no Evidently; offline)
# --------------------------------------------------------------------------------------

def _fake_evidently_dict(share, per_col_scores, threshold=0.1):
    """Build a dict shaped like evidently 0.7.x run.dict() for DataDriftPreset."""
    drifted = sum(1 for v in per_col_scores.values() if v > threshold)
    metrics_list = [{
        "metric_name": "DriftedColumnsCount(drift_share=0.5)",
        "config": {"type": "evidently:metric_v2:DriftedColumnsCount", "drift_share": 0.5},
        "value": {"count": float(drifted), "share": share},
    }]
    for col, score in per_col_scores.items():
        metrics_list.append({
            "metric_name": f"ValueDrift(column={col},method=Wasserstein,threshold={threshold})",
            "config": {"type": "evidently:metric_v2:ValueDrift", "column": col,
                       "method": "Wasserstein", "threshold": threshold},
            "value": score,
        })
    return {"metrics": metrics_list, "tests": []}


def test_summarize_extracts_count_share_and_scores():
    features = ["V1", "V2", "Amount"]
    d = _fake_evidently_dict(share=0.667, per_col_scores={"V1": 4.0, "V2": 0.02, "Amount": 7.5})
    s = _summarize(d, features, alert_share=0.3)

    assert s["drift_share"] == pytest.approx(0.667)
    assert s["drifted_columns_count"] == 2            # V1 and Amount exceed 0.1
    assert s["n_features"] == 3
    assert s["dataset_drift_detected"] is True        # 0.667 >= 0.3
    assert s["per_feature"]["Amount"] == pytest.approx(7.5)
    assert s["per_feature"]["V2"] == pytest.approx(0.02)


def test_summarize_no_drift_below_threshold():
    features = ["V1", "V2", "Amount"]
    d = _fake_evidently_dict(share=0.0, per_col_scores={"V1": 0.01, "V2": 0.02, "Amount": 0.03})
    s = _summarize(d, features, alert_share=0.3)
    assert s["dataset_drift_detected"] is False
    assert s["drifted_columns_count"] == 0


def test_summarize_fills_missing_features_with_zero():
    # A feature with no ValueDrift entry still gets a gauge value (0.0), so the dashboard
    # doesn't hold a stale reading for it.
    d = _fake_evidently_dict(share=0.5, per_col_scores={"V1": 4.0})
    s = _summarize(d, ["V1", "V2", "Amount"], alert_share=0.3)
    assert s["per_feature"]["V2"] == 0.0
    assert s["per_feature"]["Amount"] == 0.0


# --------------------------------------------------------------------------------------
# Prometheus gauge update (prometheus_client only)
# --------------------------------------------------------------------------------------

def test_update_from_summary_sets_gauges():
    summary = {
        "dataset_drift_detected": True,
        "drift_share": 0.42,
        "drifted_columns_count": 12,
        "window_size": 500,
        "timestamp": 1234.0,
        "per_feature": {"V14": 8.5, "Amount": 1.9},
    }
    metrics.update_from_summary(summary)

    assert REGISTRY.get_sample_value("fraud_input_drift_dataset_detected") == 1.0
    assert REGISTRY.get_sample_value("fraud_input_drift_share") == pytest.approx(0.42)
    assert REGISTRY.get_sample_value("fraud_input_drifted_columns_count") == 12.0
    assert REGISTRY.get_sample_value("fraud_input_drift_window_size") == 500.0
    assert REGISTRY.get_sample_value(
        "fraud_input_feature_drift_score", {"feature": "V14"}) == pytest.approx(8.5)


# --------------------------------------------------------------------------------------
# worker end-to-end (integration; spawns the isolated Evidently process)
# --------------------------------------------------------------------------------------

def _make_synthetic_reference(path, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    pd.DataFrame({c: rng.normal(0, 1, n) for c in cols}).to_csv(path, index=False)
    return cols


def test_worker_end_to_end_detects_broad_shift(monkeypatch):
    """Drive the real isolated worker: a broadly shifted current window must trip dataset
    drift. Skips (not fails) if the worker process can't run in this environment."""
    with tempfile.TemporaryDirectory() as tmp:
        ref_path = os.path.join(tmp, "reference.csv")
        cols = _make_synthetic_reference(ref_path)
        monkeypatch.setattr(config, "DRIFT_REFERENCE_FILE", ref_path)

        rng = np.random.default_rng(1)
        # Shift ~half the columns hard so the drifted share clears the 0.3 dataset threshold.
        current = {c: rng.normal(0, 1, 400) for c in cols}
        for c in cols[:15]:
            current[c] = current[c] + 6.0
        current_df = pd.DataFrame(current)

        try:
            summary = drift.compute_drift(current_df)
        except Exception as exc:  # worker unavailable in this env
            pytest.skip(f"drift worker could not run: {exc}")

        assert summary["dataset_drift_detected"] is True
        assert summary["drift_share"] >= 0.3
        assert summary["drifted_columns_count"] >= 10
