"""Phase 2 regression gate.

Rewritten from the baseline's single hard-coded assertion (`prediction == 'Y'`),
which broke the moment the MLflow "best run" changed and so was never a real gate.
These tests validate the request schema/types and the stratified split invariants —
pure, deterministic, and independent of any MLflow tracking server or the raw CSV.
"""
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from pydantic import ValidationError

from prediction_model.config import config
from prediction_model.schemas import TransactionFeatures
from prediction_model.processing.data_handling import stratified_split
from prediction_model import pipeline as pipe
from prediction_model.explain import build_explainer, explain_row


def _valid_row():
    return {name: 0.0 for name in config.FEATURES}


# ---- schema / type validation ----

def test_schema_accepts_valid_row():
    obj = TransactionFeatures(**_valid_row())
    assert set(obj.model_dump().keys()) == set(config.FEATURES)


def test_schema_rejects_missing_field():
    row = _valid_row()
    del row['Amount']
    with pytest.raises(ValidationError):
        TransactionFeatures(**row)


def test_schema_rejects_wrong_type():
    row = _valid_row()
    row['Amount'] = "not_a_number"
    with pytest.raises(ValidationError):
        TransactionFeatures(**row)


# ---- config / contract consistency ----

def test_feature_contract():
    assert config.TARGET == 'Class'
    assert len(config.FEATURES) == 30
    assert config.FEATURES[0] == 'Time' and config.FEATURES[-1] == 'Amount'
    # Pydantic schema fields must match the model's feature list exactly.
    assert set(TransactionFeatures.model_fields.keys()) == set(config.FEATURES)


# ---- stratified split invariants ----

@pytest.fixture
def imbalanced_df():
    rng = np.random.default_rng(0)
    n = 5000
    n_pos = 100  # 2% positives
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1
    rng.shuffle(y)
    data = {col: rng.normal(size=n) for col in config.FEATURES}
    data[config.TARGET] = y
    return pd.DataFrame(data)


def test_split_preserves_class_ratio(imbalanced_df):
    _, _, y_train, y_test = stratified_split(imbalanced_df)
    full_rate = imbalanced_df[config.TARGET].mean()
    assert y_train.mean() == pytest.approx(full_rate, abs=0.003)
    assert y_test.mean() == pytest.approx(full_rate, abs=0.003)


def test_split_is_deterministic(imbalanced_df):
    a = stratified_split(imbalanced_df)
    b = stratified_split(imbalanced_df)
    assert list(a[0].index) == list(b[0].index)   # X_train indices identical
    assert list(a[1].index) == list(b[1].index)   # X_test indices identical


# ---- local SHAP explanation (offline; also pins the SHAP return-shape contract) ----

@pytest.fixture
def fitted_fraud_pipeline(imbalanced_df):
    """The real preprocessing pipeline + a small XGBoost classifier, fit on synthetic
    data — no MLflow, no CSV, so this stays a pure offline gate."""
    clf = xgb.XGBClassifier(n_estimators=30, max_depth=3, scale_pos_weight=50,
                            eval_metric='aucpr', random_state=config.RANDOM_STATE)
    model = pipe.build_pipeline(clf)
    model.fit(imbalanced_df[config.FEATURES], imbalanced_df[config.TARGET])
    return model


def test_explain_row_shape_and_ranking(fitted_fraud_pipeline, imbalanced_df):
    model = fitted_fraud_pipeline
    explainer = build_explainer(model)
    row = imbalanced_df[config.FEATURES].iloc[[0]]

    result = explain_row(model, explainer, row, top_n=config.SHAP_TOP_N)

    # base_value is a single float (margin space); top_features is top-N.
    assert isinstance(result['base_value'], float)
    assert len(result['top_features']) == config.SHAP_TOP_N

    absvals = [abs(f['shap_value']) for f in result['top_features']]
    assert absvals == sorted(absvals, reverse=True)   # ranked by |shap| descending

    for f in result['top_features']:
        assert set(f.keys()) == {'feature', 'value', 'shap_value', 'direction'}
        assert f['feature'] in config.FEATURES
        expected = 'increases_fraud_risk' if f['shap_value'] > 0 else 'decreases_fraud_risk'
        assert f['direction'] == expected


def test_explain_row_respects_top_n(fitted_fraud_pipeline, imbalanced_df):
    model = fitted_fraud_pipeline
    explainer = build_explainer(model)
    row = imbalanced_df[config.FEATURES].iloc[[1]]

    assert len(explain_row(model, explainer, row, top_n=3)['top_features']) == 3


def test_best_run_id_is_cached_and_refreshed_in_background(monkeypatch):
    # Serving must not re-query MLflow inline on requests (seconds each on a bind mount):
    # only the first call queries; an expired entry is served while a background refresh runs.
    from prediction_model import predict
    calls = []
    monkeypatch.setattr(predict, "_query_best_run_id", lambda: calls.append(1) or f"run{len(calls)}")
    monkeypatch.setattr(predict, "_RUN_ID_CACHE", {"run_id": None, "resolved_at": 0.0, "refresh": None})
    clock = [1000.0]
    monkeypatch.setattr(predict.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(config, "RUN_ID_CACHE_SECONDS", 60)

    assert predict._resolve_best_run_id() == "run1"
    clock[0] += 59
    assert predict._resolve_best_run_id() == "run1"   # within TTL: no re-query
    assert len(calls) == 1

    clock[0] += 1
    assert predict._resolve_best_run_id() == "run1"   # expired: stale value served immediately
    refresh = predict._RUN_ID_CACHE["refresh"]
    if refresh is not None:
        refresh.join(timeout=5)
    assert len(calls) == 2
    assert predict._resolve_best_run_id() == "run2"   # background refresh landed
