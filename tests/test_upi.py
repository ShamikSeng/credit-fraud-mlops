"""UPI extension tests: simulator, request schema, one-hot encoding and explanations.

All offline: a small simulated stream stands in for the dataset, and nothing touches MLflow
or Evidently, so these run in CI and inside the Docker build like the rest of the suite.
"""
import pandas as pd
import pytest
from pydantic import ValidationError

from prediction_model.explain import build_explainer, explain_row
from prediction_model.processing.preprocessing import OneHotTransformer
from prediction_model.schemas import build_transaction_model
from prediction_model.upi import modeling, spec
from prediction_model.upi.simulate import simulate


@pytest.fixture(scope="module")
def stream():
    # 16 days, drift from day 8: small enough to be fast, big enough to contain both fraud types.
    return simulate(n_days=16, seed=3, n_users=3000, base_daily=1500, drift_day=8)


@pytest.fixture(scope="module")
def fitted(stream):
    return modeling.fit_model(modeling.time_slice(stream, 0, 12))


def test_simulator_schema_order_and_drift(stream):
    assert list(stream.columns[:3]) == ["txn_id", "timestamp", spec.TIME_COL]
    assert set(spec.FEATURES) <= set(stream.columns)
    assert stream[spec.TIME_COL].is_monotonic_increasing
    for col, allowed in spec.CATEGORIES.items():
        assert set(stream[col]) <= set(allowed)
    pre = stream[stream.txn_time < 8]
    post = stream[stream.txn_time >= 8]
    # Before the drift only account takeover exists; afterwards social engineering appears.
    assert set(pre[pre.is_fraud == 1].fraud_type) == {"account_takeover"}
    assert "social_engineering" in set(post[post.is_fraud == 1].fraud_type)


def test_simulator_is_deterministic():
    a = simulate(n_days=3, seed=11, n_users=500, base_daily=200)
    b = simulate(n_days=3, seed=11, n_users=500, base_daily=200)
    pd.testing.assert_frame_equal(a, b)


def _valid_row(stream):
    row = stream.iloc[0]
    return {f: (row[f].item() if hasattr(row[f], "item") else row[f]) for f in spec.FEATURES}


def test_upi_schema_validates_categories(stream):
    Model = build_transaction_model(spec.FEATURES, spec.CATEGORIES, [spec.TIME_COL])
    row = _valid_row(stream)
    assert Model(**row).txn_time is None            # metadata is optional
    with pytest.raises(ValidationError):
        Model(**{**row, "merchant_category": "casino"})   # not an allowed category
    with pytest.raises(ValidationError):
        Model(**{**row, "amount": "abc"})


def test_one_hot_uses_fixed_categories():
    X = pd.DataFrame({"upi_app": ["Paytm", "Unknown"], "amount": [1.0, 2.0]})
    out = OneHotTransformer(categories={"upi_app": spec.CATEGORIES["upi_app"]}).fit(X).transform(X)
    assert "upi_app" not in out.columns
    assert [c for c in out.columns if c.startswith("upi_app=")] == \
        [f"upi_app={v}" for v in spec.CATEGORIES["upi_app"]]
    assert out.loc[0, "upi_app=Paytm"] == 1.0
    assert out.filter(like="upi_app=").iloc[1].sum() == 0.0   # unseen value -> all zeros


def test_explanations_are_per_input_feature_and_additive(stream, fitted):
    row = stream[stream.is_fraud == 1].iloc[[0]][spec.FEATURES]
    explainer = build_explainer(fitted)
    result = explain_row(fitted, explainer, row, top_n=len(spec.FEATURES))
    names = [f["feature"] for f in result["top_features"]]
    assert sorted(names) == sorted(spec.FEATURES)          # one entry per input, no dummies
    by_name = {f["feature"]: f for f in result["top_features"]}
    assert by_name["upi_app"]["value"] == row.iloc[0]["upi_app"]   # category name, not 0/1
    # base value + all contributions == the model's margin for this row (SHAP additivity).
    margin = fitted.named_steps["classifier"].predict(
        fitted[:-1].transform(row), output_margin=True)[0]
    total = result["base_value"] + sum(f["shap_value"] for f in result["top_features"])
    assert total == pytest.approx(float(margin), abs=1e-3)


def test_time_split_keeps_holdout_after_training(stream):
    train = modeling.time_slice(stream, 0, 12)
    holdout = modeling.time_slice(stream, 12, 16)
    assert train[spec.TIME_COL].max() < 12 <= holdout[spec.TIME_COL].min()
    assert len(train) + len(holdout) == len(stream)
