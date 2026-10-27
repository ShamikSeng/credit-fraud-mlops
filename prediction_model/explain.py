"""Local SHAP explainability for the fraud model (Phase 3a).

Isolated here (like schemas.py) so the fast-moving SHAP API lives in one small,
reviewable module and the serving path imports a stable surface.

Why TreeExplainer, not KernelExplainer: XGBoost is a tree ensemble, so TreeSHAP
computes *exact* Shapley values in polynomial time and is natively supported.
KernelExplainer is the model-agnostic fallback — background-sample perturbation,
orders of magnitude slower, and only approximate — so it is the wrong tool when an
exact tree method exists.

Pipeline wrinkle: the logged artifact is a sklearn Pipeline
(MeanImputer -> RobustScaler -> XGBClassifier). TreeExplainer must wrap the *bare
classifier* and see data in the model's input space, so we transform the row through
`model[:-1]` (all steps except the classifier) before explaining. Both transformers
preserve the DataFrame columns/order, so contributor names stay Time / V1..V28 / Amount.

Units: TreeExplainer's default output is the model's **log-odds (margin)** space, not
probability. `base_value + sum(all shap values) = model margin`. The serving layer pairs
these with a separate `fraud_probability` for human readability (see schemas.py).

API-shape note (verified against shap==0.44.1 / xgboost==2.0.3): for this binary
classifier `shap_values()` returns a single (n, n_features) ndarray and
`expected_value` is a scalar. The normalizers below still defensively handle the
list / 3-D / per-class-array forms other SHAP versions can return, and the offline
test in tests/ pins the actual behavior.
"""
import numpy as np
import shap


def build_explainer(model):
    """Build a TreeExplainer over the pipeline's fitted XGBoost classifier step."""
    return shap.TreeExplainer(model.named_steps['classifier'])


def _positive_class_shap(shap_values):
    """Normalize a SHAP output to a 2-D (n_samples, n_features) positive-class array.

    Handles the shapes different SHAP/XGBoost versions can return:
    a list [class0, class1], a 3-D (n, features, classes) array, or the plain
    2-D (n, features) array this version already gives us.
    """
    if isinstance(shap_values, list):
        # Binary: positive class is the last element.
        return np.asarray(shap_values[-1])
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[..., -1]
    return arr


def _base_value(expected_value):
    """Normalize expected_value to a single float (base log-odds margin)."""
    ev = np.asarray(expected_value)
    if ev.ndim == 0:
        return float(ev)
    # Per-class base values -> positive class (last).
    return float(ev.flat[-1])


def _source_feature(column):
    """Model column -> the input feature it came from. One-hot columns are named
    "<feature>=<value>" (OneHotTransformer); every other column maps to itself."""
    return column.split('=', 1)[0]


def _display_value(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def explain_row(model, explainer, X_row, top_n):
    """Explain a single transaction.

    X_row: one-row DataFrame in the raw feature space (config.FEATURES columns).
    Returns {"base_value": float, "top_features": [ {feature, value, shap_value,
    direction}, ... ]} — the top_n features by |shap_value|, descending.

    SHAP values are additive, so the contribution of a categorical input is exactly the sum
    over its one-hot columns. Summing them reports "upi_app = Paytm" as one driver instead of
    five dummy columns, and keeps base_value + sum(contributions) = model margin.
    """
    X_transformed = model[:-1].transform(X_row)
    shap_row = _positive_class_shap(explainer.shap_values(X_transformed))[0]

    totals = {}
    for i, column in enumerate(X_transformed.columns):
        source = _source_feature(column)
        totals[source] = totals.get(source, 0.0) + float(shap_row[i])

    raw = X_row.iloc[0]
    contributions = [
        {
            'feature': feature,
            'value': _display_value(raw[feature]),   # raw submitted value (human-readable)
            'shap_value': total,                     # contribution in log-odds/margin space
            'direction': 'increases_fraud_risk' if total > 0 else 'decreases_fraud_risk',
        }
        for feature, total in totals.items()
    ]
    contributions.sort(key=lambda c: abs(c['shap_value']), reverse=True)

    return {
        'base_value': _base_value(explainer.expected_value),
        'top_features': contributions[:top_n],
    }
