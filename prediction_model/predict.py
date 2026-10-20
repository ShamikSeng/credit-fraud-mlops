import threading
import time
import pandas as pd
import numpy as np
from prediction_model.config import config
from prediction_model import explain
import mlflow


# Cache the loaded model AND its SHAP explainer, keyed on the resolved best run_id.
# We rebuild only when the winning run changes, so the (expensive) model load and
# explainer construction are amortized to once-per-model-version instead of per request.
#
# NOTE (Phase 3a): this is deliberately the same "cache-until-run_id-changes" pattern
# that should eventually replace the per-request MLflow *model reload* P0 itself
# (see PROJECT_LOG Decisions Log). The explainer just forced us to build the mechanism
# first. The run resolution below is now TTL-cached rather than per request — pinning a
# registry stage to drop it entirely remains the deferred next step.
_CACHE = {"run_id": None, "model": None, "explainer": None}

# The run resolution itself is cached (config.RUN_ID_CACHE_SECONDS): search_runs reads
# every run's meta.yaml, which made each request take seconds on the Docker bind mount.
# Once the TTL expires, requests keep getting the cached run_id while ONE background thread
# re-queries (stale-while-revalidate), so no request after the first ever pays that cost.
_RUN_ID_CACHE = {"run_id": None, "resolved_at": 0.0, "refresh": None}
_RUN_ID_LOCK = threading.Lock()


def _query_best_run_id():
    """Resolve the current 'best' run by PR-AUC (the right criterion for this imbalance)."""
    experiment = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME)
    runs_df = mlflow.search_runs(experiment_ids=experiment.experiment_id,
                                 order_by=['metrics.pr_auc DESC'])
    return runs_df.iloc[0]['run_id']


def _refresh_run_id():
    try:
        run_id = _query_best_run_id()
        with _RUN_ID_LOCK:
            _RUN_ID_CACHE.update(run_id=run_id, resolved_at=time.monotonic())
    except Exception:  # noqa: BLE001 - keep serving the cached run; retry after the next TTL
        with _RUN_ID_LOCK:
            _RUN_ID_CACHE["resolved_at"] = time.monotonic()
    finally:
        with _RUN_ID_LOCK:
            _RUN_ID_CACHE["refresh"] = None


def _resolve_best_run_id():
    """Best run_id. Only the very first call queries MLflow inline; after that, an expired
    entry is served as-is while a background thread re-queries."""
    if _RUN_ID_CACHE["run_id"] is None:
        run_id = _query_best_run_id()
        with _RUN_ID_LOCK:
            _RUN_ID_CACHE.update(run_id=run_id, resolved_at=time.monotonic())
        return run_id
    with _RUN_ID_LOCK:
        expired = time.monotonic() - _RUN_ID_CACHE["resolved_at"] >= config.RUN_ID_CACHE_SECONDS
        if expired and _RUN_ID_CACHE["refresh"] is None:
            _RUN_ID_CACHE["refresh"] = threading.Thread(target=_refresh_run_id, daemon=True)
            _RUN_ID_CACHE["refresh"].start()
        return _RUN_ID_CACHE["run_id"]


def _get_model_and_explainer():
    """Return (model, explainer) for the current best run, using the run_id-keyed cache."""
    run_id = _resolve_best_run_id()
    if run_id != _CACHE["run_id"] or _CACHE["model"] is None:
        model = mlflow.sklearn.load_model('runs:/' + run_id + config.MODEL_NAME)
        _CACHE.update(run_id=run_id, model=model, explainer=explain.build_explainer(model))
    return _CACHE["model"], _CACHE["explainer"]


def _load_best_model():
    """Model-only accessor (label paths). Shares the single cached load path above."""
    return _get_model_and_explainer()[0]


def generate_predictions(data_input):
    data = pd.DataFrame(data_input)[config.FEATURES]
    model = _load_best_model()
    prediction = model.predict(data)
    output = np.where(prediction == 1, 'fraud', 'legit')
    return {"prediction": output}


def generate_prediction_with_explanation(data_input):
    """Single-transaction prediction + fraud probability, with a local SHAP explanation
    attached only for fraud-flagged transactions (config.EXPLAIN_ONLY_FLAGGED).

    Explanations are inherently per-request (the explanation is of this specific
    transaction), so there is nothing to precompute — but the explainer itself is cached.
    """
    data = pd.DataFrame(data_input)[config.FEATURES]
    model, explainer = _get_model_and_explainer()

    fraud_probability = float(model.predict_proba(data)[:, 1][0])
    prediction = 'fraud' if model.predict(data)[0] == 1 else 'legit'

    explanation = None
    if prediction == 'fraud' or not config.EXPLAIN_ONLY_FLAGGED:
        explanation = explain.explain_row(model, explainer, data, config.SHAP_TOP_N)

    return {"prediction": prediction,
            "fraud_probability": fraud_probability,
            "explanation": explanation}


def generate_predictions_batch(data_input):
    # Slice to the model's expected columns/order so extra columns (e.g. the
    # ground-truth `Class`) don't ride into predict and misalign the feature count.
    # Batch stays label-only — per-row SHAP over a whole file is a heavier, separate concern.
    data = pd.DataFrame(data_input)[config.FEATURES]
    model = _load_best_model()
    prediction = model.predict(data)
    output = np.where(prediction == 1, 'fraud', 'legit')
    return {"prediction": output}
