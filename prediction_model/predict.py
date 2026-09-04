import pandas as pd
import numpy as np
from prediction_model.config import config
from prediction_model import explain
import mlflow
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)


# Cache the loaded model AND its SHAP explainer, keyed on the resolved best run_id.
# We rebuild only when the winning run changes, so the (expensive) model load and
# explainer construction are amortized to once-per-model-version instead of per request.
#
# NOTE (Phase 3a): this is deliberately the same "cache-until-run_id-changes" pattern
# that should eventually replace the per-request MLflow *model reload* P0 itself
# (see PROJECT_LOG Decisions Log). The explainer just forced us to build the mechanism
# first. The lightweight run resolution below still runs per request — pinning a
# registry stage to drop that too remains the deferred next step.
_CACHE = {"run_id": None, "model": None, "explainer": None}


# Resolution is cached with a TTL, separately from the model cache above.
#
# MEASURED, not assumed: _resolve_best_run_id() cost 3053 ms/call against a 392-file store
# while model.predict() cost 17 ms - a 180x overhead paid on EVERY request, so ~99.4% of
# request latency was MLflow re-reading run metadata to answer a question whose answer had
# not changed. It capped throughput at ~1.4 req/s regardless of concurrency (the work is
# CPU-bound Python under the GIL, so adding workers did not help). See SLO.md.
#
# The tradeoff this makes explicit: a newly trained better run is now picked up within
# MODEL_RESOLVE_TTL seconds rather than on the very next request. For a fraud model
# retrained on a daily-or-slower cadence, five minutes of staleness is irrelevant; three
# seconds of latency per prediction is not.
_RESOLVE_TTL = float(os.getenv("MODEL_RESOLVE_TTL", "300"))
_RESOLVE_CACHE = {"run_id": None, "at": 0.0}
_RESOLVE_LOCK = threading.Lock()


def _search_best_run_id():
    """Resolve the current 'best' run by PR-AUC (the right criterion for this imbalance)."""
    experiment = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME)
    runs_df = mlflow.search_runs(experiment_ids=experiment.experiment_id,
                                 order_by=['metrics.pr_auc DESC'])
    return runs_df.iloc[0]['run_id']


def _resolve_best_run_id(force=False):
    """Best run id, re-resolved at most once per _RESOLVE_TTL seconds.

    The lock is held across the MLflow call deliberately. Under concurrency a plain
    check-then-act would let every waiting thread miss simultaneously and all of them run
    the expensive scan - a cache stampede that reproduces exactly the behaviour this cache
    exists to remove. One thread resolves; the rest wait and then find a warm entry.
    """
    now = time.monotonic()
    with _RESOLVE_LOCK:
        fresh = (_RESOLVE_CACHE["run_id"] is not None
                 and (now - _RESOLVE_CACHE["at"]) < _RESOLVE_TTL)
        if fresh and not force:
            return _RESOLVE_CACHE["run_id"]

        run_id = _search_best_run_id()
        _RESOLVE_CACHE.update(run_id=run_id, at=time.monotonic())
        return run_id


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


# --- Readiness surface (Phase 4 SRE) -----------------------------------------------
# These two exist purely so the container can answer "can I serve traffic yet?" without
# the caller having to send a real prediction. They wrap the cache above and add no
# inference logic of their own.

def is_model_ready():
    """True once a model is resident in the cache.

    Deliberately reads _CACHE directly rather than calling _get_model_and_explainer():
    that would run _resolve_best_run_id() -> mlflow.search_runs(), a full scan of the
    MLflow file store, on *every* readiness probe (6/min/pod, forever). A probe must be
    O(1) and touch nothing but memory.
    """
    return _CACHE["model"] is not None


def warm_model():
    """Load the model once at startup so the first real request doesn't pay cold start.

    Returns True on success. Never raises: a serving process that cannot reach MLflow
    should come up live-but-unready (so /readyz reports 503 and no traffic is routed to
    it) rather than crash-loop. The bare `except Exception` is load-bearing —
    _resolve_best_run_id() raises AttributeError, not a clean lookup error, when the
    experiment is absent (mlflow.get_experiment_by_name returns None).
    """
    try:
        _get_model_and_explainer()
        return True
    except Exception:  # noqa: BLE001 - see docstring; readiness must degrade, not crash
        logger.exception("model warmup failed; service will report not-ready")
        return False


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
