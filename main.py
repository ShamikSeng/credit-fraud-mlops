from fastapi import FastAPI , File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from prediction_model.predict import (generate_predictions,
                                       generate_prediction_with_explanation,
                                       generate_predictions_batch,
                                       is_model_ready,
                                       warm_model)
from prediction_model.config import config
from prediction_model.logging_config import configure_logging
from prediction_model.schemas import TransactionFeatures, PredictionResponse
# Import metrics at startup so the drift gauges register on the default Prometheus registry
# and appear on /metrics from the first scrape. `drift` imports NO Evidently (that lives in a
# separate worker process), so this is safe on the serving path.
from prediction_model.monitoring import metrics as drift_metrics  # noqa: F401
from prediction_model.monitoring import drift
import mlflow
import io
import os
import threading
from datetime import datetime
from prometheus_fastapi_instrumentator import Instrumentator


def save_batch_prediction(file_content, filename):
    current_date = datetime.now().strftime("%Y-%m-%d")
    if filename.endswith('.csv'):
        filename = filename[:-4]

    # Colons are illegal in Windows filenames, so use %H-%M-%S rather than %H:%M:%S.
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = os.path.join(config.BATCH_OUTPUT_DIR, current_date)
    os.makedirs(folder_path, exist_ok=True)

    filename_with_datetime = f"{filename}_{current_datetime}.csv"
    output_path = os.path.join(folder_path, filename_with_datetime)

    with open(output_path, "wb") as f:
        f.write(file_content)

    return output_path

# Configure logging before anything else emits a record, so no line escapes in the
# default plain-text format.
configure_logging()

mlflow.set_tracking_uri(config.TRACKING_URI)

app = FastAPI(
    title="Credit Card Fraud Detection App using FastAPI - MLOps",
    description = "MLOps Demo",
    version='1.0'
)

origins=[
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

Instrumentator().instrument(app).expose(app)


def _label(prediction):
    return "Fraud" if prediction == "fraud" else "Legitimate"


def _record_and_maybe_check(data):
    """Buffer this inference input for drift monitoring; when the window fills, queue a
    drift recompute on the drift executor.

    NOT a BackgroundTask. Starlette awaits background tasks inside the response object,
    which is still inside the Prometheus middleware's timing block, so every 500th request
    would record the Evidently subprocess wall time as if the client had waited for it.
    drift.schedule_drift_check() hands off to a separate single-thread executor and
    returns immediately, keeping the measured latency the latency users actually see.
    """
    if not config.DRIFT_ENABLED:
        return
    if drift.record_inference(data):
        drift.schedule_drift_check()


@app.on_event("startup")
def _warm_model_on_startup():
    """Load the model once at startup, off the event loop, so the first real request
    doesn't pay the MLflow-scan + model-load + SHAP-explainer cold start.

    Dispatched to a daemon thread rather than awaited: uvicorn binds the port
    immediately and /readyz flips 503 -> 200 when the load finishes. Blocking here
    would just move cold start into an initialDelaySeconds guessing game in the
    Kubernetes probe config.
    """
    threading.Thread(target=warm_model, name="model-warmup", daemon=True).start()


@app.get("/")
def index():
    return {"message":"Welcome to the MLOps Credit Card Fraud Detection app" }


# --- Probes (Phase 4 SRE) ----------------------------------------------------------
# Both are `async def` ON PURPOSE. FastAPI dispatches every `def` path operation into
# the shared AnyIO threadpool (default 40 slots), and so does Starlette's BackgroundTask
# for sync callables. Sync probes would therefore queue behind saturated request work
# and time out — which makes the kubelet restart a pod that is merely busy, converting
# a latency problem into an availability one. `async def` handlers run on the event loop
# and must consequently touch nothing but memory: no I/O, no model load, no MLflow.

@app.get("/healthz")
async def healthz():
    """Liveness: is this process alive and serving the event loop? Nothing more.

    Deliberately does NOT check the model. Liveness failure means 'restart me', and
    restarting will not conjure an absent model artifact — it would just crash-loop.
    Model availability is a readiness concern; see /readyz.
    """
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    """Readiness: should this pod receive traffic yet?

    503 until the model is resident, so a pod with an unreachable MLflow store stays up
    and diagnosable but is pulled out of the Service endpoints instead of serving 500s.
    A pulled image with no mlruns/ mounted reports 503 here by design.
    """
    if is_model_ready():
        return {"status": "ready", "model_loaded": True}
    return JSONResponse(
        status_code=503,
        content={"status": "not_ready", "model_loaded": False},
    )


@app.post("/prediction_api", response_model=PredictionResponse)
def predict(transaction: TransactionFeatures):
    data = transaction.model_dump()
    result = generate_prediction_with_explanation([data])
    _record_and_maybe_check(data)
    return {
        "status": _label(result["prediction"]),
        "fraud_probability": result["fraud_probability"],
        "explanation": result["explanation"],   # None unless fraud-flagged
    }


@app.post("/prediction_ui")
def predict_gui(transaction: TransactionFeatures):
    # Ordered row keyed by the single source of truth for the feature list.
    data = transaction.model_dump()
    row = {col: data[col] for col in config.FEATURES}
    prediction = generate_predictions([row])["prediction"][0]
    _record_and_maybe_check(data)
    return {"status": _label(prediction)}


@app.post("/monitoring/drift")
def run_drift():
    """Force a drift computation now over the currently buffered inference inputs and update
    the Prometheus gauges. Synchronous so the caller gets the summary back (useful for demos
    and verification)."""
    try:
        summary = drift.run_drift_check()
    except Exception as exc:  # noqa: BLE001 - surface the reason to the caller
        return {"status": "error", "detail": str(exc)}
    if summary is None:
        return {"status": "no_data", "detail": "No inference inputs buffered yet."}
    return summary


@app.get("/monitoring/drift")
def last_drift():
    """Return the most recent drift summary (the auto-recompute runs every DRIFT_WINDOW inputs)."""
    summary = drift.get_last_summary()
    if summary is None:
        return {"status": "no_run", "detail": "No drift run has completed yet."}
    return summary


@app.post("/batch_prediction")
async def batch_predict(file: UploadFile = File(...)):

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content),index_col=False)

    # Ensure the CSV file contains the required features
    required_columns = config.FEATURES
    if not all(column in df.columns for column in required_columns):
        return {"error": "CSV file does not contain the required columns."}

    predictions = generate_predictions_batch(df)["prediction"]

    df['Prediction'] = predictions
    result = df.to_csv(index=False)

    save_batch_prediction(result.encode('utf-8'), file.filename)

    return StreamingResponse(io.BytesIO(result.encode('utf-8')), media_type="text/csv", headers={"Content-Disposition":"attachment; filename=predictions.csv"})



if __name__== "__main__":
    # log_config=None is required, not optional: uvicorn otherwise calls
    # logging.config.dictConfig(LOGGING_CONFIG) during startup, which runs AFTER
    # configure_logging() at import time and replaces the root handlers - silently
    # reverting the whole stream to uvicorn's plain-text format. With it disabled,
    # uvicorn's own loggers propagate to the root handler and access logs are JSON too.
    uvicorn.run(app, host="0.0.0.0", port=8005, log_config=None)
