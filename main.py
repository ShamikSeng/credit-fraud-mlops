from fastapi import FastAPI , File, UploadFile, BackgroundTasks
import json
from fastapi.responses import StreamingResponse, FileResponse, RedirectResponse
from fastapi import HTTPException
import uvicorn
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from prediction_model.predict import (generate_predictions,
                                       generate_prediction_with_explanation,
                                       generate_predictions_batch)
from prediction_model.config import config
from prediction_model.schemas import TransactionFeatures, PredictionResponse
# Import metrics at startup so the drift gauges register on the default Prometheus registry
# and appear on /metrics from the first scrape. `drift` imports NO Evidently (that lives in a
# separate worker process), so this is safe on the serving path.
from prediction_model.monitoring import metrics as drift_metrics  # noqa: F401
from prediction_model.monitoring import drift
import mlflow
import io
import os
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

mlflow.set_tracking_uri(config.TRACKING_URI)

app = FastAPI(
    title=f"Fraud Detection API - {config.DATASET_TITLE}",
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


def _record_and_maybe_check(data, background_tasks):
    """Buffer this inference input for drift monitoring; when the window fills, schedule a
    drift recompute as a BackgroundTask (off the request's latency path)."""
    if not config.DRIFT_ENABLED:
        return
    if drift.record_inference(data):
        background_tasks.add_task(drift.safe_run_drift_check)


@app.get("/")
def index():
    return {"message":"Welcome to the MLOps Credit Card Fraud Detection app" }


# Demo frontend: a single static page that calls the endpoints below from the same origin.
# `/` stays JSON (health check); the UI lives at /ui.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_INDEX = os.path.join(BASE_DIR, "frontend", "index.html")
# ULB: 200 held-out test rows (185 legit + 15 fraud, with Class) so batch scoring can be checked.
# UPI: 300 held-out payments from days 60-89 (30 fraud, with is_fraud and fraud_type).
SAMPLE_BATCH = os.path.join(BASE_DIR, "frontend",
                            "sample_batch.csv" if config.DATASET == "ulb" else "upi_sample_batch.csv")
PRESETS_FILE = os.path.join(BASE_DIR, "frontend", f"presets_{config.DATASET}.json")
GRAFANA_DASHBOARD = "fraud-input-drift" if config.DATASET == "ulb" else "upi-input-drift"


@app.get("/ui", include_in_schema=False)
def ui():
    return FileResponse(FRONTEND_INDEX)


@app.get("/ui/", include_in_schema=False)
def ui_slash():
    return RedirectResponse("/ui")


@app.get("/ui/sample_batch.csv", include_in_schema=False)
def ui_sample_batch():
    if not os.path.exists(SAMPLE_BATCH):
        raise HTTPException(status_code=404, detail=f"{os.path.basename(SAMPLE_BATCH)} not found")
    return FileResponse(SAMPLE_BATCH, media_type="text/csv")


@app.get("/ui/schema", include_in_schema=False)
def ui_schema():
    """Everything the console needs to render the form for whichever dataset this service runs."""
    fields = [{"name": f, "type": "category", "options": config.CATEGORICAL_FEATURES[f]}
              if f in config.CATEGORICAL_FEATURES else {"name": f, "type": "number"}
              for f in config.FEATURES]
    fields += [{"name": f, "type": "number", "optional": True} for f in config.METADATA_FIELDS]
    presets = {}
    if os.path.exists(PRESETS_FILE):
        with open(PRESETS_FILE, encoding="utf-8") as fh:
            presets = json.load(fh)
    return {
        "dataset": config.DATASET,
        "title": config.DATASET_TITLE,
        "fields": fields,
        "primary": ["Amount", "Time"] if config.DATASET == "ulb" else [f["name"] for f in fields],
        "target": config.TARGET,
        "presets": presets,
        "drift_window": config.DRIFT_WINDOW,
        "drift_features": len(config.DRIFT_FEATURES),
        "grafana_dashboard": GRAFANA_DASHBOARD,
    }

@app.post("/prediction_api", response_model=PredictionResponse)
def predict(transaction: TransactionFeatures, background_tasks: BackgroundTasks):
    data = transaction.model_dump()
    result = generate_prediction_with_explanation([data])
    _record_and_maybe_check(data, background_tasks)
    return {
        "status": _label(result["prediction"]),
        "fraud_probability": result["fraud_probability"],
        "explanation": result["explanation"],   # None unless fraud-flagged
        "model_version": result["model_version"],
    }


@app.post("/prediction_ui")
def predict_gui(transaction: TransactionFeatures, background_tasks: BackgroundTasks):
    # Ordered row keyed by the single source of truth for the feature list.
    data = transaction.model_dump()
    row = {col: data[col] for col in config.FEATURES}
    prediction = generate_predictions([row])["prediction"][0]
    _record_and_maybe_check(data, background_tasks)
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

    scored = generate_predictions_batch(df)

    df['Prediction'] = scored["prediction"]
    result = df.to_csv(index=False)

    save_batch_prediction(result.encode('utf-8'), file.filename)

    return StreamingResponse(io.BytesIO(result.encode('utf-8')), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=predictions.csv",
                                      "X-Model-Version": str(scored["model_version"])})



if __name__== "__main__":
    uvicorn.run(app, host="0.0.0.0",port=8005)
