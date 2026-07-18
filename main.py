from fastapi import FastAPI , File, UploadFile, BackgroundTasks
from fastapi.responses import StreamingResponse
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
import boto3
from datetime import datetime
from prometheus_fastapi_instrumentator import Instrumentator


def upload_to_s3(file_content, filename):
    s3 = boto3.client('s3')

    current_date = datetime.now().strftime("%Y-%m-%d")
    if filename.endswith('.csv'):
        filename = filename[:-4]

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    folder_path = f"{config.FOLDER}/{current_date}"

    filename_with_datetime = f"{filename}_{current_datetime}.csv"

    s3_key = f"{folder_path}/{filename_with_datetime}"

    response = s3.put_object(Bucket=config.S3_BUCKET, Key=s3_key, Body=file_content)

    return s3_key

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

@app.post("/prediction_api", response_model=PredictionResponse)
def predict(transaction: TransactionFeatures, background_tasks: BackgroundTasks):
    data = transaction.model_dump()
    result = generate_prediction_with_explanation([data])
    _record_and_maybe_check(data, background_tasks)
    return {
        "status": _label(result["prediction"]),
        "fraud_probability": result["fraud_probability"],
        "explanation": result["explanation"],   # None unless fraud-flagged
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

    predictions = generate_predictions_batch(df)["prediction"]

    df['Prediction'] = predictions
    result = df.to_csv(index=False)

    s3_key = upload_to_s3(result.encode('utf-8'), file.filename)

    return StreamingResponse(io.BytesIO(result.encode('utf-8')), media_type="text/csv", headers={"Content-Disposition":"attachment; filename=predictions.csv"})



if __name__== "__main__":
    uvicorn.run(app, host="0.0.0.0",port=8005)
