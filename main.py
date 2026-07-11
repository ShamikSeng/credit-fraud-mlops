from fastapi import FastAPI , File, UploadFile
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


@app.get("/")
def index():
    return {"message":"Welcome to the MLOps Credit Card Fraud Detection app" }

@app.post("/prediction_api", response_model=PredictionResponse)
def predict(transaction: TransactionFeatures):
    data = transaction.model_dump()
    result = generate_prediction_with_explanation([data])
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
    return {"status": _label(prediction)}


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
