"""
FastAPI application exposing the Customer Churn MAS pipeline.

Endpoints
---------
GET  /health           - liveness check
POST /train             - retrain the model from an uploaded CSV (or the default dataset)
POST /predict            - score an uploaded CSV of customers and return churn risk
GET  /model/metadata     - return metadata about the currently registered model
"""

import os
import shutil
import tempfile
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import DEFAULT_TRAIN_FILE, METADATA_PATH, MODEL_PATH
from src.graphs.prediction_graph import run_prediction
from src.graphs.training_graph import run_training

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Multi-agent (LangGraph) churn prediction pipeline",
    version="1.0.0",
)

# Allow the React dev server (and any frontend) to call this API directly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model_available": os.path.exists(MODEL_PATH)}


@app.post("/train")
async def train(file: Optional[UploadFile] = File(None)):
    """
    Run the full training graph (ingestion -> cleaning -> feature engineering ->
    training -> evaluation -> registry).

    If no file is uploaded, the default Telco churn dataset bundled with the
    project is used.
    """
    tmp_path = DEFAULT_TRAIN_FILE

    if file is not None:
        suffix = os.path.splitext(file.filename)[1] or ".csv"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

    try:
        result = run_training(tmp_path)
    finally:
        if file is not None and os.path.exists(tmp_path):
            os.remove(tmp_path)

    if result.get("errors"):
        raise HTTPException(status_code=400, detail=result["errors"])

    return JSONResponse(
        {
            "model_name": result["model_name"],
            "metrics": result["evaluation_report"]["metrics"],
            "passed_quality_gate": result["evaluation_report"]["passed"],
            "candidate_results": result["candidate_results"],
            "registry": result["registry_report"],
            "logs": result["logs"],
        }
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Score an uploaded CSV of customers and return churn probability + risk level
    for each customer.
    """
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=400, detail="No trained model found. Call /train first.")

    suffix = os.path.splitext(file.filename)[1] or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = run_prediction(tmp_path)
    finally:
        os.remove(tmp_path)

    if result.get("errors"):
        raise HTTPException(status_code=400, detail=result["errors"])

    predictions = result["predictions_df"].to_dict(orient="records")

    return JSONResponse(
        {
            "n_customers": result["prediction_report"]["n_customers"],
            "risk_level_counts": result["prediction_report"]["risk_level_counts"],
            "avg_churn_probability": result["prediction_report"]["avg_churn_probability"],
            "predictions": predictions,
        }
    )


@app.get("/model/metadata")
def model_metadata():
    if not os.path.exists(METADATA_PATH):
        raise HTTPException(status_code=404, detail="No model has been trained yet.")

    import json

    with open(METADATA_PATH) as f:
        return json.load(f)
