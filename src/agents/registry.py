"""
Model Registry Agent
----------------------
Responsibilities:
  - Persist the trained pipeline (preprocessing + model) to disk via joblib
    so the Prediction Agent can load it without retraining
  - Save model metadata (metrics, model type, training timestamp, feature columns)
  - Register the model in the MLflow Model Registry (if it passed the quality gate)
  - Save reference statistics of the training data for the Monitoring/Drift Agent
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

import joblib
import mlflow

from src.config import (
    METADATA_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    MODELS_DIR,
    PIPELINE_PATH,
    REFERENCE_STATS_PATH,
    SKOPS_TRUSTED_TYPES,
)
from src.state import ChurnPipelineState


def _compute_reference_stats(df) -> Dict[str, Any]:
    numeric_df = df.select_dtypes(include="number")
    stats = {
        col: {
            "mean": float(numeric_df[col].mean()),
            "std": float(numeric_df[col].std()),
            "min": float(numeric_df[col].min()),
            "max": float(numeric_df[col].max()),
        }
        for col in numeric_df.columns
    }
    return stats


def registry_agent(state: ChurnPipelineState) -> ChurnPipelineState:
    """LangGraph node: persist the trained model and register it with MLflow."""
    logs = state.get("logs", [])
    errors = state.get("errors", [])

    required = ["model", "evaluation_report", "model_name", "candidate_results"]
    missing = [k for k in required if k not in state]
    if missing:
        errors.append(f"[registry] Missing state keys: {missing} - evaluation must run first.")
        return {**state, "errors": errors, "logs": logs}

    os.makedirs(MODELS_DIR, exist_ok=True)

    pipeline = state["model"]
    report = state["evaluation_report"]

    # --- Persist pipeline (preprocessing + model) -----------------------------
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(pipeline, PIPELINE_PATH)  # same object: ColumnTransformer is a pipeline step

    # --- Persist metadata -----------------------------------------------------
    metadata = {
        "model_name": state["model_name"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": report["metrics"],
        "candidate_results": state["candidate_results"],
        "passed_quality_gate": report["passed"],
        "feature_columns": list(state["X_test"].columns),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    # --- Persist reference stats for drift monitoring --------------------------
    if "feature_df" in state:
        ref_stats = _compute_reference_stats(state["feature_df"])
        with open(REFERENCE_STATS_PATH, "w") as f:
            json.dump(ref_stats, f, indent=2)

    # --- Register in MLflow Model Registry -------------------------------------
    registry_report: Dict[str, Any] = {
        "model_path": MODEL_PATH,
        "metadata_path": METADATA_PATH,
        "registered_in_mlflow": False,
    }

    if report["passed"]:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"register__{state['model_name']}"):
                mlflow.log_metrics(report["metrics"])
                mlflow.sklearn.log_model(
                    pipeline,
                    name="model",
                    registered_model_name=MLFLOW_REGISTERED_MODEL_NAME,
                    skops_trusted_types=SKOPS_TRUSTED_TYPES,
                )
            registry_report["registered_in_mlflow"] = True
            registry_report["registered_model_name"] = MLFLOW_REGISTERED_MODEL_NAME
        except Exception as exc:  # noqa: BLE001
            errors.append(f"[registry] MLflow registration failed: {exc}")

    logs.append(
        f"[registry] Saved model to '{MODEL_PATH}' and metadata to '{METADATA_PATH}'. "
        f"MLflow registration: {registry_report['registered_in_mlflow']}."
    )

    return {
        **state,
        "registry_report": registry_report,
        "errors": errors,
        "logs": logs,
    }
