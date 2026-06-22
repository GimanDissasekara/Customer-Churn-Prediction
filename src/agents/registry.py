"""
Model Registry Agent
----------------------
Responsibilities:
  - Persist the trained pipeline (preprocessing + model) to disk via joblib
  - Save model metadata (metrics, model type, training timestamp, feature columns)
  - Register the model in the MLflow Model Registry (if it passed the quality gate)
  - Save reference statistics of the training data for the Monitoring/Drift Agent
  - Compose a final pipeline-complete message
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
    return {
        col: {
            "mean": float(numeric_df[col].mean()),
            "std":  float(numeric_df[col].std()),
            "min":  float(numeric_df[col].min()),
            "max":  float(numeric_df[col].max()),
        }
        for col in numeric_df.columns
    }


def _compose_message(
    model_name: str,
    metrics: dict,
    model_path: str,
    metadata_path: str,
    registered: bool,
    n_ref_cols: int,
) -> dict:
    model_labels = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }
    lines = [
        "Pipeline complete. All artifacts have been saved.",
        "",
        "Artifacts written to disk:",
        f"  • {os.path.basename(model_path)} — fitted sklearn Pipeline "
        f"(preprocessing + {model_labels.get(model_name, model_name)}) via joblib",
        f"  • {os.path.basename(metadata_path)} — model name, training timestamp, "
        f"metrics, candidate comparison, feature column list",
        f"  • reference_stats.json — baseline statistics for {n_ref_cols} numeric "
        f"features (mean/std/min/max) for future drift detection",
        "",
        "MLflow Model Registry:",
    ]

    if registered:
        lines.append(
            f"  ✓ New version of '{MLFLOW_REGISTERED_MODEL_NAME}' registered successfully.\n"
            f"    This version can be promoted to Staging or Production via the MLflow UI."
        )
    else:
        lines.append(
            f"  ✗ Registration skipped — model did not pass the quality gate.\n"
            f"    Model is saved to disk for debugging but will not be served."
        )

    lines.append("")
    lines.append(
        "System is ready. Send a customer CSV to POST /predict to score new customers."
    )

    return {
        "sender": "Registry Agent",
        "receiver": None,
        "content": "\n".join(lines),
    }


def registry_agent(state: ChurnPipelineState) -> ChurnPipelineState:
    """LangGraph node: persist the trained model and register it with MLflow."""
    logs = state.get("logs", [])
    errors = state.get("errors", [])
    agent_messages = state.get("agent_messages", [])

    required = ["model", "evaluation_report", "model_name", "candidate_results"]
    missing = [k for k in required if k not in state]
    if missing:
        errors.append(f"[registry] Missing state keys: {missing} - evaluation must run first.")
        return {**state, "errors": errors, "logs": logs, "agent_messages": agent_messages}

    os.makedirs(MODELS_DIR, exist_ok=True)

    pipeline = state["model"]
    report   = state["evaluation_report"]

    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(pipeline, PIPELINE_PATH)

    metadata = {
        "model_name":         state["model_name"],
        "trained_at":         datetime.now(timezone.utc).isoformat(),
        "metrics":            report["metrics"],
        "candidate_results":  state["candidate_results"],
        "passed_quality_gate": report["passed"],
        "feature_columns":    list(state["X_test"].columns),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    n_ref_cols = 0
    if "feature_df" in state:
        ref_stats = _compute_reference_stats(state["feature_df"])
        n_ref_cols = len(ref_stats)
        with open(REFERENCE_STATS_PATH, "w") as f:
            json.dump(ref_stats, f, indent=2)

    registry_report: Dict[str, Any] = {
        "model_path":           MODEL_PATH,
        "metadata_path":        METADATA_PATH,
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

    msg = _compose_message(
        model_name=state["model_name"],
        metrics=report["metrics"],
        model_path=MODEL_PATH,
        metadata_path=METADATA_PATH,
        registered=registry_report["registered_in_mlflow"],
        n_ref_cols=n_ref_cols,
    )
    agent_messages.append(msg)

    return {
        **state,
        "registry_report": registry_report,
        "errors": errors,
        "logs": logs,
        "agent_messages": agent_messages,
    }
