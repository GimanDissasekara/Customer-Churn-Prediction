"""
Evaluation Agent
------------------
Responsibilities:
  - Compute final evaluation metrics for the selected model on the held-out test set
  - Log the final metrics to MLflow under a dedicated "best_model" run
  - Decide whether the model passes the quality gate
"""

from typing import Any, Dict

import mlflow
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from src.state import ChurnPipelineState

# Minimum ROC-AUC required for the model to be considered "good enough" to register
QUALITY_GATE_ROC_AUC = 0.70


def evaluation_agent(state: ChurnPipelineState) -> ChurnPipelineState:
    """LangGraph node: evaluate the selected model and apply a quality gate."""
    logs = state.get("logs", [])
    errors = state.get("errors", [])

    required = ["model", "X_test", "y_test", "model_name"]
    missing = [k for k in required if k not in state]
    if missing:
        errors.append(f"[evaluation] Missing state keys: {missing} - training must run first.")
        return {**state, "errors": errors, "logs": logs}

    model = state["model"]
    X_test = state["X_test"]
    y_test = state["y_test"]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    cm = confusion_matrix(y_test, y_pred).tolist()

    passed = metrics["roc_auc"] >= QUALITY_GATE_ROC_AUC

    report: Dict[str, Any] = {
        "model_name": state["model_name"],
        "metrics": metrics,
        "confusion_matrix": cm,
        "quality_gate": QUALITY_GATE_ROC_AUC,
        "passed": passed,
    }

    # Log a final consolidated run for the chosen best model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"best_model__{state['model_name']}"):
        mlflow.log_params({"selected_model": state["model_name"], "passed_quality_gate": passed})
        mlflow.log_metrics(metrics)

    logs.append(
        f"[evaluation] {state['model_name']} -> "
        f"accuracy={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}, roc_auc={metrics['roc_auc']:.4f}. "
        f"Quality gate ({QUALITY_GATE_ROC_AUC}) {'PASSED' if passed else 'FAILED'}."
    )

    return {
        **state,
        "metrics": metrics,
        "evaluation_report": report,
        "errors": errors,
        "logs": logs,
    }
