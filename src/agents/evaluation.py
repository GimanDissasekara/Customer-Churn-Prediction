"""
Evaluation Agent
------------------
Responsibilities:
  - Compute final evaluation metrics for the selected model on the held-out test set
  - Log the final metrics to MLflow under a dedicated "best_model" run
  - Decide whether the model passes the quality gate
  - Compose a handoff message to the Registry Agent
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

QUALITY_GATE_ROC_AUC = 0.70


def _compose_message(
    model_name: str,
    metrics: dict,
    cm: list,
    passed: bool,
    quality_gate: float,
) -> dict:
    # confusion matrix: [[TN, FP], [FN, TP]]
    tn = cm[0][0] if len(cm) > 1 else 0
    fp = cm[0][1] if len(cm) > 1 else 0
    fn = cm[1][0] if len(cm) > 1 else 0
    tp = cm[1][1] if len(cm) > 1 else 0
    n_test = tn + fp + fn + tp

    lines = [
        f"Evaluation of '{model_name}' on {n_test:,} held-out samples:",
        "",
        "Performance metrics:",
        f"  • ROC-AUC  : {metrics['roc_auc']:.4f}  "
        f"{'✓ PASSES' if passed else '✗ FAILS'} quality gate (≥ {quality_gate})",
        f"  • F1 Score : {metrics['f1']:.4f}",
        f"  • Precision: {metrics['precision']:.4f}  "
        f"(of all flagged churners, {metrics['precision']*100:.1f}% actually churned)",
        f"  • Recall   : {metrics['recall']:.4f}  "
        f"(caught {metrics['recall']*100:.1f}% of all actual churners)",
        f"  • Accuracy : {metrics['accuracy']:.4f}",
        "",
        "Confusion matrix:",
        f"  True Positives  (caught churners)  : {tp:,}",
        f"  True Negatives  (correct retains)  : {tn:,}",
        f"  False Positives (false alarms)     : {fp:,}",
        f"  False Negatives (missed churners)  : {fn:,}",
    ]

    if fn > 0:
        lines.append(
            f"  ⚠ {fn:,} actual churners were missed (false negatives). "
            f"These customers will leave without a retention intervention."
        )
    if fp > 0:
        lines.append(
            f"  ℹ {fp:,} non-churners were flagged (false positives). "
            f"Retention spend on these customers is wasted but unavoidable at this threshold."
        )

    lines.append("")
    if passed:
        lines.append(
            f"✓ Quality gate passed. Model is approved for registration.\n"
            f"  Handing evaluation_report to Registry Agent."
        )
    else:
        lines.append(
            f"✗ Quality gate FAILED (ROC-AUC {metrics['roc_auc']:.4f} < {quality_gate}).\n"
            f"  Model will be saved to disk for reference but will NOT be registered in MLflow.\n"
            f"  Handing evaluation_report to Registry Agent."
        )

    return {
        "sender": "Evaluation Agent",
        "receiver": "Registry Agent",
        "content": "\n".join(lines),
    }


def evaluation_agent(state: ChurnPipelineState) -> ChurnPipelineState:
    """LangGraph node: evaluate the selected model and apply a quality gate."""
    logs = state.get("logs", [])
    errors = state.get("errors", [])
    agent_messages = state.get("agent_messages", [])

    required = ["model", "X_test", "y_test", "model_name"]
    missing = [k for k in required if k not in state]
    if missing:
        errors.append(f"[evaluation] Missing state keys: {missing} - training must run first.")
        return {**state, "errors": errors, "logs": logs, "agent_messages": agent_messages}

    model  = state["model"]
    X_test = state["X_test"]
    y_test = state["y_test"]

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_test, y_proba)),
    }

    cm = confusion_matrix(y_test, y_pred).tolist()
    passed = metrics["roc_auc"] >= QUALITY_GATE_ROC_AUC

    report: Dict[str, Any] = {
        "model_name":     state["model_name"],
        "metrics":        metrics,
        "confusion_matrix": cm,
        "quality_gate":   QUALITY_GATE_ROC_AUC,
        "passed":         passed,
    }

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

    msg = _compose_message(
        model_name=state["model_name"],
        metrics=metrics,
        cm=cm,
        passed=passed,
        quality_gate=QUALITY_GATE_ROC_AUC,
    )
    agent_messages.append(msg)

    return {
        **state,
        "metrics": metrics,
        "evaluation_report": report,
        "errors": errors,
        "logs": logs,
        "agent_messages": agent_messages,
    }
