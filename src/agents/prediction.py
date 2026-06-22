"""
Prediction Agent
------------------
Responsibilities:
  - Load the persisted model pipeline (preprocessing + classifier)
  - Generate churn probability and risk level for each customer
  - Output a tidy results dataframe: customerID, churn_probability, risk_level
  - Compose a final pipeline-complete message
"""

from typing import Any, Dict

import joblib
import pandas as pd

from src.config import ID_COLUMN, MODEL_PATH, RISK_THRESHOLDS, TARGET_COLUMN
from src.state import ChurnPipelineState


def _risk_level(prob: float) -> str:
    if prob >= RISK_THRESHOLDS["High"]:
        return "High"
    if prob >= RISK_THRESHOLDS["Medium"]:
        return "Medium"
    return "Low"


def _compose_message(
    n_customers: int,
    risk_counts: dict,
    avg_prob: float,
    top_prob: float,
) -> dict:
    high   = risk_counts.get("High", 0)
    medium = risk_counts.get("Medium", 0)
    low    = risk_counts.get("Low", 0)

    high_pct   = high   / n_customers * 100 if n_customers else 0
    medium_pct = medium / n_customers * 100 if n_customers else 0
    low_pct    = low    / n_customers * 100 if n_customers else 0

    lines = [
        f"Scoring complete. Scored {n_customers:,} customers using the saved churn model.",
        "",
        "Risk distribution:",
        f"  • High risk   (≥70% churn prob): {high:,} customers  ({high_pct:.1f}%)",
        f"  • Medium risk (40–69%):           {medium:,} customers  ({medium_pct:.1f}%)",
        f"  • Low risk    (<40%):             {low:,} customers  ({low_pct:.1f}%)",
        "",
        f"Average churn probability across all customers: {avg_prob*100:.1f}%",
        f"Highest individual churn probability: {top_prob*100:.1f}%",
        "",
    ]

    if high > 0:
        lines.append(
            f"⚠ {high:,} customers are at HIGH risk. "
            f"Recommend immediate retention outreach for this group."
        )
    if medium > 0:
        lines.append(
            f"ℹ {medium:,} customers are at MEDIUM risk. "
            f"Monitor closely — targeted promotions or check-in calls may help."
        )

    lines.append("")
    lines.append(
        "Results are available in the predictions table. "
        "Use 'Export CSV' to download for CRM upload."
    )

    return {
        "sender": "Prediction Agent",
        "receiver": None,
        "content": "\n".join(lines),
    }


def prediction_agent(state: ChurnPipelineState) -> ChurnPipelineState:
    """LangGraph node: score customers using the trained pipeline."""
    logs = state.get("logs", [])
    errors = state.get("errors", [])
    agent_messages = state.get("agent_messages", [])

    if "feature_df" not in state:
        errors.append("[prediction] No feature_df in state - feature engineering must run first.")
        return {**state, "errors": errors, "logs": logs, "agent_messages": agent_messages}

    df = state["feature_df"].copy()

    try:
        pipeline = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        errors.append(f"[prediction] No trained model found at '{MODEL_PATH}'. Run training first.")
        return {**state, "errors": errors, "logs": logs, "agent_messages": agent_messages}

    drop_cols = [c for c in [ID_COLUMN, TARGET_COLUMN] if c in df.columns]
    X = df.drop(columns=drop_cols)

    probabilities = pipeline.predict_proba(X)[:, 1]

    results = pd.DataFrame()
    if ID_COLUMN in df.columns:
        results[ID_COLUMN] = df[ID_COLUMN].values
    else:
        results[ID_COLUMN] = range(len(df))

    results["churn_probability"] = probabilities.round(4)
    results["risk_level"]        = [_risk_level(p) for p in probabilities]

    if TARGET_COLUMN in df.columns:
        results["actual_churn"] = df[TARGET_COLUMN].values

    report: Dict[str, Any] = {
        "n_customers":          len(results),
        "risk_level_counts":    results["risk_level"].value_counts().to_dict(),
        "avg_churn_probability": float(probabilities.mean()),
    }

    logs.append(
        f"[prediction] Scored {len(results)} customers. "
        f"Risk distribution: {report['risk_level_counts']}."
    )

    msg = _compose_message(
        n_customers=len(results),
        risk_counts=report["risk_level_counts"],
        avg_prob=report["avg_churn_probability"],
        top_prob=float(probabilities.max()),
    )
    agent_messages.append(msg)

    return {
        **state,
        "predictions_df":    results,
        "prediction_report": report,
        "errors":            errors,
        "logs":              logs,
        "agent_messages":    agent_messages,
    }
