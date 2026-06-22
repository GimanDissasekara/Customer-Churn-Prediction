"""
Prediction Agent
------------------
Responsibilities:
  - Load the persisted model pipeline (preprocessing + classifier)
  - Apply the same cleaning / feature engineering as training
    (handled by re-using the cleaning and feature_engineering agents upstream
    in the prediction graph)
  - Generate churn probability and risk level for each customer
  - Output a tidy results dataframe: customerID, churn_probability, risk_level
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


def prediction_agent(state: ChurnPipelineState) -> ChurnPipelineState:
    """LangGraph node: score customers using the trained pipeline."""
    logs = state.get("logs", [])
    errors = state.get("errors", [])

    if "feature_df" not in state:
        errors.append("[prediction] No feature_df in state - feature engineering must run first.")
        return {**state, "errors": errors, "logs": logs}

    df = state["feature_df"].copy()

    try:
        pipeline = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        errors.append(f"[prediction] No trained model found at '{MODEL_PATH}'. Run training first.")
        return {**state, "errors": errors, "logs": logs}

    drop_cols = [c for c in [ID_COLUMN, TARGET_COLUMN] if c in df.columns]
    X = df.drop(columns=drop_cols)

    probabilities = pipeline.predict_proba(X)[:, 1]

    results = pd.DataFrame()
    if ID_COLUMN in df.columns:
        results[ID_COLUMN] = df[ID_COLUMN].values
    else:
        results[ID_COLUMN] = range(len(df))

    results["churn_probability"] = probabilities.round(4)
    results["risk_level"] = [_risk_level(p) for p in probabilities]

    if TARGET_COLUMN in df.columns:
        results["actual_churn"] = df[TARGET_COLUMN].values

    report: Dict[str, Any] = {
        "n_customers": len(results),
        "risk_level_counts": results["risk_level"].value_counts().to_dict(),
        "avg_churn_probability": float(probabilities.mean()),
    }

    logs.append(
        f"[prediction] Scored {len(results)} customers. "
        f"Risk distribution: {report['risk_level_counts']}."
    )

    return {
        **state,
        "predictions_df": results,
        "prediction_report": report,
        "errors": errors,
        "logs": logs,
    }
