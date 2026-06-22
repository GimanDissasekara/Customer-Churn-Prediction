"""
Feature Engineering Agent
--------------------------
Responsibilities:
  - Build derived features that capture customer behavior:
      * tenure_group        - bucketed tenure (New / Mid / Loyal / Veteran)
      * usage_intensity     - MonthlyCharges relative to tenure
      * avg_monthly_spend   - TotalCharges / tenure (lifetime average spend)
      * num_streaming_services / num_security_services - service counts
      * payment_behavior    - automatic vs manual payment grouping
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.state import ChurnPipelineState

STREAMING_COLS = ["StreamingTV", "StreamingMovies"]
SECURITY_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]


def _tenure_group(tenure: float) -> str:
    if tenure <= 12:
        return "New"
    if tenure <= 24:
        return "Mid"
    if tenure <= 48:
        return "Loyal"
    return "Veteran"


def _payment_behavior(payment_method: str) -> str:
    if isinstance(payment_method, str) and "automatic" in payment_method.lower():
        return "Automatic"
    return "Manual"


def _count_yes(df: pd.DataFrame, cols: list) -> pd.Series:
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(0, index=df.index)
    return (df[present] == "Yes").sum(axis=1)


def feature_engineering_agent(state: ChurnPipelineState) -> ChurnPipelineState:
    """LangGraph node: derive new predictive features."""
    logs = state.get("logs", [])
    errors = state.get("errors", [])

    if "clean_df" not in state:
        errors.append("[feature_engineering] No clean_df in state - cleaning must run first.")
        return {**state, "errors": errors, "logs": logs}

    df = state["clean_df"].copy()
    report: Dict[str, Any] = {"new_features": []}

    # --- Tenure group --------------------------------------------------
    if "tenure" in df.columns:
        df["tenure_group"] = df["tenure"].apply(_tenure_group)
        report["new_features"].append("tenure_group")

    # --- Usage intensity: spend per month of tenure -----------------------
    if "MonthlyCharges" in df.columns and "tenure" in df.columns:
        df["usage_intensity"] = df["MonthlyCharges"] / df["tenure"].replace(0, 1)
        report["new_features"].append("usage_intensity")

    # --- Average monthly spend over lifetime -------------------------------
    if "TotalCharges" in df.columns and "tenure" in df.columns:
        df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
        df["avg_monthly_spend"] = df["avg_monthly_spend"].fillna(df["MonthlyCharges"])
        report["new_features"].append("avg_monthly_spend")

    # --- Service counts -----------------------------------------------------
    df["num_streaming_services"] = _count_yes(df, STREAMING_COLS)
    df["num_security_services"] = _count_yes(df, SECURITY_COLS)
    report["new_features"].extend(["num_streaming_services", "num_security_services"])

    # --- Payment behavior -----------------------------------------------------
    if "PaymentMethod" in df.columns:
        df["payment_behavior"] = df["PaymentMethod"].apply(_payment_behavior)
        report["new_features"].append("payment_behavior")

    # Replace any inf/-inf produced by divisions with 0
    df = df.replace([np.inf, -np.inf], 0)

    report["n_rows"] = len(df)
    report["n_cols"] = df.shape[1]

    logs.append(
        f"[feature_engineering] Added features: {report['new_features']}. "
        f"Resulting shape: {df.shape}."
    )

    return {
        **state,
        "feature_df": df,
        "feature_report": report,
        "errors": errors,
        "logs": logs,
    }
