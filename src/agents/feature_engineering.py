"""
Feature Engineering Agent
--------------------------
Responsibilities:
  - Build derived features that capture customer behavior
  - Compose a handoff message to the Training or Prediction Agent
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.state import ChurnPipelineState

STREAMING_COLS = ["StreamingTV", "StreamingMovies"]
SECURITY_COLS  = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]


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


def _compose_message(df: pd.DataFrame, new_features: list, mode: str) -> dict:
    next_agent = "Training Agent" if mode == "train" else "Prediction Agent"
    lines = [
        f"Feature engineering complete. Created {len(new_features)} new features.",
        f"Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns.",
        "",
        "New features and key statistics:",
    ]

    if "tenure_group" in df.columns:
        vc = df["tenure_group"].value_counts()
        n = len(df)
        lines.append(
            f"  • tenure_group distribution:\n"
            + "".join(
                f"      {k}: {vc.get(k, 0):,} ({vc.get(k, 0)/n*100:.1f}%)\n"
                for k in ["New", "Mid", "Loyal", "Veteran"]
            ).rstrip()
        )
        new_pct = vc.get("New", 0) / n * 100
        if new_pct > 15:
            lines.append(
                f"    ⚠ {new_pct:.0f}% of customers are 'New' (≤12 months). "
                f"This segment has the highest churn rate — "
                f"the model should weight this group carefully."
            )

    if "usage_intensity" in df.columns:
        lines.append(
            f"  • usage_intensity (MonthlyCharges / tenure): "
            f"mean={df['usage_intensity'].mean():.2f}, "
            f"std={df['usage_intensity'].std():.2f}"
        )

    if "avg_monthly_spend" in df.columns:
        lines.append(
            f"  • avg_monthly_spend (TotalCharges / tenure): "
            f"mean={df['avg_monthly_spend'].mean():.2f}, "
            f"std={df['avg_monthly_spend'].std():.2f}"
        )

    if "num_streaming_services" in df.columns:
        vc = df["num_streaming_services"].value_counts().sort_index()
        lines.append(
            "  • num_streaming_services: "
            + ", ".join(f"{k} services={v:,}" for k, v in vc.items())
        )

    if "num_security_services" in df.columns:
        vc = df["num_security_services"].value_counts().sort_index()
        lines.append(
            "  • num_security_services: "
            + ", ".join(f"{k}={v:,}" for k, v in vc.items())
        )

    if "payment_behavior" in df.columns:
        vc = df["payment_behavior"].value_counts()
        n = len(df)
        manual_pct = vc.get("Manual", 0) / n * 100
        lines.append(
            f"  • payment_behavior: "
            f"Manual={vc.get('Manual', 0):,} ({manual_pct:.1f}%), "
            f"Automatic={vc.get('Automatic', 0):,} ({100-manual_pct:.1f}%)"
        )
        if manual_pct > 40:
            lines.append(
                f"    ⚠ {manual_pct:.0f}% use manual payment — "
                f"this group churns significantly more than automatic-payment customers."
            )

    lines.append("")
    lines.append(f"Handing feature_df to {next_agent}.")

    return {
        "sender": "Feature Engineering Agent",
        "receiver": next_agent,
        "content": "\n".join(lines),
    }


def feature_engineering_agent(state: ChurnPipelineState) -> ChurnPipelineState:
    """LangGraph node: derive new predictive features."""
    logs = state.get("logs", [])
    errors = state.get("errors", [])
    agent_messages = state.get("agent_messages", [])

    if "clean_df" not in state:
        errors.append("[feature_engineering] No clean_df in state - cleaning must run first.")
        return {**state, "errors": errors, "logs": logs, "agent_messages": agent_messages}

    df = state["clean_df"].copy()
    mode = state.get("mode", "train")
    report: Dict[str, Any] = {"new_features": []}

    if "tenure" in df.columns:
        df["tenure_group"] = df["tenure"].apply(_tenure_group)
        report["new_features"].append("tenure_group")

    if "MonthlyCharges" in df.columns and "tenure" in df.columns:
        df["usage_intensity"] = df["MonthlyCharges"] / df["tenure"].replace(0, 1)
        report["new_features"].append("usage_intensity")

    if "TotalCharges" in df.columns and "tenure" in df.columns:
        df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
        df["avg_monthly_spend"] = df["avg_monthly_spend"].fillna(df["MonthlyCharges"])
        report["new_features"].append("avg_monthly_spend")

    df["num_streaming_services"] = _count_yes(df, STREAMING_COLS)
    df["num_security_services"]  = _count_yes(df, SECURITY_COLS)
    report["new_features"].extend(["num_streaming_services", "num_security_services"])

    if "PaymentMethod" in df.columns:
        df["payment_behavior"] = df["PaymentMethod"].apply(_payment_behavior)
        report["new_features"].append("payment_behavior")

    df = df.replace([np.inf, -np.inf], 0)

    report["n_rows"] = len(df)
    report["n_cols"] = df.shape[1]

    logs.append(
        f"[feature_engineering] Added features: {report['new_features']}. "
        f"Resulting shape: {df.shape}."
    )

    msg = _compose_message(df, report["new_features"], mode)
    agent_messages.append(msg)

    return {
        **state,
        "feature_df": df,
        "feature_report": report,
        "errors": errors,
        "logs": logs,
        "agent_messages": agent_messages,
    }
