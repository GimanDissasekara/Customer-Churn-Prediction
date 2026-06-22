"""
Data Cleaning Agent
--------------------
Responsibilities:
  - Coerce numeric columns (e.g. TotalCharges arrives as a string with blanks)
  - Impute missing values
  - Drop duplicate rows
  - Normalize inconsistent categorical formatting
  - Encode the target column (train mode only)
  - Compose a handoff message to the Feature Engineering Agent
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.config import TARGET_COLUMN
from src.state import ChurnPipelineState


def _coerce_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """TotalCharges is read as object dtype because some rows are blank strings."""
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def _compose_message(
    duplicates_dropped: int,
    numeric_fill: dict,
    n_rows: int,
    n_cols: int,
    has_target: bool,
    n_churn: int,
    n_total: int,
) -> dict:
    lines = []

    lines.append(f"Cleaning complete. Final shape: {n_total:,} rows × {n_cols} columns.")
    lines.append("")

    lines.append("Changes applied to raw_df:")

    if numeric_fill:
        for col, val in numeric_fill.items():
            lines.append(
                f"  • {col}: coerced to float64 → imputed missing values with median ({val:,.2f})"
            )
    else:
        lines.append("  • No numeric imputation needed — no null values found.")

    lines.append(
        f"  • Categorical normalization: mapped 'No internet service' / "
        f"'No phone service' → 'No' across all categorical columns."
    )

    if duplicates_dropped > 0:
        lines.append(f"  • Dropped {duplicates_dropped} duplicate rows.")
    else:
        lines.append("  • No duplicate rows found.")

    if has_target:
        pct_churn = (n_churn / n_total * 100) if n_total > 0 else 0
        n_no_churn = n_total - n_churn
        lines.append(
            f"  • Target encoding: Churn Yes→1, No→0\n"
            f"    → {n_churn:,} churners ({pct_churn:.1f}%) | "
            f"{n_no_churn:,} non-churners ({100 - pct_churn:.1f}%)"
        )
        if pct_churn < 30:
            lines.append(
                f"    ⚠ Class imbalance detected ({pct_churn:.1f}% positive). "
                f"Training Agent will use class_weight='balanced' to compensate."
            )

    lines.append("")
    lines.append("Handing clean_df to Feature Engineering Agent.")

    return {
        "sender": "Cleaning Agent",
        "receiver": "Feature Engineering Agent",
        "content": "\n".join(lines),
    }


def cleaning_agent(state: ChurnPipelineState) -> ChurnPipelineState:
    """LangGraph node: clean and standardize the raw dataframe."""
    logs = state.get("logs", [])
    errors = state.get("errors", [])
    agent_messages = state.get("agent_messages", [])

    if "raw_df" not in state:
        errors.append("[cleaning] No raw_df found in state - ingestion must run first.")
        return {**state, "errors": errors, "logs": logs, "agent_messages": agent_messages}

    df = state["raw_df"].copy()
    report: Dict[str, Any] = {}

    df = _coerce_total_charges(df)

    n_before = len(df)
    df = df.drop_duplicates()
    report["duplicates_dropped"] = n_before - len(df)

    numeric_fill = {}
    for col in ["TotalCharges", "MonthlyCharges", "tenure"]:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            numeric_fill[col] = float(median_val)
            df[col] = df[col].fillna(median_val)
    report["numeric_imputation"] = numeric_fill
    report["imputed_columns"] = list(numeric_fill.keys())

    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    for col in categorical_cols:
        if col == "customerID":
            continue
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(
            {"No internet service": "No", "No phone service": "No"}
        )

    n_churn = 0
    has_target = TARGET_COLUMN in df.columns
    if has_target:
        n_before_target = len(df)
        df = df.dropna(subset=[TARGET_COLUMN])
        report["rows_dropped_missing_target"] = n_before_target - len(df)

        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0}).astype("Int64")
        n_unmapped = int(df[TARGET_COLUMN].isnull().sum())
        if n_unmapped:
            report["unmapped_target_rows_dropped"] = n_unmapped
            df = df.dropna(subset=[TARGET_COLUMN])
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
        n_churn = int(df[TARGET_COLUMN].sum())

    report["n_rows_after"] = len(df)
    report["remaining_nulls"] = df.isnull().sum().to_dict()

    logs.append(
        f"[cleaning] Cleaned dataset -> {len(df)} rows. "
        f"Dropped {report['duplicates_dropped']} duplicates, "
        f"imputed columns: {list(numeric_fill.keys()) or 'none'}."
    )

    msg = _compose_message(
        duplicates_dropped=report["duplicates_dropped"],
        numeric_fill=numeric_fill,
        n_rows=len(df),
        n_cols=df.shape[1],
        has_target=has_target,
        n_churn=n_churn,
        n_total=len(df),
    )
    agent_messages.append(msg)

    return {
        **state,
        "clean_df": df,
        "cleaning_report": report,
        "errors": errors,
        "logs": logs,
        "agent_messages": agent_messages,
    }
