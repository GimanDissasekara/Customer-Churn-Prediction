"""
Data Cleaning Agent
--------------------
Responsibilities:
  - Coerce numeric columns (e.g. TotalCharges arrives as a string with blanks)
  - Impute missing values
  - Drop duplicate rows
  - Normalize inconsistent categorical formatting
  - Encode the target column (train mode only)
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


def cleaning_agent(state: ChurnPipelineState) -> ChurnPipelineState:
    """LangGraph node: clean and standardize the raw dataframe."""
    logs = state.get("logs", [])
    errors = state.get("errors", [])

    if "raw_df" not in state:
        errors.append("[cleaning] No raw_df found in state - ingestion must run first.")
        return {**state, "errors": errors, "logs": logs}

    df = state["raw_df"].copy()
    report: Dict[str, Any] = {}

    # --- Fix dtypes ----------------------------------------------------
    df = _coerce_total_charges(df)

    # --- Drop duplicates -------------------------------------------------
    n_before = len(df)
    df = df.drop_duplicates()
    report["duplicates_dropped"] = n_before - len(df)

    # --- Impute missing numeric values -----------------------------------
    numeric_fill = {}
    for col in ["TotalCharges", "MonthlyCharges", "tenure"]:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            numeric_fill[col] = float(median_val)
            df[col] = df[col].fillna(median_val)
    report["numeric_imputation"] = numeric_fill

    # --- Normalize categorical text formatting ----------------------------
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    for col in categorical_cols:
        if col == "customerID":
            continue
        df[col] = df[col].astype(str).str.strip()
        # Standardize "no internet/phone service" variants -> "No"
        df[col] = df[col].replace(
            {"No internet service": "No", "No phone service": "No"}
        )

    # --- Drop rows with missing target (train mode) ------------------------
    if TARGET_COLUMN in df.columns:
        n_before_target = len(df)
        df = df.dropna(subset=[TARGET_COLUMN])
        report["rows_dropped_missing_target"] = n_before_target - len(df)

        # Encode target: Yes -> 1, No -> 0
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0}).astype("Int64")
        n_unmapped = int(df[TARGET_COLUMN].isnull().sum())
        if n_unmapped:
            report["unmapped_target_rows_dropped"] = n_unmapped
            df = df.dropna(subset=[TARGET_COLUMN])
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    report["n_rows_after"] = len(df)
    report["remaining_nulls"] = df.isnull().sum().to_dict()

    logs.append(
        f"[cleaning] Cleaned dataset -> {len(df)} rows. "
        f"Dropped {report['duplicates_dropped']} duplicates, "
        f"imputed columns: {list(numeric_fill.keys()) or 'none'}."
    )

    return {
        **state,
        "clean_df": df,
        "cleaning_report": report,
        "errors": errors,
        "logs": logs,
    }
