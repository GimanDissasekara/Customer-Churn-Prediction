"""
Data Ingestion Agent
---------------------
Responsibilities:
  - Load the CSV file
  - Validate that required columns are present
  - Validate / coerce basic data types
  - Report basic dataset statistics
"""

from typing import Any, Dict

import pandas as pd

from src.config import REQUIRED_COLUMNS, TARGET_COLUMN
from src.state import ChurnPipelineState


def ingestion_agent(state: ChurnPipelineState) -> ChurnPipelineState:
    """LangGraph node: load and validate the input CSV."""
    logs = state.get("logs", [])
    errors = state.get("errors", [])

    input_path = state["input_path"]
    mode = state.get("mode", "train")

    try:
        df = pd.read_csv(input_path)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"[ingestion] Failed to read CSV '{input_path}': {exc}")
        return {**state, "errors": errors, "logs": logs}

    report: Dict[str, Any] = {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "columns": list(df.columns),
    }

    # --- Schema validation -------------------------------------------------
    required = list(REQUIRED_COLUMNS)
    if mode == "train":
        required = required + [TARGET_COLUMN]

    missing_cols = [c for c in required if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in required]

    report["missing_columns"] = missing_cols
    report["extra_columns"] = extra_cols
    report["has_target"] = TARGET_COLUMN in df.columns

    if missing_cols:
        errors.append(
            f"[ingestion] Dataset is missing required columns: {missing_cols}"
        )

    # --- Basic data quality checks -----------------------------------------
    report["duplicate_rows"] = int(df.duplicated().count() - df.drop_duplicates().shape[0])
    report["null_counts"] = df.isnull().sum().to_dict()

    logs.append(
        f"[ingestion] Loaded '{input_path}' with {report['n_rows']} rows, "
        f"{report['n_cols']} columns. Missing columns: {missing_cols or 'none'}."
    )

    return {
        **state,
        "raw_df": df,
        "ingestion_report": report,
        "errors": errors,
        "logs": logs,
    }
