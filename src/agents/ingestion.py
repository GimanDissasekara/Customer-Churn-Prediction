"""
Data Ingestion Agent
---------------------
Responsibilities:
  - Load the CSV file
  - Validate that required columns are present
  - Validate / coerce basic data types
  - Report basic dataset statistics
  - Compose a handoff message to the Cleaning Agent
"""

import os
from typing import Any, Dict

import pandas as pd

from src.config import REQUIRED_COLUMNS, TARGET_COLUMN
from src.state import ChurnPipelineState


def _compose_message(
    input_path: str,
    n_rows: int,
    n_cols: int,
    missing_cols: list,
    null_counts: dict,
    n_dupes: int,
    has_target: bool,
    mode: str,
) -> dict:
    non_null_cols = {k: v for k, v in null_counts.items() if v > 0}
    null_summary = (
        ", ".join(f"{k}={v}" for k, v in non_null_cols.items())
        if non_null_cols
        else "none"
    )

    lines = [
        f"Loaded {n_rows:,} customer records from '{os.path.basename(input_path)}' "
        f"({n_cols} columns).",
        "",
    ]

    if missing_cols:
        lines.append(f"⚠ Missing required columns: {missing_cols}.")
        lines.append("  Pipeline will be halted — please check your dataset schema.")
    else:
        lines.append("✓ Schema validation passed — all required columns are present.")

    lines.append(f"  Null cells detected: {null_summary}")
    lines.append(f"  Duplicate rows: {n_dupes}")

    if "TotalCharges" in null_counts and null_counts.get("TotalCharges", 0) == 0:
        # dtype object means blank strings exist but pd reads them as non-null strings
        lines.append(
            "⚠ TotalCharges appears as object dtype — blank strings may be present "
            "for new customers with zero tenure. Will need numeric coercion."
        )

    if mode == "train":
        lines.append(
            f"  Target column '{TARGET_COLUMN}' {'present ✓' if has_target else 'MISSING ✗'}."
        )
    else:
        lines.append("  Running in predict mode — target column is not required.")

    lines.append("")
    lines.append("Handing raw_df to Cleaning Agent.")

    return {
        "sender": "Ingestion Agent",
        "receiver": "Cleaning Agent",
        "content": "\n".join(lines),
    }


def ingestion_agent(state: ChurnPipelineState) -> ChurnPipelineState:
    """LangGraph node: load and validate the input CSV."""
    logs = state.get("logs", [])
    errors = state.get("errors", [])
    agent_messages = state.get("agent_messages", [])

    input_path = state["input_path"]
    mode = state.get("mode", "train")

    try:
        df = pd.read_csv(input_path)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"[ingestion] Failed to read CSV '{input_path}': {exc}")
        return {**state, "errors": errors, "logs": logs, "agent_messages": agent_messages}

    report: Dict[str, Any] = {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "columns": list(df.columns),
    }

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

    n_dupes = int(df.duplicated().sum())
    null_counts = df.isnull().sum().to_dict()
    report["duplicate_rows"] = n_dupes
    report["null_counts"] = null_counts

    logs.append(
        f"[ingestion] Loaded '{input_path}' with {report['n_rows']} rows, "
        f"{report['n_cols']} columns. Missing columns: {missing_cols or 'none'}."
    )

    msg = _compose_message(
        input_path=input_path,
        n_rows=len(df),
        n_cols=df.shape[1],
        missing_cols=missing_cols,
        null_counts=null_counts,
        n_dupes=n_dupes,
        has_target=report["has_target"],
        mode=mode,
    )
    agent_messages.append(msg)

    return {
        **state,
        "raw_df": df,
        "ingestion_report": report,
        "errors": errors,
        "logs": logs,
        "agent_messages": agent_messages,
    }
