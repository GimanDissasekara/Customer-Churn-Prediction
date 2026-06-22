"""
Shared state schema passed between agents in the LangGraph workflows.
"""

from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd


class ChurnPipelineState(TypedDict, total=False):
    """
    Shared mutable state object that flows through every node (agent)
    of the LangGraph training and prediction graphs.

    Not every field is populated at every stage - agents read what they
    need and write what they produce.
    """

    # --- Input ---
    input_path: str                     # path to the source CSV file
    mode: str                            # "train" or "predict"

    # --- Ingestion Agent outputs ---
    raw_df: pd.DataFrame
    ingestion_report: Dict[str, Any]

    # --- Cleaning Agent outputs ---
    clean_df: pd.DataFrame
    cleaning_report: Dict[str, Any]

    # --- Feature Engineering Agent outputs ---
    feature_df: pd.DataFrame
    feature_report: Dict[str, Any]

    # --- Model Training Agent outputs ---
    model: Any
    pipeline: Any                       # fitted sklearn ColumnTransformer
    model_name: str
    candidate_results: Dict[str, Dict[str, float]]
    X_test: pd.DataFrame
    y_test: pd.Series

    # --- Evaluation Agent outputs ---
    metrics: Dict[str, float]
    evaluation_report: Dict[str, Any]

    # --- Model Registry Agent outputs ---
    registry_report: Dict[str, Any]

    # --- Prediction Agent outputs ---
    predictions_df: pd.DataFrame
    prediction_report: Dict[str, Any]

    # --- Monitoring Agent outputs ---
    drift_report: Dict[str, Any]

    # --- Inter-agent conversation messages ---
    # Each dict: {"sender": str, "receiver": str, "content": str}
    # Agents append one message per handoff so the UI can show the conversation.
    agent_messages: List[Dict[str, Any]]

    # --- Bookkeeping ---
    errors: List[str]
    logs: List[str]
