"""
Model Training Agent
----------------------
Responsibilities:
  - Build a preprocessing pipeline (impute / scale numeric, one-hot encode categorical)
  - Split data into train/test sets
  - Train multiple candidate models: Logistic Regression, Random Forest, XGBoost
  - Score each candidate on the held-out test set (ROC-AUC primary metric)
  - Select the best-performing model
  - Log every run (params + metrics + model artifact) to MLflow
  - Compose a handoff message to the Evaluation Agent
"""

from typing import Any, Dict

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import (
    CATEGORICAL_COLUMNS,
    ENGINEERED_CATEGORICAL_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
    ID_COLUMN,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    NUMERIC_COLUMNS,
    RANDOM_STATE,
    SKOPS_TRUSTED_TYPES,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.state import ChurnPipelineState


def _build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric_features = [c for c in NUMERIC_COLUMNS + ENGINEERED_NUMERIC_FEATURES if c in df.columns]
    categorical_features = [
        c for c in CATEGORICAL_COLUMNS + ENGINEERED_CATEGORICAL_FEATURES if c in df.columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor


def _candidate_models() -> Dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }


MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def _compose_message(
    candidate_results: Dict[str, Dict[str, float]],
    best_name: str,
    n_train: int,
    n_test: int,
) -> dict:
    best = candidate_results[best_name]
    lines = [
        f"Training complete. Evaluated {len(candidate_results)} candidate models on "
        f"{n_train:,} training samples (held out {n_test:,} for evaluation).",
        "",
        "Model comparison (held-out test set):",
    ]

    # Sort by roc_auc descending
    ranked = sorted(candidate_results.items(), key=lambda x: x[1]["roc_auc"], reverse=True)
    for i, (name, m) in enumerate(ranked):
        marker = "★ BEST" if name == best_name else f"  #{i+1}"
        lines.append(
            f"  {marker}  {MODEL_LABELS.get(name, name)}\n"
            f"          ROC-AUC={m['roc_auc']:.4f} | F1={m['f1']:.4f} | "
            f"Precision={m['precision']:.4f} | Recall={m['recall']:.4f}"
        )

    lines.append("")
    lines.append(
        f"Selected '{MODEL_LABELS.get(best_name, best_name)}' as the best model "
        f"(highest ROC-AUC = {best['roc_auc']:.4f})."
    )

    # Note on class imbalance handling
    lines.append(
        "  Note: Logistic Regression and Random Forest were trained with "
        "class_weight='balanced' to handle the ~26% churn / ~74% non-churn imbalance."
    )

    lines.append("")
    lines.append(
        f"Passing fitted pipeline + held-out test set ({n_test:,} samples) "
        f"to Evaluation Agent for final metric computation."
    )

    return {
        "sender": "Training Agent",
        "receiver": "Evaluation Agent",
        "content": "\n".join(lines),
    }


def training_agent(state: ChurnPipelineState) -> ChurnPipelineState:
    """LangGraph node: train and select the best churn classifier."""
    logs = state.get("logs", [])
    errors = state.get("errors", [])
    agent_messages = state.get("agent_messages", [])

    if "feature_df" not in state:
        errors.append("[training] No feature_df in state - feature engineering must run first.")
        return {**state, "errors": errors, "logs": logs, "agent_messages": agent_messages}

    df = state["feature_df"].copy()

    if TARGET_COLUMN not in df.columns:
        errors.append("[training] Target column missing - cannot train.")
        return {**state, "errors": errors, "logs": logs, "agent_messages": agent_messages}

    drop_cols = [c for c in [ID_COLUMN, TARGET_COLUMN] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = _build_preprocessor(X_train)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    candidate_results: Dict[str, Dict[str, float]] = {}
    fitted_pipelines: Dict[str, Pipeline] = {}

    for name, model in _candidate_models().items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

        with mlflow.start_run(run_name=name, nested=False):
            pipeline.fit(X_train, y_train)

            y_proba = pipeline.predict_proba(X_test)[:, 1]
            y_pred  = pipeline.predict(X_test)

            metrics = {
                "accuracy":  accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall":    recall_score(y_test, y_pred, zero_division=0),
                "f1":        f1_score(y_test, y_pred, zero_division=0),
                "roc_auc":   roc_auc_score(y_test, y_proba),
            }

            mlflow.log_params({"model_type": name})
            try:
                mlflow.log_params(model.get_params())
            except Exception:  # noqa: BLE001
                pass
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )

        candidate_results[name] = metrics
        fitted_pipelines[name] = pipeline

        logs.append(f"[training] {name}: ROC-AUC={metrics['roc_auc']:.4f}, F1={metrics['f1']:.4f}")

    best_name = max(candidate_results, key=lambda k: candidate_results[k]["roc_auc"])
    best_pipeline = fitted_pipelines[best_name]

    logs.append(f"[training] Best model selected: {best_name} (ROC-AUC={candidate_results[best_name]['roc_auc']:.4f})")

    msg = _compose_message(
        candidate_results=candidate_results,
        best_name=best_name,
        n_train=len(X_train),
        n_test=len(X_test),
    )
    agent_messages.append(msg)

    return {
        **state,
        "model": best_pipeline,
        "pipeline": best_pipeline,
        "model_name": best_name,
        "candidate_results": candidate_results,
        "X_test": X_test,
        "y_test": y_test,
        "errors": errors,
        "logs": logs,
        "agent_messages": agent_messages,
    }
