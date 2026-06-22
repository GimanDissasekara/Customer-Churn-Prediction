"""
Central configuration for the Customer Churn MAS pipeline.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

MODELS_DIR = os.path.join(BASE_DIR, "models")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

DEFAULT_TRAIN_FILE = os.path.join(RAW_DATA_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

MODEL_PATH = os.path.join(MODELS_DIR, "churn_model.pkl")
PIPELINE_PATH = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
REFERENCE_STATS_PATH = os.path.join(MODELS_DIR, "reference_stats.json")

# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------
ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

# Columns expected in the raw Telco churn dataset
REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Engineered feature names added by the Feature Engineering Agent
ENGINEERED_NUMERIC_FEATURES = [
    "usage_intensity",
    "avg_monthly_spend",
    "num_streaming_services",
    "num_security_services",
]

ENGINEERED_CATEGORICAL_FEATURES = [
    "tenure_group",
    "payment_behavior",
]

# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Risk level thresholds applied to predicted churn probability
RISK_THRESHOLDS = {
    "High": 0.7,
    "Medium": 0.4,
    "Low": 0.0,
}

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_EXPERIMENT_NAME = "customer_churn_mas"
# MLflow 3.x deprecated the plain filesystem store; use a local SQLite DB so
# the model registry features (registered models, versions, stages) work.
MLFLOW_DB_PATH = os.path.join(MLRUNS_DIR, "mlflow.db")
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.replace(os.sep, '/')}"
MLFLOW_REGISTERED_MODEL_NAME = "customer_churn_classifier"

# skops audits serialized sklearn models and rejects unknown types by default.
# These are the types present in our pipeline (numpy internals + XGBoost) that
# we explicitly trust because they come from our own training code.
SKOPS_TRUSTED_TYPES: list[str] = [
    "numpy.dtype",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
]
