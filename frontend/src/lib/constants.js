export const MODEL_LABELS = {
  logistic_regression: "Logistic Regression",
  random_forest: "Random Forest",
  xgboost: "XGBoost",
};

export const RISK_COLORS = {
  High: "#f87171",
  Medium: "#fbbf24",
  Low: "#34d399",
};

export const CHART_TOOLTIP_STYLE = {
  contentStyle: { background: "#0f172a", border: "1px solid #334155" },
};

// Agent definitions for the training pipeline (in execution order)
// `stateKey` = the key written to ChurnPipelineState by this agent
// `stepMs`   = estimated runtime used for the loading animation only
export const TRAINING_AGENTS = [
  {
    key: "ingestion",
    label: "Ingestion Agent",
    description: "Load CSV → validate schema → report null counts",
    stateKey: "raw_df",
    stepMs: 1500,
  },
  {
    key: "cleaning",
    label: "Cleaning Agent",
    description: "Fix dtypes · impute nulls · normalize categoricals · encode target",
    stateKey: "clean_df",
    stepMs: 1500,
  },
  {
    key: "feature_engineering",
    label: "Feature Engineering Agent",
    description: "Build 6 derived features: tenure_group, usage_intensity, avg_monthly_spend, service counts, payment_behavior",
    stateKey: "feature_df",
    stepMs: 2000,
  },
  {
    key: "training",
    label: "Training Agent",
    description: "Train LR + RF + XGBoost · log each run to MLflow · pick best by ROC-AUC",
    stateKey: "model",
    stepMs: 35000,
  },
  {
    key: "evaluation",
    label: "Evaluation Agent",
    description: "Compute final metrics on held-out test set · apply quality gate (ROC-AUC ≥ 0.70)",
    stateKey: "evaluation_report",
    stepMs: 4000,
  },
  {
    key: "registry",
    label: "Registry Agent",
    description: "Save pipeline to disk (joblib) · write metadata JSON · register in MLflow",
    stateKey: "registry_report",
    stepMs: 5000,
  },
];

// Agent definitions for the prediction pipeline (in execution order)
export const PREDICTION_AGENTS = [
  {
    key: "ingestion",
    label: "Ingestion Agent",
    description: "Load customer CSV → validate schema",
    stateKey: "raw_df",
    stepMs: 1500,
  },
  {
    key: "cleaning",
    label: "Cleaning Agent",
    description: "Apply the same cleaning steps as training for consistent input",
    stateKey: "clean_df",
    stepMs: 1500,
  },
  {
    key: "feature_engineering",
    label: "Feature Engineering Agent",
    description: "Generate the same 6 derived features the model was trained on",
    stateKey: "feature_df",
    stepMs: 2000,
  },
  {
    key: "prediction",
    label: "Prediction Agent",
    description: "Load saved model · generate churn_probability + risk_level per customer",
    stateKey: "predictions_df",
    stepMs: 4000,
  },
];
