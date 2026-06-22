# Customer Churn Prediction System (Multi-Agent System + LangGraph)

An end-to-end ML pipeline that predicts customer churn from the Telco Customer
Churn dataset, built as a set of specialized agents orchestrated with
**LangGraph**, tracked with **MLflow**, and served via **FastAPI**.

## Folder structure

```
Customer Churn MAS Project/
├── data/
│   ├── raw/                       # source CSVs (WA_Fn-UseC_-Telco-Customer-Churn.csv)
│   └── processed/                 # (reserved for cached intermediate data)
├── models/                        # persisted model + pipeline + metadata (created after training)
│   ├── churn_model.pkl
│   ├── preprocessing_pipeline.pkl
│   ├── model_metadata.json
│   └── reference_stats.json
├── mlruns/                        # MLflow tracking store (sqlite db + artifacts)
├── src/
│   ├── config.py                  # paths, schema, model + MLflow settings
│   ├── state.py                   # shared LangGraph state schema
│   ├── agents/
│   │   ├── ingestion.py           # Data Ingestion Agent
│   │   ├── cleaning.py            # Data Cleaning Agent
│   │   ├── feature_engineering.py # Feature Engineering Agent
│   │   ├── training.py            # Model Training Agent (LR / RF / XGBoost)
│   │   ├── evaluation.py          # Evaluation Agent (metrics + quality gate)
│   │   ├── registry.py            # Model Registry Agent (joblib + MLflow registry)
│   │   └── prediction.py          # Prediction Agent
│   ├── graphs/
│   │   ├── training_graph.py      # LangGraph training workflow
│   │   └── prediction_graph.py    # LangGraph prediction workflow
│   └── api/
│       └── main.py                # FastAPI app
├── frontend/                      # React (Vite + Tailwind) UI
│   ├── src/
│   │   ├── pages/                 # Dashboard, Train, Predict
│   │   ├── components/            # Layout, Card, MetricGrid, RiskBadge
│   │   └── api.js                 # API client (proxies /api -> FastAPI on :8000)
│   └── package.json
├── requirements.txt
└── README.md
```

## Agent pipeline

**Training graph**
```
Ingestion -> Cleaning -> Feature Engineering -> Training -> Evaluation -> Registry
```

**Prediction graph**
```
Ingestion -> Cleaning -> Feature Engineering -> Prediction
```

Each agent is a plain function `(state) -> state` operating on a shared
`ChurnPipelineState` TypedDict. Conditional edges short-circuit the graph if
any agent records an error.

### Agent responsibilities

| Agent | Responsibility |
|---|---|
| Ingestion | Load CSV, validate required columns/schema, report nulls/duplicates |
| Cleaning | Fix `TotalCharges` dtype, impute missing values, drop duplicates, normalize categorical text, encode target |
| Feature Engineering | `tenure_group`, `usage_intensity`, `avg_monthly_spend`, `num_streaming_services`, `num_security_services`, `payment_behavior` |
| Training | Build preprocessing `ColumnTransformer` (scaling + one-hot encoding), train Logistic Regression / Random Forest / XGBoost, log each to MLflow, pick the best by ROC-AUC |
| Evaluation | Compute accuracy/precision/recall/F1/ROC-AUC on the held-out test set, apply a quality gate (ROC-AUC >= 0.70) |
| Registry | Persist the winning pipeline with `joblib`, save metadata + reference stats, register in the MLflow Model Registry if the quality gate passed |
| Prediction | Load the persisted pipeline, score new customers, output `customerID`, `churn_probability`, `risk_level` |

Risk levels: `High` (>=0.7), `Medium` (>=0.4), `Low` (<0.4) - configurable in `src/config.py`.

## Setup

```bash
cd "Customer Churn MAS Project"
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Usage

### 1. Train the model (CLI)

```bash
python -m src.graphs.training_graph data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

This prints agent-by-agent logs, the candidate model comparison, and the
final evaluation report. Outputs are written to `models/` and an MLflow run
is recorded in `mlruns/mlflow.db`.

### 2. Score new customers (CLI)

```bash
python -m src.graphs.prediction_graph data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 3. Run the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

Endpoints:

- `GET /health` - liveness + whether a trained model exists
- `POST /train` - retrain (optionally upload a CSV; defaults to the bundled Telco dataset)
- `POST /predict` - upload a CSV of customers, get back churn probability + risk level per customer
- `GET /model/metadata` - metadata (metrics, model type, training time) for the current model

Example:

```bash
curl -X POST http://localhost:8000/train
curl -X POST -F "file=@data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv" http://localhost:8000/predict
```

### 4. Run the React frontend

```bash
cd frontend
npm install
npm run dev
```

Open the printed local URL (default http://localhost:5173). The dev server
proxies `/api/*` requests to the FastAPI backend at `http://localhost:8000`
(configured in `vite.config.js`), so make sure the API is running first
(`uvicorn src.api.main:app --reload --port 8000`).

Pages:

- **Dashboard** - current model type, evaluation metrics, model comparison chart, feature list
- **Train** - upload a CSV (or use the bundled dataset) and run the full training pipeline, with live metrics + model comparison chart
- **Predict** - upload a CSV of customers, view risk distribution and a filterable/paginated table of churn probabilities

### 5. View MLflow experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Open http://localhost:5000 to see all training runs (Logistic Regression,
Random Forest, XGBoost), their metrics, and the registered model versions.

## Extending the system

- **Monitoring/Drift Agent**: `models/reference_stats.json` stores per-feature
  mean/std/min/max from the training data - compare against new batches
  (e.g. population stability index) to detect drift.
- **Streamlit dashboard**: build a UI on top of `/predict` and `/model/metadata`
  to visualize churn risk and (optionally) SHAP explanations.
- **SHAP explanations**: load `models/churn_model.pkl` and run
  `shap.Explainer` on the fitted classifier step of the pipeline.
- **Deployment**: containerize with the provided `requirements.txt` and run
  `uvicorn src.api.main:app --host 0.0.0.0 --port 8000` behind a reverse proxy.
