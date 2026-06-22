# Customer Churn Prediction — Multi-Agent System

## What This Project Does

This project predicts which telecom customers are likely to cancel their subscription (**churn**).
It is built as a **Multi-Agent System (MAS)** — instead of one big monolithic script, the ML
pipeline is split into small, focused **agents**, each responsible for one stage of the data
science workflow. Those agents are wired together and orchestrated by **LangGraph**.

The end product is:
- A trained scikit-learn classification pipeline (stored on disk + registered in MLflow)
- A REST API (FastAPI) that can retrain the model or score new customers
- A React dashboard that exposes all of this through a browser UI

---

## Dataset

**Source:** IBM Telco Customer Churn dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)

| Property | Value |
|---|---|
| Rows | 7,043 customers |
| Columns | 21 raw features + 1 target |
| Target | `Churn` — Yes / No (binary classification) |
| Class imbalance | ~26% churn, ~74% non-churn |

### Raw Features

| Feature | Type | Description |
|---|---|---|
| `customerID` | ID | Unique customer identifier |
| `gender` | Categorical | Male / Female |
| `SeniorCitizen` | Binary (0/1) | Whether the customer is a senior |
| `Partner` | Categorical | Has a partner (Yes/No) |
| `Dependents` | Categorical | Has dependents (Yes/No) |
| `tenure` | Numeric | Months the customer has been with the company |
| `PhoneService` | Categorical | Has phone service |
| `MultipleLines` | Categorical | Has multiple phone lines |
| `InternetService` | Categorical | DSL / Fiber optic / No |
| `OnlineSecurity` | Categorical | Has online security add-on |
| `OnlineBackup` | Categorical | Has online backup add-on |
| `DeviceProtection` | Categorical | Has device protection add-on |
| `TechSupport` | Categorical | Has tech support add-on |
| `StreamingTV` | Categorical | Streams TV |
| `StreamingMovies` | Categorical | Streams movies |
| `Contract` | Categorical | Month-to-month / One year / Two year |
| `PaperlessBilling` | Categorical | Uses paperless billing |
| `PaymentMethod` | Categorical | Electronic check / Mailed check / Bank transfer / Credit card |
| `MonthlyCharges` | Numeric | Current monthly bill ($) |
| `TotalCharges` | Numeric | Total amount charged to date ($) |
| `Churn` | Target | Did the customer leave? (Yes → 1, No → 0) |

---

## Architecture — Multi-Agent System

The core idea is that each stage of the ML workflow is an **agent**: an isolated Python function
that reads from a shared state dictionary, does its work, and writes results back to that state.
**LangGraph** manages which agent runs next and short-circuits the pipeline on errors.

```
                        TRAINING PIPELINE
  ┌───────────┐    ┌──────────┐    ┌─────────────────────┐
  │ Ingestion │ →  │ Cleaning │ →  │ Feature Engineering │
  └───────────┘    └──────────┘    └─────────────────────┘
                                              │
                                              ▼
                                       ┌──────────┐
                                       │ Training │  (LR + RF + XGBoost)
                                       └──────────┘
                                              │
                                              ▼
                                      ┌────────────┐
                                      │ Evaluation │  (quality gate)
                                      └────────────┘
                                              │
                                              ▼
                                       ┌──────────┐
                                       │ Registry │  (disk + MLflow)
                                       └──────────┘

                       PREDICTION PIPELINE
  ┌───────────┐    ┌──────────┐    ┌─────────────────────┐    ┌────────────┐
  │ Ingestion │ →  │ Cleaning │ →  │ Feature Engineering │ →  │ Prediction │
  └───────────┘    └──────────┘    └─────────────────────┘    └────────────┘
```

Each arrow above is a **conditional edge** — if any agent writes an error to the shared state,
LangGraph stops the pipeline immediately rather than cascading bad data downstream.

---

## Shared State — `ChurnPipelineState`

**File:** `src/state.py`

All agents communicate through a single `TypedDict` that flows through the graph.
No agent imports another agent directly — they only read and write this shared state object.

```python
class ChurnPipelineState(TypedDict, total=False):
    # Input
    input_path: str          # path to the CSV file
    mode: str                # "train" or "predict"

    # Populated by each agent
    raw_df: pd.DataFrame         # after Ingestion
    clean_df: pd.DataFrame       # after Cleaning
    feature_df: pd.DataFrame     # after Feature Engineering
    model: Any                   # best fitted sklearn Pipeline
    X_test / y_test              # held-out evaluation set
    evaluation_report: dict      # metrics + quality gate result
    predictions_df: pd.DataFrame # scored customer results
    registry_report: dict        # artifact paths + MLflow status

    errors: List[str]            # any agent can append here to halt the graph
    logs: List[str]              # human-readable trace of what each agent did
```

---

## Agent Breakdown

### 1. Ingestion Agent
**File:** `src/agents/ingestion.py`

**What it does:**
Loads the raw CSV file and validates that the dataset is usable before any processing begins.

**Key functions:**

| Function | Purpose |
|---|---|
| `ingestion_agent(state)` | LangGraph node — reads the CSV, validates the schema, reports basic stats |

**Data science significance:**
- Checks that all 20 required feature columns + the target column are present
- Reports duplicate row count and null counts per column so downstream agents know what to expect
- In *predict* mode, the target column (`Churn`) is not required since we are scoring new customers

**Output added to state:** `raw_df`, `ingestion_report`

---

### 2. Cleaning Agent
**File:** `src/agents/cleaning.py`

**What it does:**
Fixes data quality issues that would otherwise silently corrupt model training.

**Key functions:**

| Function | Purpose |
|---|---|
| `_coerce_total_charges(df)` | `TotalCharges` is stored as a string in the raw CSV (blank strings for new customers with zero charges). Converts it to float using `pd.to_numeric(..., errors='coerce')` |
| `cleaning_agent(state)` | LangGraph node — runs all cleaning steps and returns `clean_df` |

**Cleaning steps:**

1. **Type coercion** — `TotalCharges` (string → float); blank strings become `NaN`
2. **Drop duplicates** — removes exact duplicate rows
3. **Numeric imputation** — fills `NaN` in `TotalCharges`, `MonthlyCharges`, and `tenure` with their **median** (median is preferred over mean here because these columns can be right-skewed)
4. **Categorical normalization** — strips whitespace; maps `"No internet service"` and `"No phone service"` to `"No"` (they are functionally the same for modelling purposes)
5. **Target encoding** — `Yes → 1`, `No → 0`; rows where the target cannot be mapped are dropped

**Output added to state:** `clean_df`, `cleaning_report`

---

### 3. Feature Engineering Agent
**File:** `src/agents/feature_engineering.py`

**What it does:**
Creates 6 new features from the existing raw columns. These features capture **customer
behavior patterns** that are more informative than the raw numbers alone.

**Key functions:**

| Function | Purpose |
|---|---|
| `_tenure_group(tenure)` | Bins raw tenure (months) into 4 ordinal categories |
| `_payment_behavior(payment_method)` | Collapses 4 payment methods into Automatic / Manual |
| `_count_yes(df, cols)` | Counts how many "Yes" values a customer has across a list of binary columns |
| `feature_engineering_agent(state)` | LangGraph node — adds all derived features and returns `feature_df` |

**Engineered Features:**

| Feature | Type | How It's Made | Why It Matters |
|---|---|---|---|
| `tenure_group` | Categorical | Bins `tenure` into New (≤12m), Mid (≤24m), Loyal (≤48m), Veteran (>48m) | Churn risk is strongly non-linear with tenure — new customers churn at much higher rates than veterans |
| `usage_intensity` | Numeric | `MonthlyCharges / max(tenure, 1)` | Measures how much a customer spends relative to how long they've been a customer — high spend + low tenure is a churn risk signal |
| `avg_monthly_spend` | Numeric | `TotalCharges / max(tenure, 1)` | Lifetime average bill — catches customers whose spending has grown over time |
| `num_streaming_services` | Numeric | Count of "Yes" across `StreamingTV`, `StreamingMovies` (0–2) | Customers subscribed to more services have more switching friction |
| `num_security_services` | Numeric | Count of "Yes" across `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport` (0–4) | Same switching-friction argument; security add-ons indicate deeper product engagement |
| `payment_behavior` | Categorical | `"Automatic"` if payment method contains "automatic", else `"Manual"` | Customers on automatic payment churn less — manual payment customers must actively choose to stay each month |

**Output added to state:** `feature_df`, `feature_report`

---

### 4. Training Agent
**File:** `src/agents/training.py`

**What it does:**
Trains three candidate models, evaluates all of them on the held-out test set, and selects the
best one by ROC-AUC.

**Key functions:**

| Function | Purpose |
|---|---|
| `_build_preprocessor(df)` | Builds a `ColumnTransformer` with separate pipelines for numeric and categorical features |
| `_candidate_models()` | Returns a dict of the three candidate classifiers |
| `training_agent(state)` | LangGraph node — trains all candidates, logs to MLflow, selects best model |

**Preprocessing pipeline (built with scikit-learn):**

```
Numeric columns  →  SimpleImputer(median)  →  StandardScaler
Categorical cols →  SimpleImputer(most_frequent)  →  OneHotEncoder(handle_unknown='ignore')
                                        ↓
                              ColumnTransformer
                                        ↓
                              + classifier (LR / RF / XGBoost)
                          = one sklearn Pipeline object
```

The entire preprocessing + model is wrapped in a single `sklearn.Pipeline` — this means
the same transformations that were applied during training are automatically applied at
prediction time with no extra code.

**Candidate models:**

| Model | Key Hyperparameters | Notes |
|---|---|---|
| `LogisticRegression` | `max_iter=1000`, `class_weight='balanced'` | Fast, interpretable baseline; `class_weight='balanced'` compensates for the 26/74 class imbalance |
| `RandomForestClassifier` | `n_estimators=300`, `max_depth=10`, `class_weight='balanced'` | Ensemble of 300 decision trees; handles non-linear relationships |
| `XGBClassifier` | `n_estimators=300`, `learning_rate=0.05`, `max_depth=4`, `subsample=0.8` | Gradient boosting — typically strongest on tabular data |

**Train / test split:** 80% train / 20% test, stratified on the target to preserve the 26/74 class ratio.

Each candidate gets its own MLflow run with params + metrics logged. The best model by
**ROC-AUC** is selected and passed to the Evaluation Agent.

**Output added to state:** `model`, `pipeline`, `model_name`, `candidate_results`, `X_test`, `y_test`

---

### 5. Evaluation Agent
**File:** `src/agents/evaluation.py`

**What it does:**
Computes the final metrics for the selected best model on the held-out test set and applies
a **quality gate** — a hard threshold that must be met before the model is registered.

**Key functions:**

| Function | Purpose |
|---|---|
| `evaluation_agent(state)` | LangGraph node — scores the best model, logs metrics to MLflow, returns evaluation report |

**Metrics computed:**

| Metric | What It Measures | Why It's Used Here |
|---|---|---|
| **ROC-AUC** | Area under the Receiver Operating Characteristic curve | Primary selection metric; measures the model's ability to rank churners above non-churners across all probability thresholds — robust to class imbalance |
| **F1 Score** | Harmonic mean of precision and recall | Good single-number summary when both false positives (wasted retention spend) and false negatives (missed churners) matter |
| **Precision** | Of all predicted churners, how many actually churned | Relevant if retention campaigns have a high cost-per-contact |
| **Recall** | Of all actual churners, how many did we catch | Relevant if missing a churner is very costly |
| **Accuracy** | Overall correct predictions / total | Reported for completeness; less meaningful with class imbalance |

**Quality gate:** `ROC-AUC ≥ 0.70`

If the best model does not pass the quality gate, it is still saved to disk for reference but
**not registered in the MLflow Model Registry**. This prevents a poorly trained model
from being accidentally served in production.

**Output added to state:** `metrics`, `evaluation_report`

---

### 6. Registry Agent
**File:** `src/agents/registry.py`

**What it does:**
Persists everything needed to use the model in the future — the serialized pipeline, metadata,
reference statistics for drift monitoring, and an MLflow Model Registry entry.

**Key functions:**

| Function | Purpose |
|---|---|
| `_compute_reference_stats(df)` | Computes per-column mean, std, min, max for numeric features of the training set |
| `registry_agent(state)` | LangGraph node — saves all artifacts and registers the model |

**Artifacts saved:**

| File | Location | Purpose |
|---|---|---|
| `churn_model.pkl` | `models/` | Serialized sklearn Pipeline (preprocessing + classifier) via joblib |
| `preprocessing_pipeline.pkl` | `models/` | Same pipeline object — available under a separate name for pipelines that need only the transformer |
| `model_metadata.json` | `models/` | Model name, training timestamp, all metrics, candidate comparison, feature column list |
| `reference_stats.json` | `models/` | Training data statistics per numeric feature — used as a baseline to detect **data drift** when scoring new data |

**MLflow registration:**
If the quality gate passed, a new version of `customer_churn_classifier` is registered in the
MLflow Model Registry. This enables version tracking and stage promotion (Staging → Production).

**Output added to state:** `registry_report`

---

### 7. Prediction Agent
**File:** `src/agents/prediction.py`

**What it does:**
Loads the saved model from disk and scores a new batch of customers.

**Key functions:**

| Function | Purpose |
|---|---|
| `_risk_level(prob)` | Maps a churn probability to a human-readable risk tier |
| `prediction_agent(state)` | LangGraph node — loads the model, generates probabilities and risk labels for every customer |

**Risk tiers:**

| Tier | Probability Threshold | Interpretation |
|---|---|---|
| **High** | ≥ 0.70 | Strong churn signal — prioritise for retention outreach |
| **Medium** | 0.40 – 0.69 | Moderate risk — worth monitoring |
| **Low** | < 0.40 | Likely to stay — no immediate action needed |

**Output added to state:** `predictions_df` (`customerID`, `churn_probability`, `risk_level`, optionally `actual_churn`), `prediction_report`

---

## LangGraph Graphs

LangGraph turns the agents into a directed acyclic graph (DAG) by defining nodes (agents) and
edges (transitions between them).

### Training Graph — `src/graphs/training_graph.py`

**Key functions:**

| Function | Purpose |
|---|---|
| `_has_errors(state)` | Conditional edge function — returns `"error"` if any agent wrote to `state["errors"]`, otherwise `"continue"`. This is what makes the pipeline fail-fast |
| `build_training_graph()` | Assembles and compiles the 6-node StateGraph |
| `run_training(input_path)` | Convenience wrapper — initialises state and calls `app.invoke()` |

### Prediction Graph — `src/graphs/prediction_graph.py`

**Key functions:**

| Function | Purpose |
|---|---|
| `build_prediction_graph()` | Assembles the 4-node prediction StateGraph (skips Training, Evaluation, Registry) |
| `run_prediction(input_path)` | Convenience wrapper |

**Why re-run Ingestion → Cleaning → Feature Engineering for prediction?**
Because new data arriving in production may have the same quality issues as the training data
(blank `TotalCharges`, inconsistent categorical labels, etc.). Running the same cleaning and
feature engineering steps ensures the input to the model is always in the shape the model
was trained on.

---

## MLflow Experiment Tracking

**Tracking URI:** SQLite database at `mlruns/mlflow.db`
**Experiment name:** `customer_churn_mas`

Every model candidate in a training run gets its own MLflow run with:
- **Params:** model type, all hyperparameters from `get_params()`
- **Metrics:** accuracy, precision, recall, F1, ROC-AUC
- **Artifact:** the fitted sklearn Pipeline (serialized via skops)

A separate consolidated run named `best_model__<model_name>` is logged by the Evaluation Agent
to give a clean single-run view of the chosen model.

**Trusted types for skops serialization** (configured in `src/config.py`):
```python
SKOPS_TRUSTED_TYPES = ["numpy.dtype", "xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]
```
skops audits model files for unexpected Python types before loading them. These three are
explicitly trusted because they come from our own training code.

---

## Configuration — `src/config.py`

All tunable constants live in one place:

| Constant | Value | Purpose |
|---|---|---|
| `RANDOM_STATE` | 42 | Seed for all train/test splits and model random states (reproducibility) |
| `TEST_SIZE` | 0.2 | 20% held-out test set |
| `RISK_THRESHOLDS` | High ≥ 0.7, Medium ≥ 0.4 | Churn probability → risk tier mapping |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlruns/mlflow.db` | Local SQLite for the MLflow backend |
| `MLFLOW_REGISTERED_MODEL_NAME` | `customer_churn_classifier` | Name in the MLflow Model Registry |
| `SKOPS_TRUSTED_TYPES` | `[numpy.dtype, xgboost.*, ...]` | Types allowed when deserializing saved models |

---

## REST API — `src/api/main.py`

Built with **FastAPI**. Start with:
```bash
uvicorn src.api.main:app --reload --port 8000
```

| Endpoint | Method | What It Does |
|---|---|---|
| `/health` | GET | Returns `{ status, model_available }`. Used by the UI to show the model status indicator |
| `/train` | POST | Accepts an optional CSV upload. If no file provided, uses the bundled Telco dataset. Runs the full training graph and returns metrics, candidate comparison, registry info, and pipeline logs |
| `/predict` | POST | Accepts a CSV upload. Runs the prediction graph and returns `churn_probability` + `risk_level` for each customer |
| `/model/metadata` | GET | Returns the contents of `models/model_metadata.json` — model name, training time, metrics, feature columns |

CORS is configured with `allow_origins=["*"]` so the React dev server on port 5173 can
call the API on port 8000 without browser security errors.

---

## React Frontend — `frontend/`

Built with **Vite + React + Tailwind CSS v4 + Recharts**. Start with:
```bash
cd frontend
npm run dev   # → http://localhost:5173
```

Vite proxies all `/api/*` requests to `http://localhost:8000` (stripping the `/api` prefix),
so the frontend never needs to know the backend port.

### Pages

| Page | Route | What It Shows |
|---|---|---|
| **Dashboard** | `/` | Current model name, training time, quality gate status, evaluation metrics, bar chart comparing all candidate models, list of feature columns used |
| **Train** | `/train` | File upload (or use bundled dataset), "Run training pipeline" button, live loading state, post-training metrics + model comparison chart + pipeline log |
| **Predict** | `/predict` | CSV upload, scoring results — summary stats, pie chart of risk distribution, paginated table with sort by probability, filter by risk tier, customer ID search, Export CSV |

### Components

| Component | File | Purpose |
|---|---|---|
| `Layout` | `components/Layout.jsx` | Sticky header with navigation and a live model-status dot (green = ready, amber = no model, grey = API down) |
| `Card` | `components/Card.jsx` | Consistent dark card container with optional title |
| `MetricGrid` | `components/MetricGrid.jsx` | Displays the 5 evaluation metrics in a responsive grid |
| `RiskBadge` | `components/RiskBadge.jsx` | Colour-coded pill badge for High / Medium / Low risk |
| `ProbabilityBar` | `components/ProbabilityBar.jsx` | Inline progress bar that colour-codes by risk threshold (red ≥70%, amber ≥40%, green <40%) |
| `Spinner` | `components/Spinner.jsx` | Animated SVG spinner for loading states |

### Shared Constants

`frontend/src/lib/constants.js` — single source of truth for `MODEL_LABELS`, `RISK_COLORS`,
and `CHART_TOOLTIP_STYLE`. Previously these were duplicated across multiple page files.

---

## Project File Structure

```
Customer Churn MAS Project/
│
├── src/
│   ├── config.py                  # all constants and paths
│   ├── state.py                   # ChurnPipelineState TypedDict
│   │
│   ├── agents/
│   │   ├── ingestion.py           # load + validate CSV
│   │   ├── cleaning.py            # fix dtypes, impute, encode target
│   │   ├── feature_engineering.py # build 6 derived features
│   │   ├── training.py            # train LR + RF + XGBoost, pick best
│   │   ├── evaluation.py          # final metrics + quality gate
│   │   ├── registry.py            # save artifacts + MLflow registration
│   │   └── prediction.py          # load model, score new customers
│   │
│   ├── graphs/
│   │   ├── training_graph.py      # 6-node training DAG
│   │   └── prediction_graph.py    # 4-node prediction DAG
│   │
│   └── api/
│       └── main.py                # FastAPI app (4 endpoints)
│
├── frontend/                      # React UI (Vite + Tailwind + Recharts)
│   └── src/
│       ├── api.js                 # axios client (proxied to :8000)
│       ├── lib/constants.js       # shared MODEL_LABELS, RISK_COLORS
│       ├── components/            # Layout, Card, MetricGrid, etc.
│       └── pages/                 # Dashboard, Train, Predict
│
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/                        # written at runtime after training
│   ├── churn_model.pkl
│   ├── preprocessing_pipeline.pkl
│   ├── model_metadata.json
│   └── reference_stats.json
│
├── mlruns/
│   └── mlflow.db                  # SQLite MLflow tracking store
│
└── requirements.txt
```

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.10+ |
| ML models | scikit-learn | ≥ 1.3 |
| Gradient boosting | XGBoost | ≥ 2.0 |
| Orchestration | LangGraph | ≥ 0.2 |
| Experiment tracking | MLflow | ≥ 2.10 |
| Model serialization | joblib + skops | — |
| API | FastAPI + Uvicorn | ≥ 0.110 |
| Frontend framework | React + Vite | 19 / 8 |
| Styling | Tailwind CSS | v4 |
| Charts | Recharts | ≥ 3 |
| HTTP client | Axios | ≥ 1.17 |

---

## How to Run

### 1. Train from the command line
```bash
# Uses the bundled Telco dataset
python -m src.graphs.training_graph

# Or specify your own CSV
python -m src.graphs.training_graph data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 2. Score customers from the command line
```bash
python -m src.graphs.prediction_graph data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 3. Run via the API + UI
```bash
# Terminal 1 — backend
uvicorn src.api.main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
npm run dev
```
Then open **http://localhost:5173** in your browser.
