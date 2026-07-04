# MLOps Pipeline API

An end-to-end MLOps pipeline for a text classification model — built with FastAPI, MLflow (experiment tracking + model registry), and scikit-learn. Implements a full champion/challenger workflow: train new versions, compare them against production, and promote only the ones that actually win.

---

## How It Works

```
POST /train      →  train a new model version           → log params/metrics/model to MLflow
POST /promote     →  compare latest version to champion   → promote only if it performs better
POST /reload      →  reload in-memory champion from MLflow → no restart required
POST /predict     →  classify text                        → served from in-memory champion
GET  /model/info  →  metadata for the current champion
GET  /health      →  health check
```

---

## Table of Contents

- [Why Champion/Challenger?](#why-championchallenger)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [Running Tests](#running-tests)
- [API Endpoints](#api-endpoints)
- [Request & Response Schemas](#request--response-schemas)
- [Example Usage](#example-usage)
- [Docker](#docker)

---

## Why Champion/Challenger?

A model that scores well in training isn't automatically the one that should be serving predictions. Accuracy varies run to run, and blindly overwriting production with whatever was trained most recently is how regressions slip in unnoticed.

This API separates those two moments on purpose:

```
Train  (/train)    → creates a new version, logs everything to MLflow, does NOT touch production
Promote (/promote) → compares the new version's test accuracy against the current champion
                      → only assigns the @champion alias if the new version actually wins
```

Every version ever trained stays in MLflow's registry regardless of whether it was promoted — nothing is discarded, and any version can be inspected or manually promoted later from the MLflow UI if needed. This is the same alias-based pattern (`@champion`) MLflow 3.x replaced the older Staging/Production stage labels with.

---

## Model Details

A TF-IDF + Logistic Regression pipeline classifying text into five categories from the 20 Newsgroups dataset:

| Category |
|---|
| `sci.space` |
| `alt.atheism` |
| `talk.religion.misc` |
| `soc.religion.christian` |
| `sci.med` |

| Hyperparameter | Default | Description |
|---|---|---|
| `C` | `1.0` | Regularization strength |
| `max_features` | `50000` | Max TF-IDF vocabulary size |
| `ngram_range` | `(1, 1)` | N-gram range |
| `max_iter` | `1000` | Max solver iterations |

Every training run also logs the exact class-name order (`target_names.json`) as an artifact alongside the model — this is what lets `/predict` map a raw class index back to a real category string like `"sci.space"` instead of returning a bare integer.

---

## Project Structure

```
mlops-pipeline-api/
├── .github/
│   └── workflows/
│       └── ci.yml                — GitHub Actions CI pipeline
├── app/
│   ├── __init__.py
│   ├── main.py                    — FastAPI app, routes, app.state lifecycle
│   ├── schemas.py                 — Request/response schemas
│   └── services/
│       ├── __init__.py
│       ├── training.py            — train_model() — fit pipeline, log to MLflow
│       ├── serving.py             — Load champion model/metadata from registry
│       └── promotion.py           — Champion/challenger comparison logic
├── tests/
│   ├── __init__.py
│   ├── conftest.py                — Shared fixtures (fake MLflow objects)
│   ├── test_training.py
│   ├── test_serving.py
│   ├── test_promotion.py
│   └── test_main.py
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── Dockerfile                     — API image
├── Dockerfile.mlflow              — MLflow tracking server image
├── pytest.ini
├── README.md
└── requirements.txt
```

---

## Requirements

- Python 3.12+
- Docker and Docker Compose
- MLflow tracking server (handled by Docker Compose, or run standalone for local dev)

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/HonourJAH/mlops-pipeline-api.git
cd mlops-pipeline-api
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start an MLflow tracking server

```bash
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts
```

### 5. Set the tracking URI (if not using the default)

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
```

### 6. Start the API server

```bash
uvicorn app.main:app --reload
```

API available at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`
MLflow UI at `http://localhost:5000`

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | Tracking server the API and training code connect to |

> **Important:** In Docker Compose, this must be set to the service name (`http://mlflow:5000`), never `localhost` — containers reach each other by Compose service name, not `127.0.0.1`. `docker-compose.yml` already sets this correctly; only override it if you're running the API outside Docker against a differently-hosted MLflow server.

---

## Running Tests

MLflow is never called for real — every service function accepts an injectable `client` parameter, and tests pass a mock directly instead of hitting a live tracking server.

```bash
pytest -v
```

Run with coverage:

```bash
pytest -v --cov=app --cov-report=term-missing
```

---

## API Endpoints

| Method | Endpoint | Description | Status Code |
|---|---|---|---|
| `POST` | `/train` | Train a new model version, log to MLflow (does not promote) | `201 Created` |
| `POST` | `/promote` | Compare latest version to champion, promote if better | `200 OK` |
| `POST` | `/reload` | Reload in-memory champion model from the registry | `200 OK` |
| `POST` | `/predict` | Classify text using the current champion | `200 OK` |
| `GET` | `/model/info` | Metadata for the currently loaded champion | `200 OK` |
| `GET` | `/health` | Health check + whether a champion is loaded | `200 OK` |

---

## Request & Response Schemas

### `POST /train`

**Request body** — all fields optional, defaults shown:

```json
{
  "C": 1.0,
  "max_features": 50000,
  "ngram_range": [1, 1],
  "max_iter": 1000
}
```

**Response:**

```json
{
  "run_id": "2970ca14b01a488bb3fb28563baf4792",
  "training_accuracy": 0.958,
  "test_accuracy": 0.7133,
  "params": {
    "C": 1.0,
    "max_features": 50000,
    "ngram_range": [1, 1],
    "max_iter": 1000
  },
  "message": "Training complete. Run ID: 2970ca14b01a488bb3fb28563baf4792. Review metrics in the MLflow UI and promote to @champion if this version outperforms the current one."
}
```

---

### `POST /promote`

**Response — promoted:**

```json
{
  "promoted": true,
  "reason": "Challenger outperformed the current champion.",
  "new_champion_version": "11",
  "challenger_accuracy": 0.8123,
  "previous_champion_accuracy": 0.7133,
  "previous_champion_version": "10"
}
```

**Response — not promoted:**

```json
{
  "promoted": false,
  "reason": "Challenger did not outperform the current champion.",
  "challenger_version": "11",
  "challenger_accuracy": 0.68,
  "champion_version": "10",
  "champion_accuracy": 0.7133
}
```

---

### `POST /predict`

**Request body:**

```json
{ "text": "NASA launched a new spacecraft into orbit today" }
```

`text` requires a minimum of 10 characters.

**Response:**

```json
{
  "text": "NASA launched a new spacecraft into orbit today",
  "category": "sci.space",
  "confidence": 0.8123,
  "model_version": "11"
}
```

---

### `GET /model/info`

```json
{
  "model_name": "newsgroups-classifier",
  "version": "11",
  "aliases": ["champion"],
  "run_id": "2970ca14b01a488bb3fb28563baf4792"
}
```

---

### `GET /health`

```json
{
  "status": "healthy",
  "champion_loaded": true,
  "model_version": "11"
}
```

---

## Example Usage

### Full train → promote → predict cycle

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" -d '{}'
```

```bash
curl -X POST http://localhost:8000/promote
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "NASA launched a new spacecraft into orbit today"}'
```

### Reload after manually promoting a version in the MLflow UI

```bash
curl -X POST http://localhost:8000/reload
```

### Check what's currently serving

```bash
curl http://localhost:8000/model/info
```

---

## Docker

### Run with Docker Compose

Starts the MLflow tracking server and the API together, wired to reach each other by Compose service name.

```bash
docker compose up --build
```

### Stop everything

```bash
docker compose down
```

### Services

| Service | Port | Description |
|---|---|---|
| `api` | `8000` | FastAPI server |
| `mlflow` | `5000` | MLflow tracking server + model registry |

### Persistence

MLflow's backend store, artifact store (including every `target_names.json`), and the cached 20 Newsgroups dataset are all mounted as named volumes (`mlflow-data`, `sklearn-data`), so the full run/version history survives container restarts. Use `docker compose down -v` only when you intentionally want a clean slate — it deletes both volumes, including every trained version and the current champion.

### Build the image only

```bash
docker build -t mlops-pipeline-api .
docker build -t mlops-pipeline-mlflow -f Dockerfile.mlflow .
```
