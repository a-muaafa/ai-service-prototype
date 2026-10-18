# AI Service Integration Prototype

> **⚠️ Prototype Notice:** This project uses **synthetic training data** generated
> from a fixed formula. Predictions are for **demonstration purposes only** and do
> not represent real property market values.

A REST API built with **FastAPI** that estimates property values using a trained
**scikit-learn GradientBoostingRegressor**. The project demonstrates a complete ML
service pipeline: data generation → training → evaluation → API serving → Docker
containerisation → automated testing.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client (curl / app)                      │
│                                                                 │
│  POST /api/v1/predict  { square_footage, bedrooms, … }         │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                      FastAPI Application                       │
│                                                               │
│  1. Middleware   → Request-ID · Rate limit · Auth · Size      │
│  2. Validation   → Pydantic schema (NaN/Inf rejected)         │
│  3. Prediction   → sklearn Pipeline.predict(features)         │
│  4. Response     → { predicted_value_usd, model_version, … }  │
└───────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│  model.joblib (sklearn Pipeline)                              │
│  ├── ColumnTransformer (StandardScaler + OrdinalEncoder)      │
│  └── GradientBoostingRegressor (200 trees, depth 4)           │
└───────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
ai-service-prototype/
├── main.py               # FastAPI app, schemas, model wrapper
├── train_model.py        # Synthetic data generation + training + evaluation
├── test_main.py          # pytest test suite (30+ test cases)
├── requirements.txt      # Pinned Python dependencies
├── Dockerfile            # Multi-stage production build
├── .dockerignore
├── .gitignore
├── .env.example          # Environment variable template
├── pyproject.toml        # ruff + pytest configuration
├── .github/
│   └── workflows/
│       └── ci.yml        # GitHub Actions CI pipeline
└── README.md
```

**Generated artifacts** (after training):

```
├── model.joblib          # Serialised sklearn Pipeline
├── model_metadata.json   # Version, metrics, feature ranges, sklearn version
└── evaluation_report.txt # Human-readable evaluation summary
```

---

## Requirements

- **Python** ≥ 3.11
- **Docker** (optional, for containerised deployment)
- See [`requirements.txt`](requirements.txt) for pinned package versions

---

## Quickstart (Local)

### 1. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py
```

This generates three files:
- `model.joblib` — serialised pipeline
- `model_metadata.json` — version, metrics, feature ranges
- `evaluation_report.txt` — human-readable summary

### 3. Start the API

```bash
python main.py
# or
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 4. Interactive docs

- **Swagger UI** → http://127.0.0.1:8000/docs
- **ReDoc** → http://127.0.0.1:8000/redoc

---

## Run with Docker

```bash
# 1. Generate the model first (Docker build needs it)
python train_model.py

# 2. Build
docker build -t ai-service-prototype .

# 3. Run
docker run -p 8000:8000 ai-service-prototype

# 4. (Optional) Run with API key authentication
docker run -p 8000:8000 -e API_KEY=my-secret-key ai-service-prototype
```

---

## Run Tests

```bash
# Train the model first (tests require it)
python train_model.py

# Run all tests
pytest -v
```

---

## API Reference

### `GET /` — Liveness probe

Returns 200 if the process is running.

```json
{ "status": "ok", "service": "AI Service Integration Prototype" }
```

### `GET /api/v1/health` — Process health

Returns 200 regardless of model state. Use for load-balancer health checks.

```json
{ "status": "healthy", "service": "AI Service Integration Prototype", "uptime_seconds": 42.7 }
```

### `GET /api/v1/ready` — Model readiness

Returns 200 when the model is loaded; **503** when not.

```json
{ "status": "ready", "model_loaded": true, "model_version": "2.0.0-gbr", "sklearn_version": "1.4.2" }
```

### `POST /api/v1/predict` — Property value estimation

**Request body:**

```json
{
  "square_footage": 1500.0,
  "num_bedrooms": 3,
  "num_bathrooms": 2.0,
  "age_years": 10,
  "location_tier": "suburban",
  "has_garage": true
}
```

| Field            | Type   | Constraints                                  |
|------------------|--------|----------------------------------------------|
| `square_footage` | float  | 200 < value ≤ 20,000 (sq ft)                |
| `num_bedrooms`   | int    | 1–20                                         |
| `num_bathrooms`  | float  | 1.0–10.0, multiples of 0.5                  |
| `age_years`      | int    | 0–200 (years)                                |
| `location_tier`  | string | `"urban"` \| `"suburban"` \| `"rural"`       |
| `has_garage`     | bool   | `true` / `false`                             |

> **NaN, Infinity, and extra fields are rejected.** Missing required fields
> return a 422 error.

**Success `200 OK`:**

```json
{
  "status": "success",
  "predicted_value_usd": 231450.75,
  "currency": "USD",
  "model_version": "2.0.0-gbr",
  "extrapolation_warning": false,
  "input_summary": { "square_footage": 1500.0, "..." : "..." }
}
```

**Validation error `422`:**

```json
{
  "status": "error",
  "detail": "[body -> square_footage] Input should be greater than 200",
  "request_id": "a1b2c3d4-..."
}
```

### HTTP Status Codes

| Code | Meaning                                                 |
|------|---------------------------------------------------------|
| 200  | Success                                                 |
| 401  | Unauthorized — API key required but missing/invalid     |
| 413  | Request body too large                                  |
| 422  | Validation error — invalid input                        |
| 429  | Rate limit exceeded                                     |
| 500  | Internal server error                                   |
| 503  | Service unavailable — model not loaded                  |

### Response Headers

Every response includes:
- `X-Request-ID` — unique request identifier for troubleshooting
- `X-Response-Time` — server processing time

---

## Configuration

All settings are configured via environment variables (see [`.env.example`](.env.example)):

| Variable          | Default | Description                                         |
|-------------------|---------|-----------------------------------------------------|
| `API_KEY`         | *(off)* | Enable API-key auth (header: `X-API-Key`)           |
| `RATE_LIMIT_MAX`  | `120`   | Max requests per client IP per window               |
| `RATE_LIMIT_WINDOW`| `60`   | Rate limit window (seconds)                         |
| `CORS_ORIGINS`    | *(off)* | Comma-separated allowed origins                     |
| `MAX_REQUEST_SIZE`| `16384` | Max request body (bytes)                            |

---

## Model Details

### Training Data

Data is **entirely synthetic**, generated from a known pricing formula:

```
price = sqft × 120 × location_multiplier
      + max(0, bedrooms − 1) × 8,000
      + max(0, bathrooms − 1) × 5,500
      − age × 1,200
      + garage × 15,000
      + noise(μ=0, σ=8,000)
```

| Location | Multiplier |
|----------|-----------|
| Urban    | 1.40      |
| Suburban | 1.00      |
| Rural    | 0.65      |

### Evaluation Metrics

Results from 5,000 synthetic samples (80/20 split, `random_state=42`):

| Metric | Model (GBR) | Baseline (mean) |
|--------|-------------|-----------------|
| MAE    | ~$5,700     | ~$185,000       |
| R²     | ~0.999      | 0.0             |

> **Note:** These metrics reflect the model's ability to approximate the known
> synthetic formula, not its accuracy on real-world data.

### Extrapolation Warning

When input features fall outside the training data range, the response
includes `"extrapolation_warning": true`. The prediction may be unreliable
in these cases.

### Replacing the Model

1. Update `train_model.py` with new data / configuration
2. Run `python train_model.py` to regenerate artifacts
3. Verify with `pytest -v`
4. Rebuild Docker image: `docker build -t ai-service-prototype .`

---

## Known Limitations

1. **Synthetic data only** — no real market data is used
2. **No uncertainty quantification** — predictions are point estimates
3. **In-memory rate limiter** — resets on restart, not shared across workers
4. **Single-worker default** — for sklearn thread safety; scale with replicas
5. **joblib security** — model file must only come from trusted sources

---

## Tech Stack

| Layer         | Technology                               |
|---------------|------------------------------------------|
| API           | FastAPI 0.111                            |
| Validation    | Pydantic v2                              |
| ML Model      | scikit-learn GradientBoostingRegressor    |
| Serialisation | joblib                                   |
| Server        | Uvicorn                                  |
| Container     | Docker (multi-stage, non-root)           |
| Testing       | pytest + HTTPX TestClient                |
| Linting       | Ruff                                     |
| CI            | GitHub Actions                           |
