# COMP 3610 – Assignment 4: MLOps & Model Deployment

**Course:** COMP 3610 – Big Data Analytics  
**University:** The University of the West Indies  
**Semester:** II, 2025–2026

A containerized, API-accessible tip prediction service built on top of NYC Yellow Taxi Trip Records. The service uses a PyTorch neural network trained in Assignment 2, tracked with MLflow, served via FastAPI, and orchestrated with Docker Compose.

---

## Prerequisites

| Tool                                       | Version                           |
| ------------------------------------------ | --------------------------------- |
| Python                                     | 3.11+                             |
| Docker Desktop (or Docker Engine on Linux) | Latest                            |
| Docker Compose                             | v2+ (bundled with Docker Desktop) |
| Git                                        | Any recent version                |

---

## Project Structure

```
assignment4/
├── assignment4.ipynb       # MLflow experiments, API dev, Docker walkthrough
├── app.py                  # FastAPI prediction service
├── test_app.py             # pytest test suite
├── Dockerfile              # Container definition for the API
├── docker-compose.yml      # Service orchestration (API + MLflow)
├── requirements.txt        # Pinned Python dependencies
├── README.md               # This file
├── .gitignore
├── .dockerignore
├── data/
│   └── taxi_zone_lookup.csv
└── models/                 # Gitignored — generated at runtime
    └── preprocessor.joblib
```

---

## Quickstart (Docker — Recommended)

### 1. Clone the repository

```bash
git clone https://github.com/DatDereDaag/COMP-3610-Assignment-4.git
cd COMP-3610-Assignment-4
```

### 2. Run MLflow server

```bash
mlflow ui --port 5000 --host 0.0.0.0  --disable-security-middleware
```

### 3. Run Notebook Experiment up to Part 3

### 4. Run DockerFile

```bash
docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 my-ml-api
```

### 5. Make A Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "VendorID": 1,
    "RatecodeID": 1,
    "PULocationID": 237,
    "DOLocationID": 140,
    "payment_type": 1,
    "passenger_count": 2,
    "trip_distance": 3.5,
    "fare_amount": 12.50,
    "extra": 0.5,
    "mta_tax": 0.5,
    "tolls_amount": 0.0,
    "improvement_surcharge": 0.3,
    "total_amount": 15.80,
    "congestion_surcharge": 2.5,
    "Airport_fee": 0.0,
    "trip_duration_mins": 18.0,
    "pickup_hour": 14,
    "pickup_day_of_week": 2
  }'
```

Or you can run Part 3 of the notebook

---

## Running Locally (Without Docker)

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Docker Services

```bash
docker compose up --build
```

### 4. Run Experiment Notebook

### 5. Make a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "VendorID": 1,
    "RatecodeID": 1,
    "PULocationID": 237,
    "DOLocationID": 140,
    "payment_type": 1,
    "passenger_count": 2,
    "trip_distance": 3.5,
    "fare_amount": 12.50,
    "extra": 0.5,
    "mta_tax": 0.5,
    "tolls_amount": 0.0,
    "improvement_surcharge": 0.3,
    "total_amount": 15.80,
    "congestion_surcharge": 2.5,
    "Airport_fee": 0.0,
    "trip_duration_mins": 18.0,
    "pickup_hour": 14,
    "pickup_day_of_week": 2
  }'
```

### 6. Shut down

```bash
docker compose down
```

---

## API Endpoints

| Method | Endpoint         | Description                                         |
| ------ | ---------------- | --------------------------------------------------- |
| GET    | `/`              | Welcome message and endpoint listing                |
| GET    | `/health`        | API status and model load status                    |
| GET    | `/info`          | Model name, version, features, and training metrics |
| POST   | `/predict`       | Single trip tip prediction                          |
| POST   | `/predict/batch` | Batch prediction (up to 100 records)                |

Interactive docs are available at `http://localhost:8000/docs` once the service is running.

### Example response (`/predict`)

```json
{
  "predicted_tip_amount": 2.85,
  "prediction_id": "f3a1c2d4-...",
  "model_version": "1"
}
```

---

## Running Tests

```bash
pytest test_app.py -v
```

The test suite covers single prediction, batch prediction, input validation (missing fields, out-of-range values), health check, and edge cases.

---

## MLflow UI

With the MLflow server running, open `http://localhost:5000` to view:

- All logged experiment runs under `taxi-tip-prediction`
- Side-by-side metric comparisons (MAE, RMSE, R²)
- Registered model versions in the Model Registry

---

## Environment Variables

| Variable              | Default                 | Description                   |
| --------------------- | ----------------------- | ----------------------------- |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server address         |
| `MODEL_NAME`          | `taxi-tip-regressor`    | Registered model name to load |

These are set in `docker-compose.yml` for the containerized deployment.

---

## Model Details

| Item            | Value                                                                                           |
| --------------- | ----------------------------------------------------------------------------------------------- |
| Model type      | PyTorch Neural Network (regression)                                                             |
| Target variable | `tip_amount`                                                                                    |
| Input features  | 39 features including trip distance, fare breakdown, time features, and one-hot encoded borough |
| Preprocessor    | Scikit-learn pipeline saved as `models/preprocessor.joblib`                                     |

---

## Notes

- Do **not** commit the NYC Taxi `.parquet` data files, `mlruns/`, or `models/` — these are gitignored.
- The notebook includes code to download the data programmatically if it is not present.
- `random_state=42` is used throughout for reproducibility.
