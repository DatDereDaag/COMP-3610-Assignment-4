from fastapi.testclient import TestClient
from app import app
import pytest

#Setup the client from app
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c  # all tests in the module run here, then shutdown happens


#--------------------------------
# Positive Test Cases
#--------------------------------

def test_root(client):
    response = client.get("/")
    assert response.status_code == 200

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] == "up"
    assert data["model_version"] != "unavaiable"

def test_info(client):
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "model_version" in data
    assert "feature_names" in data
    assert "training_metrics" in data

def test_single_prediction(client):
    response = client.post("/predict", json={
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
        }
    )

    assert response.status_code == 200
    data= response.json()
    assert "predicted_tip_amount" in data
    assert "prediction_id" in data
    assert "model_version" in data

def test_batch_prediction(client):
    response = client.post("/predict/batch", json={
        "records": [
            {
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
            },
            {
                "VendorID": 2,
                "RatecodeID": 2,
                "PULocationID": 132,
                "DOLocationID": 237,
                "payment_type": 1,
                "passenger_count": 1,
                "trip_distance": 18.5,
                "fare_amount": 45.00,
                "extra": 1.0,
                "mta_tax": 0.5,
                "tolls_amount": 6.50,
                "improvement_surcharge": 0.3,
                "total_amount": 58.30,
                "congestion_surcharge": 2.5,
                "Airport_fee": 1.25,
                "trip_duration_mins": 45.0,
                "pickup_hour": 9,
                "pickup_day_of_week": 1
            },
            {
                "VendorID": 1,
                "RatecodeID": 1,
                "PULocationID": 161,
                "DOLocationID": 170,
                "payment_type": 2,
                "passenger_count": 3,
                "trip_distance": 2.1,
                "fare_amount": 8.50,
                "extra": 0.0,
                "mta_tax": 0.5,
                "tolls_amount": 0.0,
                "improvement_surcharge": 0.3,
                "total_amount": 10.80,
                "congestion_surcharge": 0.0,
                "Airport_fee": 0.0,
                "trip_duration_mins": 8.5,
                "pickup_hour": 23,
                "pickup_day_of_week": 5
            }
        ]
    })

    assert response.status_code == 200
    data= response.json()
    assert "predictions" in data
    assert "count" in data
    assert "processing_time_ms" in data

#--------------------------------
# Negative Test Cases
#--------------------------------
def test_missing_field_single_prediction(client):
    #Tests missing pickup_hour
    response = client.post("/predict", json={
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
        "pickup_day_of_week": 2
        }
    )

    assert response.status_code == 422

def test_invalid_field_single_prediction(client):
    #Tests with fare amount as string
    response = client.post("/predict", json={
        "VendorID": 1,
        "RatecodeID": 1,
        "PULocationID": 237,
        "DOLocationID": 140,
        "payment_type": 1,
        "passenger_count": 2,
        "trip_distance": 3.5,
        "fare_amount": "12.50",
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
        }
    )

    assert response.status_code == 422

def test_invalid_field_single_prediction(client):
    #Tests with out of range fare amount (<0)
    response = client.post("/predict", json={
        "VendorID": 1,
        "RatecodeID": 1,
        "PULocationID": 237,
        "DOLocationID": 140,
        "payment_type": 1,
        "passenger_count": 2,
        "trip_distance": 3.5,
        "fare_amount": -10,
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
        }
    )

    assert response.status_code == 422

def test_single_prediction_edge_case(client):
    #Trip distance set to 0
    response = client.post("/predict", json={
        "VendorID": 1,
        "RatecodeID": 1,
        "PULocationID": 237,
        "DOLocationID": 140,
        "payment_type": 1,
        "passenger_count": 2,
        "trip_distance": 0,
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
        }
    )

    assert response.status_code == 422