#All imports below
import os
import uuid
import time

from typing import List

import mlflow
from mlflow.tracking import MlflowClient

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field

import torch
import torch.nn as nn

import numpy as np

import pandas as pd

import joblib

from models.regression_nn import RegressionNeuralNetwork #This is the class needed for the neural network

#----------------------------------
# Loading Model On Startup
#----------------------------------
tip_predictor_model_state = None
zone_df = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    #Loading the preprocessor and lookup table for taxi zones
    global zone_df
    try:
        preprocessor = joblib.load("models/preprocessor.joblib") #Get our preprocessor used for the data
        zone_df = pd.read_csv("data/taxi_zone_lookup.csv") #Get the taxi zone lookup table
    except FileNotFoundError as e:
        raise RuntimeError(f"Required file not found: {e}\n")

    #Loading the neural network from mlflow
    global tip_predictor_model_state
    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        versions = client.get_latest_versions("taxi-tip-regressor", stages=["None"])
        latest = versions[0]

        best_model = mlflow.pytorch.load_model(f"models:/taxi-tip-regressor/{latest.version}") #Loads the latest version of the model registered
        best_model.eval() #Set to eval mode

        run = client.get_run(latest.run_id)

        tip_predictor_model_state = {
            'model': best_model,
            'preprocessor': preprocessor,
            'version': str(latest.version),
            'model_name': 'taxi-tip-regressor',
            'features': [
                "passenger_count", "trip_distance", "RatecodeID", "payment_type",
                "fare_amount", "extra", "mta_tax", "tolls_amount",
                "improvement_surcharge", "total_amount", "congestion_surcharge",
                "Airport_fee", "trip_duration_minutes", "trip_speed_mph",
                "log_trip_distance", "fare_per_mile", "fare_per_minute",
                "pickup_borough_*", "dropoff_borough_*",
                "VendorID", "PULocationID", "DOLocationID",
                "pickup_hour", "pickup_day_of_week", "is_weekend",
            ],
            'metrics': {
                    "MAE":  run.data.metrics.get("mae"),
                    "RMSE": run.data.metrics.get("rmse"),
                    "R2":   run.data.metrics.get("r2"),
            }
        }

    except Exception as e:
        raise RuntimeError(f"Model loading failed: {e}")

    print("Neural Network Model has been loaded successfully")
    
    yield
    print("Shutting down...")

#----------------------------------
# App initialization
#----------------------------------
app = FastAPI(
    title="Taxi Tip Prediction API", 
    description="API for predicting taxi tips based on various features", 
    version="1.0.0", 
    lifespan=lifespan
    )

#----------------------------------
# Data Preprocessing Functins
#----------------------------------

#--------Borrough Encoding---------

BOROUGH_CATEGORIES = ["Bronx", "Brooklyn", "EWR", "Manhattan", "N/A", "Queens", "Staten Island", "Unknown"]

#This function simulates one hot encoding on all boroughs
#It has to add all boroughs to the input and give them a value
#of 1 if true or 0 if false
def one_hot_borough(borough_name: str, prefix: str) -> dict:
    return {f"{prefix}_{cat}": (1.0 if borough_name == cat else 0.0) for cat in BOROUGH_CATEGORIES}

#--------Rebuilding OneHotEncoded Columns and Derived Columns-----------

def build_raw_row(data: "TripFeatures") -> pd.DataFrame:
    #Recreating derived columns that the neural network was trained on
    trip_duration_minutes = data.trip_duration_mins
    trip_speed_mph = (data.trip_distance / trip_duration_minutes * 60) if trip_duration_minutes > 0 else 0.0
    log_trip_distance = float(np.log1p(data.trip_distance))
    fare_per_mile = data.fare_amount / data.trip_distance if data.trip_distance > 0 else 0.0 #Reasonably the distance would never be 0
    fare_per_minute = data.fare_amount / trip_duration_minutes if trip_duration_minutes > 0 else 0.0
 
    #Get borrough from lookup table
    pickup_borough = zone_df.loc[data.PULocationID, "Borough"] if data.PULocationID in zone_df.index else "Unknown"
    dropoff_borough = zone_df.loc[data.DOLocationID, "Borough"] if data.DOLocationID in zone_df.index else "Unknown"
 
    #Below simulates OneHotEncoding from the preprocessing done on the data
    #that was used to train the neural network
    pickup_ohe  = one_hot_borough(pickup_borough,  "pickup_borough")
    dropoff_ohe = one_hot_borough(dropoff_borough, "dropoff_borough")
 
    is_weekend = data.pickup_day_of_week in (5, 6)
 
    #Recreating all 39 columns
    row = {
        "passenger_count":       data.passenger_count,
        "trip_distance":         data.trip_distance,
        "RatecodeID":            data.RatecodeID,
        "payment_type":          data.payment_type,
        "fare_amount":           data.fare_amount,
        "extra":                 data.extra,
        "mta_tax":               data.mta_tax,
        "tolls_amount":          data.tolls_amount,
        "improvement_surcharge": data.improvement_surcharge,
        "total_amount":          data.total_amount,
        "congestion_surcharge":  data.congestion_surcharge,
        "Airport_fee":           data.Airport_fee,
        "trip_duration_minutes": trip_duration_minutes,
        "trip_speed_mph":        trip_speed_mph,
        "log_trip_distance":     log_trip_distance,
        "fare_per_mile":         fare_per_mile,
        "fare_per_minute":       fare_per_minute,
        **pickup_ohe,    # pickup_borough_Bronx ... pickup_borough_Unknown
        **dropoff_ohe,   # dropoff_borough_Bronx ... dropoff_borough_Unknown
        "VendorID":              data.VendorID,
        "PULocationID":          data.PULocationID,
        "DOLocationID":          data.DOLocationID,
        "pickup_hour":           data.pickup_hour,
        "pickup_day_of_week":    data.pickup_day_of_week,
        "is_weekend":            is_weekend,
    }
 
    return pd.DataFrame([row])
 

#--------Rebuilding Preprocessed Columns-----------

def preprocess_and_tensorize(raw_df: pd.DataFrame) -> torch.Tensor:
    preprocessor = tip_predictor_model_state['preprocessor']
    X = preprocessor.transform(raw_df)
 
    if hasattr(X, "toarray"):  # handle sparse output
        X = X.toarray()
 
    X = np.nan_to_num(X.astype(np.float32))
    return torch.tensor(X, dtype=torch.float32)
 

#-----------------------------------
# API Response Models
#-----------------------------------

#--------Input Validation Models------------------

#Model for validating incoming trip features for prediction
class TripFeatures(BaseModel):
    VendorID: int         = Field(..., ge=1, le=2,   description="Vendor (1 or 2)", examples=[1])
    RatecodeID: int       = Field(..., ge=1, le=6,   description="Rate code (1=Standard, 2=JFK...)", examples=[1])
    PULocationID: int     = Field(..., ge=1, le=265, description="TLC pickup zone ID", examples=[237])
    DOLocationID: int     = Field(..., ge=1, le=265, description="TLC dropoff zone ID", examples=[140])
    payment_type: int     = Field(..., ge=1, le=6,   description="1=Credit card, 2=Cash, 3=No Charge, 4=Dispute, 0=Other", examples=[1])
    passenger_count: int  = Field(..., ge=1, le=5,   description="Number of passengers", examples=[2])
 
    # Fare breakdown
    trip_distance: float        = Field(..., gt=0,          description="Miles", examples=[3.5])
    fare_amount: float          = Field(..., ge=0, le=500,  description="Metered fare (USD)", examples=[12.50])
    extra: float                = Field(default=0.0, ge=0,  description="Extras/surcharges", examples=[0.5])
    mta_tax: float              = Field(default=0.5, ge=0,  description="MTA tax", examples=[0.5])
    tolls_amount: float         = Field(default=0.0, ge=0,  description="Tolls", examples=[0.0])
    improvement_surcharge: float= Field(default=0.3, ge=0,  description="Improvement surcharge", examples=[0.3])
    total_amount: float         = Field(..., ge=0,          description="Total charge excl. tip", examples=[15.80])
    congestion_surcharge: float = Field(default=2.5, ge=0,  description="Congestion surcharge", examples=[2.5])
    Airport_fee: float          = Field(default=0.0, ge=0,  description="Airport fee", examples=[0.0])
 
    # Time features
    trip_duration_mins: float   = Field(..., gt=0,          description="Trip duration in minutes", examples=[18.0])
    pickup_hour: int            = Field(..., ge=0, le=23,   description="Pickup hour 24 hour format (0-23)", examples=[14])
    pickup_day_of_week: int     = Field(..., ge=0, le=6,    description="0=Monday ... 6=Sunday", examples=[2])


#Model for validating incoming batch of trip features for prediction
class BatchTripFeatures(BaseModel):
    records: List[TripFeatures] = Field(...,min_length = 1, max_length=100, description="List of 1–100 trip records")


#--------Response Models---------------------------

#Health check response
class HealthResponse(BaseModel):
    status: str
    model_loaded: str
    model_version: str

#Model info response
class InfoResponse(BaseModel):
    model_name: str
    model_version: str 
    feature_names: List[str]
    training_metrics: dict

#Model for taxi tip prediction response
class PredictionResponse(BaseModel):
    predicted_tip_amount: float
    prediction_id: str
    model_version: str

#Model for taxi tip prediction batch response
class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int
    processing_time_ms: float

#-----------------------------------
# API Endpoints
#-----------------------------------
@app.get("/")
def root():
    return {
        "message": "Welcome to the Taxi Tip Prediction API",
        "description": "Use the /predict endpoint to estimate the tip amount for a taxi trip based on trip features.",
        "endpoints": {
            "/health": "GET: Check the health status of the API and model",
            "/info": "GET: Get information about the loaded model",
            "/predict": "POST: Predict the tip amount for a single taxi trip",
            "/predict/batch ": "POST Predict the tip amounts for multiple taxi trips"
        },
    }

@app.get("/health")
def health_check():
    return HealthResponse(
        status = 'healthy',
        model_loaded = 'up' if tip_predictor_model_state is not None else 'unavailable',
        model_version=tip_predictor_model_state['version'] if tip_predictor_model_state else "unavailable"
    )

@app.get("/info")
def model_info():
    if tip_predictor_model_state is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return InfoResponse(
        model_name = tip_predictor_model_state['model_name'],
        model_version = tip_predictor_model_state['version'],
        feature_names = tip_predictor_model_state['features'],
        training_metrics = tip_predictor_model_state['metrics']
    )

@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: TripFeatures):
    #Check to see if the model was loaded
    if tip_predictor_model_state is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    raw_df = build_raw_row(input_data) #Convert input into a pandas dataframe with 1 row
    X_tensor = preprocess_and_tensorize(raw_df) #Convert into tensor for predictions
 
    #Zero gradients so they aren't stored
    with torch.no_grad():
        prediction = tip_predictor_model_state['model'](X_tensor).item()

    return PredictionResponse(
        predicted_tip_amount=round(float(prediction), 2),
        prediction_id=str(uuid.uuid4()),
        model_version=tip_predictor_model_state['version'],
    )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(input_data: BatchTripFeatures):
    start_time = time.time() #Track time taken for batch

    #Check to see if the model was loaded
    if tip_predictor_model_state is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    #Build dataframe of all inputs for the neural network to predict
    raw_df = pd.concat([build_raw_row(r) for r in input_data.records], ignore_index=True)

    X_tensor = preprocess_and_tensorize(raw_df) #Convert the dataframe into a tensor
 
    #Zero gradients so they aren't stored
    with torch.no_grad():
        predictions = tip_predictor_model_state['model'](X_tensor).squeeze().tolist()
 
    #Convert to list if only one input was given to avoid errors
    if not isinstance(predictions, list):
        predictions = [predictions]
    
    #Create the predictions list with each individual prediction response
    predictions_list = [
        PredictionResponse(
            predicted_tip_amount=round(float(val), 2), #Rounds the prediction to 2 decimal places
            prediction_id=str(uuid.uuid4()), #Unique id
            model_version=tip_predictor_model_state['version'],
        )
        for val in predictions
    ]

    elapsed_time = (time.time() - start_time) * 1000 #Get the elapsed time since the start in ms

    return BatchPredictionResponse(
        predictions=predictions_list,
        count=len(predictions_list),
        processing_time_ms=round(elapsed_time, 2),
    )

#----------------------------------
# Global Exception Handdler
#----------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again.",
        },
    )

