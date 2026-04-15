#This file contains all API endpoints for the application

#All imports below
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

#----------------------------------
# Neural Network Class Definition
#----------------------------------
class RegressionNeuralNetwork(nn.Module):
    def __init__(self,  input_size, hidden_sizes=[128, 64],  dropout_rate=0.3 ):
        super().__init__()

        layers = []
        prev_size = input_size 

        #Adding layers to network dynamically
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU()) 
            layers.append(nn.Dropout(dropout_rate)) 
            prev_size = hidden_size 
        
        # Output layer (single neuron for regression) 
        layers.append(nn.Linear(prev_size, 1)) 
        
        #Set layers which we dynamically built
        self.network = nn.Sequential(*layers) 

    def forward(self, x):
        return self.network(x).squeeze()

#----------------------------------
# Loading Model On Startup
#----------------------------------
tip_predictor_model_state = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tip_predictor_model_state
    print("Loading model...")
    #Loading the neural network from mlflow
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        client = MlflowClient()

        versions = client.get_latest_versions("taxi-tip-regressor", stages=["None"])
        latest = versions[0]

        best_model = mlflow.pytorch.load_model(f"models:/taxi-tip-regressor/{latest.version}") #Loads the latest version of the model registered
        best_model.eval() #Set to eval mode

        run = client.get_run(latest.run_id)

        tip_predictor_model_state = {
            'model': best_model,
            'version': str(latest.version),
            'model_name': 'taxi-tip-regressor',
            'features': [
                "trip_distance",
                "trip_speed",
                "trip_duration_mins",
                "pickup_hour",
                "fare_amount",
                "passenger_count",
                "payment_type"
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

#-----------------------------------
# API Response Models
#-----------------------------------

#--------Input Validation Models------------------

#Model for validating incoming trip features for prediction
class TripFeatures(BaseModel):
    trip_distance: float = Field(..., gt=0, description="Trip distance in miles", examples=[3.5, 0.2])
    trip_speed: float = Field(..., gt=0, le=80, description="Trip speed in mph", examples =[20, 20.5])
    trip_duration_mins: float = Field(...,gt=0, description = "Trip duration in minutes", examples = [5, 24.5])
    pickup_hour: int = Field(..., ge=0, le=23, description="Hour of pickup in 24 hour format (0-23)",examples=[14, 1])
    fare_amount: float = Field( ..., ge=0, le=500, description="Base fare in USD", examples=[12.50])
    passenger_count: int = Field(..., ge=1, le=5, description="Number of passengers (5)", examples=[2])
    payment_type: int = Field(..., ge=1, le=6, description="Payment type code (1=Credit card, 2=Cash, 3=No Charge, 4=Dispute, 0=Other)",examples=[1])

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
    
    #Extracting trip features from input data
    trip_features = [[
        input_data.trip_distance,
        input_data.trip_speed,
        input_data.trip_duration_mins,
        input_data.pickup_hour,
        input_data.fare_amount,
        input_data.passenger_count,
        input_data.payment_type,
    ]]

    X_tensor = torch.tensor(trip_features, dtype=torch.float32) #Convert to tensor for predictions

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

    records = input_data.records
    
    #Below we convert the 2d trip features array into a numpy array for easier calculations 
    trip_features = np.array([[
        r.trip_distance, r.trip_speed, r.trip_duration_mins,
        r.pickup_hour, r.fare_amount, r.passenger_count, r.payment_type
    ] for r in records], dtype=np.float32)

    X_tensor = torch.tensor(trip_features) #Convert to tensor for predictions

    #Evaluate all predictions in the X tensor without storing gradients
    with torch.no_grad():
        predictions = tip_predictor_model_state['model'](X_tensor)
    
    #Create the predictions list with each individual prediction response
    predictions_list = [
        PredictionResponse(
            predicted_tip_amount=round(float(val), 2), #Rounds the prediction to 2 decimal places
            prediction_id=str(uuid.uuid4()), #Unique id
            model_version=tip_predictor_model_state['version'],
        )
        for val in predictions.squeeze().tolist() 
    ]

    elapsed_time = (time.time() - start_time) * 1000 #Get the elapsed time since the start in ms

    return BatchPredictionResponse(
        predictions=predictions_list,
        count=len(predictions_list),
        processing_time_ms=round(elapsed_time, 2),
    )