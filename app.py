#This file contains all API endpoints for the application

#All imports below
import sys
import uuid

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field

import numpy as np

import torch
import torch.nn as nn

import joblib
import time

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
        
        # Output layer (single neuron for binary classification) 
        layers.append(nn.Linear(prev_size, 1)) 
        
        #Set layers which we dynamically built
        self.network = nn.Sequential(*layers) 

    def forward(self, x):
        return self.network(x).squeeze()

#----------------------------------
# Loading Model On Startup
#----------------------------------
tip_predictor_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    #Loading the neural network
    try:
        checkpoint = torch.load('models/nn_regressor.pth')

        # Recreate architecture using saved config
        tip_predictor_model = RegressionNeuralNetwork(checkpoint['input_size'])
        tip_predictor_model.load_state_dict(checkpoint['model_state_dict'])
        tip_predictor_model.eval() #Switch the model to evaluation mode
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

#Model for validating incoming trip features for prediction
class TripFeatures(BaseModel):
    trip_distance: float = Field(..., gt=0, description="Trip distance in miles", examples=[3.5, 0.2],)
    trip_speed: float = Field(..., gt=0, le=80, description="Trip speed in mph", examples =[20, 20.5])
    trip_duration_mins: float = Field(...,gt=0, description = "Trip duration in minutes", examples = [5, 24.5])
    pickup_hour: int = Field(..., ge=0, le=23, description="Hour of pickup in 24 hour format (0-23)",examples=[14, 1],)
    fare_amount: float = Field( ..., ge=0, le=500, description="Base fare in USD", examples=[12.50],)
    passenger_count: int = Field(..., ge=1, le=5, description="Number of passengers (5)", examples=[2],)
    payment_type: int = Field(..., ge=1, le=6, description="Payment type code (1=Credit card, 2=Cash, 3=No Charge, 4=Dispute, 0=Other)",examples=[1],)


#Model for taxi tip prediction response
class PredictionResponse(BaseModel):
    predicted_tip: float
    prediction_id: str
    model_version: str

#-----------------------------------
# API Endpoints
#-----------------------------------
@app.get("/")
def root():
    return {
        "message": "Welcome to the Taxi Tip Prediction API",
        "description": "Use the /predict endpoint to estimate the tip amount for a taxi trip based on trip features.",
        "endpoint": "/predict",
        "method": "POST",
        "example_request_body": {
            "trip_distance": 3.5,
            "trip_speed": 20.0,
            "trip_duration_mins": 10.0,
            "pickup_hour": 14,
            "fare_amount": 12.5,
            "passenger_count": 2,
            "payment_type": 1
        }
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: TripFeatures):

    #Check to see if the model was loaded
    if tip_predictor_model is None:
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
        preds = tip_predictor_model(X_tensor).item()

    return PredictionResponse(
        prediction=round(float(preds), 4),
        prediction_id=str(uuid.uuid4()),
        model_version="1.0.0",
    )