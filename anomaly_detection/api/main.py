import os
from typing import Any, List, Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError, field_validator, validator

from anomaly_detection.data.nab_loader import NABLoader
from anomaly_detection.models.factory import ModelFactory

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Anomaly Detection Service",
    description="A service for detecting anomalies in time series data",
    version="1.0.0",
)

# Store trained models in memory (in production, use a proper database)
trained_models = {}


class TrainingRequest(BaseModel):
    algorithm: str
    data_path: str
    parameters: Optional[dict] = None


class PredictionRequest(BaseModel):
    algorithm: str
    data: List[Any]  # Accept any type of data and validate in the endpoint
    model_id: Optional[str] = None


class PredictionResponse(BaseModel):
    is_anomaly: bool
    score: float
    threshold: float


@app.post("/train")
async def train_model(request: TrainingRequest):
    """
    Train an anomaly detection model using the specified algorithm and data.

    Args:
        request: TrainingRequest containing:
            - algorithm: The algorithm to use (isolation_forest or random_cut_forest)
            - data_path: Path to the training data
            - parameters: Optional parameters for the model

    Returns:
        dict: Status and model ID
    """
    try:
        # Load data
        loader = NABLoader()
        X, _ = loader.load_dataset(request.data_path)

        # Create model with parameters
        model_params = request.parameters or {}
        if request.algorithm == "random_cut_forest":
            # Add AWS credentials if using RCF
            model_params.update(
                {
                    "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
                    "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
                    "region_name": os.getenv("AWS_REGION", "us-west-2"),
                }
            )

        model = ModelFactory.create_model(request.algorithm, **model_params)

        # Train model
        model.fit(X)

        # Generate unique model ID
        model_id = f"{request.algorithm}_{len(trained_models)}"

        # Store model
        trained_models[model_id] = model

        return {
            "status": "success",
            "model_id": model_id,
            "message": f"Model trained successfully with {len(X)} samples",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Make predictions using a trained anomaly detection model.

    Args:
        request: PredictionRequest containing:
            - algorithm: The algorithm used
            - data: List of values to predict
            - model_id: ID of the trained model to use

    Returns:
        PredictionResponse: Prediction results including anomaly status and score
    """
    # Validate input data
    if not request.data:
        raise HTTPException(status_code=500, detail="Empty data provided")

    try:
        # Try converting all values to float
        data_values = [float(x) for x in request.data]
    except (ValueError, TypeError):
        raise HTTPException(status_code=500, detail="All values must be numeric")

    # Get model first to fail fast if model doesn't exist
    if request.model_id not in trained_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model {request.model_id} not found. Please train a model first.",
        )

    try:
        # Convert input data to numpy array and reshape to 2D
        data = np.array(data_values, dtype=float)
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)  # Convert to 2D array with shape (n_samples, 1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

    try:
        model = trained_models[request.model_id]

        # Make prediction
        score = float(model.predict(data)[0])  # Convert to float for JSON serialization
        threshold = float(model.threshold)  # Convert to float for JSON serialization
        is_anomaly = score > threshold

        return PredictionResponse(
            is_anomaly=bool(is_anomaly), score=score, threshold=threshold
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors with 500 status code."""
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}
