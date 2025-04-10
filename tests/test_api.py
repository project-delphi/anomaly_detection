import os

import numpy as np
import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

from anomaly_detection.api.main import app

# Load environment variables for testing
load_dotenv()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def synthetic_data():
    # Create a small synthetic dataset for testing
    np.random.seed(42)
    n_samples = 100
    timestamp = np.arange(n_samples)
    normal_data = np.sin(np.linspace(0, 2 * np.pi, n_samples)) + np.random.normal(
        0, 0.1, n_samples
    )

    # Insert an anomaly
    normal_data[50] += 5

    return {"timestamp": timestamp.tolist(), "value": normal_data.tolist()}


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_train_model_isolation_forest(client, synthetic_data):
    """Test training an Isolation Forest model."""
    # Save synthetic data to a temporary file
    import pandas as pd

    df = pd.DataFrame(synthetic_data)
    test_data_path = "data/test_data.csv"
    df.to_csv(test_data_path, index=False)

    try:
        # Train model
        response = client.post(
            "/train",
            json={
                "algorithm": "isolation_forest",
                "data_path": "test_data.csv",
                "parameters": {"contamination": 0.1},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "model_id" in data
        assert data["model_id"].startswith("isolation_forest_")
        assert "message" in data

        # Clean up
        os.remove(test_data_path)
    except Exception as e:
        # Ensure cleanup even if test fails
        if os.path.exists(test_data_path):
            os.remove(test_data_path)
        raise e


def test_train_model_invalid_algorithm(client):
    """Test training with an invalid algorithm."""
    response = client.post(
        "/train", json={"algorithm": "invalid_algorithm", "data_path": "test_data.csv"}
    )
    assert response.status_code == 500


def test_predict_without_training(client):
    """Test prediction without training a model first."""
    response = client.post(
        "/predict",
        json={
            "algorithm": "isolation_forest",
            "model_id": "nonexistent_model",
            "data": [1.0, 2.0, 3.0],
        },
    )
    assert response.status_code == 404


def test_train_and_predict_workflow(client, synthetic_data):
    """Test the complete workflow of training and prediction."""
    # Save synthetic data
    import pandas as pd

    df = pd.DataFrame(synthetic_data)
    test_data_path = "data/test_data.csv"
    df.to_csv(test_data_path, index=False)

    try:
        # Train model
        train_response = client.post(
            "/train",
            json={
                "algorithm": "isolation_forest",
                "data_path": "test_data.csv",
                "parameters": {"contamination": 0.1},
            },
        )
        assert train_response.status_code == 200
        model_id = train_response.json()["model_id"]

        # Make prediction with normal data
        normal_data = [0.0, 0.1, 0.2]
        predict_response = client.post(
            "/predict",
            json={
                "algorithm": "isolation_forest",
                "model_id": model_id,
                "data": normal_data,
            },
        )
        assert predict_response.status_code == 200
        prediction = predict_response.json()
        assert "is_anomaly" in prediction
        assert "score" in prediction
        assert "threshold" in prediction
        assert isinstance(prediction["is_anomaly"], bool)
        assert isinstance(prediction["score"], float)
        assert isinstance(prediction["threshold"], float)

        # Make prediction with anomalous data
        anomalous_data = [0.0, 5.0, 0.2]  # Large spike in the middle
        predict_response = client.post(
            "/predict",
            json={
                "algorithm": "isolation_forest",
                "model_id": model_id,
                "data": anomalous_data,
            },
        )
        assert predict_response.status_code == 200
        prediction = predict_response.json()
        assert prediction["is_anomaly"] is True
        assert prediction["score"] > prediction["threshold"]

        # Clean up
        os.remove(test_data_path)
    except Exception as e:
        # Ensure cleanup even if test fails
        if os.path.exists(test_data_path):
            os.remove(test_data_path)
        raise e


def test_invalid_prediction_data(client):
    """Test prediction with invalid data format."""
    # Train a model first
    train_response = client.post(
        "/train",
        json={
            "algorithm": "isolation_forest",
            "data_path": "synthetic_data.csv",
            "parameters": {"contamination": 0.1},
        },
    )
    model_id = train_response.json()["model_id"]

    # Test with empty data
    response = client.post(
        "/predict",
        json={"algorithm": "isolation_forest", "model_id": model_id, "data": []},
    )
    assert response.status_code == 500

    # Test with non-numeric data
    response = client.post(
        "/predict",
        json={
            "algorithm": "isolation_forest",
            "model_id": model_id,
            "data": ["not", "a", "number"],
        },
    )
    assert response.status_code == 500
