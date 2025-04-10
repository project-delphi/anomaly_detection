import numpy as np
import pytest
from anomaly_detection.models.factory import ModelFactory

def test_isolation_forest():
    # Create synthetic data
    np.random.seed(42)
    X_normal = np.random.randn(100, 2)
    X_anomaly = np.random.randn(10, 2) + 5  # Shifted distribution
    
    # Create and train model
    model = ModelFactory.create_model("isolation_forest", contamination=0.1)
    model.fit(X_normal)
    
    # Test predictions
    scores_normal = model.predict(X_normal)
    scores_anomaly = model.predict(X_anomaly)
    
    # Check that anomaly scores are higher than normal scores
    assert np.mean(scores_anomaly) > np.mean(scores_normal)
    
    # Check that threshold is set
    assert model.threshold is not None

def test_model_factory():
    # Test supported models
    supported_models = ModelFactory.get_supported_models()
    assert "isolation_forest" in supported_models
    assert "random_cut_forest" in supported_models
    
    # Test creating unsupported model
    with pytest.raises(ValueError):
        ModelFactory.create_model("unsupported_model")
    
    # Test creating models with parameters
    model = ModelFactory.create_model("isolation_forest", contamination=0.2)
    assert model.contamination == 0.2 