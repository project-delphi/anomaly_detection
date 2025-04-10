from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class AnomalyDetector(ABC):
    """Base class for anomaly detection models."""

    def __init__(self, **kwargs):
        self.model = None
        self.threshold = None
        self.parameters = kwargs

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """Fit the model to the training data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores for the input data."""
        pass

    def is_anomaly(self, X: np.ndarray) -> np.ndarray:
        """Determine if samples are anomalies based on the threshold."""
        scores = self.predict(X)
        return scores > self.threshold

    def set_threshold(self, threshold: float) -> None:
        """Set the anomaly detection threshold."""
        self.threshold = threshold

    def get_parameters(self) -> Dict[str, Any]:
        """Get the model parameters."""
        return self.parameters
