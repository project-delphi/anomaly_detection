import numpy as np
from sklearn.ensemble import IsolationForest

from .base import AnomalyDetector


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest based anomaly detector."""

    def __init__(self, contamination: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination, random_state=42, **kwargs
        )

    def fit(self, X: np.ndarray) -> None:
        """Fit the Isolation Forest model."""
        self.model.fit(X)
        # Set threshold based on contamination
        scores = self.model.score_samples(X)
        self.threshold = np.percentile(scores, 100 * self.contamination)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores."""
        return -self.model.score_samples(X)  # Convert to positive scores
