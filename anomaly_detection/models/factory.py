from typing import Dict, Type
from .base import AnomalyDetector
from .isolation_forest import IsolationForestDetector
from .random_cut_forest import RandomCutForestDetector

class ModelFactory:
    """Factory class for creating anomaly detection models."""
    
    _models: Dict[str, Type[AnomalyDetector]] = {
        "isolation_forest": IsolationForestDetector,
        "random_cut_forest": RandomCutForestDetector
    }
    
    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> AnomalyDetector:
        """
        Create an anomaly detection model.
        
        Args:
            model_type: Type of model to create
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            An instance of the requested anomaly detection model
            
        Raises:
            ValueError: If the requested model type is not supported
        """
        if model_type not in cls._models:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return cls._models[model_type](**kwargs)
    
    @classmethod
    def get_supported_models(cls) -> list:
        """Get list of supported model types."""
        return list(cls._models.keys()) 