import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd


class NABLoader:
    """Loader for NAB (Numenta Anomaly Benchmark) datasets."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir

    def load_dataset(
        self, dataset_name: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load a NAB dataset and return features and labels.

        Args:
            dataset_name: Name of the dataset to load

        Returns:
            Tuple containing:
            - features: numpy array of shape (n_samples, n_features)
            - labels: numpy array of shape (n_samples,) or None if no labels
        """
        # Construct path to dataset
        dataset_path = os.path.join(self.data_dir, dataset_name)

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset {dataset_name} not found in {self.data_dir}"
            )

        # Load data
        df = pd.read_csv(dataset_path)

        # Extract features (assuming first column is timestamp)
        features = df.iloc[:, 1:].values

        # Check if labels exist (they might be in a separate file)
        labels_path = os.path.join(
            self.data_dir, "labels", f"{dataset_name}_labels.csv"
        )
        labels = None

        if os.path.exists(labels_path):
            labels_df = pd.read_csv(labels_path)
            labels = labels_df.iloc[:, 1].values

        return features, labels

    def get_available_datasets(self) -> list:
        """Get list of available datasets in the data directory."""
        return [f for f in os.listdir(self.data_dir) if f.endswith(".csv")]
