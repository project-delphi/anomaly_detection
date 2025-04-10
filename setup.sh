#!/bin/bash

# Exit on error
set -e

echo "Setting up anomaly detection environment..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install the package in development mode
echo "Installing package in development mode..."
pip install -e .

# Create data directory if it doesn't exist
echo "Creating data directory..."
mkdir -p data

# Create synthetic dataset for testing
echo "Creating synthetic dataset..."
python -c "
import numpy as np
import pandas as pd

# Generate synthetic time series data
np.random.seed(42)
n_samples = 1000
timestamp = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
normal_data = np.sin(np.linspace(0, 10*np.pi, n_samples)) + np.random.normal(0, 0.1, n_samples)

# Insert anomalies
anomaly_indices = [100, 300, 600, 800]
normal_data[anomaly_indices] += 5

# Create DataFrame
df = pd.DataFrame({
    'timestamp': timestamp,
    'value': normal_data
})

# Save to CSV
df.to_csv('data/synthetic_data.csv', index=False)
print('Created synthetic dataset with', len(anomaly_indices), 'anomalies')
"

# Create .env file if it doesn't exist
echo "Setting up environment variables..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env file. Please edit it with your AWS credentials if using Random Cut Forest."
fi

echo "Setup completed successfully!"
echo "To run tests: source .venv/bin/activate && pytest tests/ -v"
echo "To start the API server: source .venv/bin/activate && uvicorn anomaly_detection.api.main:app --reload"
