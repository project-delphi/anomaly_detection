# Anomaly Detection Service

A Python-based service for detecting anomalies in time series data using Random Cut Forest and Isolation Forest algorithms.

## Features

- REST API endpoints for training and inference
- Support for multiple anomaly detection algorithms:
  - Random Cut Forest (AWS)
  - Isolation Forest
- Integration with NAB (Numenta Anomaly Benchmark) datasets
- Easy-to-use training and prediction interfaces

## Project Structure

```bash
anomaly_detection/
├── api/           # FastAPI endpoints
├── core/          # Core anomaly detection logic
├── data/          # Data processing and loading
├── models/        # Model implementations
├── utils/         # Utility functions
└── tests/         # Test suite
```

## Setup

1. Create a virtual environment:

```bash
python -m venv ./.venv
source ./.venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your AWS credentials if using RCF
```

## Usage

1. Start the API server:

```bash
uvicorn anomaly_detection.api.main:app --reload
```

2. Train a model:

```bash
curl -X POST "http://localhost:8000/train" -H "Content-Type: application/json" -d '{"algorithm": "isolation_forest", "data_path": "path/to/data.csv"}'
```

3. Make predictions:

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"algorithm": "isolation_forest", "data": [1.2, 2.3, 3.4]}'
```

## Testing

Run tests with coverage:

```bash
pytest --cov=anomaly_detection tests/
```

## License

MIT
