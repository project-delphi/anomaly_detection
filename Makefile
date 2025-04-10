.PHONY: install test lint format check-format check-imports clean

install:
	pip install -e .
	pip install pytest pytest-cov black isort

test:
	pytest tests/ --cov=anomaly_detection --cov-report=term-missing -v

lint:
	black --check .
	isort --check-only .

format:
	black .
	isort .

check-format:
	black --check .

check-imports:
	isort --check-only .

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".coverage" -exec rm -r {} +
	find . -type d -name "htmlcov" -exec rm -r {} +
	find . -type f -name "coverage.xml" -delete 