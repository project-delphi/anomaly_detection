from setuptools import setup, find_packages

setup(
    name="anomaly_detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "boto3>=1.26.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-dotenv>=0.19.0",
        "pytest>=6.2.5",
        "pytest-cov>=2.12.0",
    ],
    python_requires=">=3.8",
) 