"""Setup script for Time Series Analysis project."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "torch>=1.12.0",
        "yfinance>=0.1.70",
        "plotly>=5.0.0",
        "streamlit>=1.20.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0"
    ]

setup(
    name="timeseries-gru-analysis",
    version="1.0.0",
    author="Time Series Analysis Project",
    author_email="your.email@example.com",
    description="A comprehensive time series analysis project with GRU neural networks and modern forecasting methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/timeseries-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "full": [
            "statsmodels>=0.13.0",
            "pmdarima>=2.0.0",
            "prophet>=1.1.0",
            "darts>=0.20.0",
            "sktime>=0.15.0",
            "tslearn>=0.5.0",
            "tensorboard>=2.9.0",
            "jupyter>=1.0.0",
            "ipywidgets>=7.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "timeseries-analysis=main:main",
            "timeseries-demo=demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "*.md", "*.txt"],
    },
    keywords=[
        "time-series",
        "forecasting",
        "gru",
        "lstm",
        "neural-networks",
        "anomaly-detection",
        "arima",
        "prophet",
        "machine-learning",
        "deep-learning",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/timeseries-analysis/issues",
        "Source": "https://github.com/yourusername/timeseries-analysis",
        "Documentation": "https://github.com/yourusername/timeseries-analysis/blob/main/README.md",
    },
)
