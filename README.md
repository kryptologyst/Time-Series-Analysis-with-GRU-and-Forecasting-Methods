# Time Series Analysis with GRU and Forecasting Methods

A comprehensive Python project for time series analysis featuring GRU neural networks, multiple forecasting methods, anomaly detection, and interactive visualization.

## Features

### Core Functionality
- **GRU Neural Networks**: Enhanced GRU models with dropout, batch normalization, and bidirectional support
- **Multiple Forecasting Methods**: ARIMA, Auto-ARIMA, Prophet, and naive forecasting methods
- **Anomaly Detection**: Isolation Forest, statistical methods (Z-score, IQR), and autoencoder-based detection
- **Interactive Visualization**: Streamlit web interface with Plotly charts
- **Configuration Management**: YAML-based configuration system
- **Comprehensive Testing**: Unit tests for all components

### Advanced Features
- **Synthetic Data Generation**: Realistic time series data for testing and development
- **Model Persistence**: Save and load trained models
- **Performance Metrics**: MSE, MAE, R² for model evaluation
- **Early Stopping**: Prevent overfitting with configurable patience
- **Gradient Clipping**: Stable training for deep networks
- **Learning Rate Scheduling**: Adaptive learning rate adjustment

## Project Structure

```
0285_GRU_for_sequence_modeling/
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── data_preprocessing.py      # Data loading and preprocessing
│   ├── gru_model.py              # GRU model and trainer
│   ├── forecasting_methods.py    # Multiple forecasting methods
│   ├── anomaly_detection.py      # Anomaly detection algorithms
│   └── visualization.py          # Plotting and visualization utilities
├── config/                       # Configuration files
│   └── config.yaml              # Main configuration
├── tests/                        # Unit tests
│   └── test_timeseries.py       # Comprehensive test suite
├── notebooks/                    # Jupyter notebooks (optional)
├── data/                         # Data storage
├── models/                       # Saved models
├── plots/                        # Generated plots
├── results/                      # Analysis results
├── logs/                         # Log files
├── main.py                      # Main application script
├── streamlit_app.py             # Streamlit web interface
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Time-Series-Analysis-with-GRU-and-Forecasting-Methods.git
   cd Time-Series-Analysis-with-GRU-and-Forecasting-Methods
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Optional Dependencies

For enhanced functionality, you can install additional packages:

```bash
# Advanced time series forecasting
pip install darts sktime tslearn

# Training visualization
pip install tensorboard

# Jupyter notebook support
pip install jupyter ipywidgets
```

## Quick Start

### Command Line Interface

Run the complete analysis pipeline:

```bash
python main.py
```

This will:
1. Load data (real stock data or synthetic)
2. Train GRU model
3. Run multiple forecasting methods
4. Perform anomaly detection
5. Generate visualizations
6. Save results

### Streamlit Web Interface

Launch the interactive web dashboard:

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

### Configuration

Modify `config/config.yaml` to customize:

- **Data source**: Real stock data or synthetic data
- **Model parameters**: GRU architecture, training settings
- **Forecasting methods**: Enable/disable specific methods
- **Anomaly detection**: Configure detection algorithms
- **Visualization**: Plot settings and output formats

## Usage Examples

### Basic GRU Training

```python
from src.data_preprocessing import TimeSeriesPreprocessor, generate_synthetic_data
from src.gru_model import GRUModel, GRUTrainer
import torch
from torch.utils.data import DataLoader, TensorDataset

# Generate synthetic data
data = generate_synthetic_data(n_samples=1000, random_state=42)
values = data['Close'].values

# Preprocess data
preprocessor = TimeSeriesPreprocessor()
scaled_data = preprocessor.normalize_data(values.reshape(-1, 1))
X, y = preprocessor.create_sequences(scaled_data, sequence_length=30)
X_train, X_test, y_train, y_test = preprocessor.train_test_split_sequences(X, y)

# Create and train GRU model
model = GRUModel(input_size=1, hidden_size=50, num_layers=2)
trainer = GRUTrainer(model, learning_rate=0.001)

# Prepare data loaders
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Train model
history = trainer.train(train_loader, epochs=100)

# Make predictions
predictions = trainer.predict(train_loader)
```

### Forecasting Comparison

```python
from src.forecasting_methods import ForecastingMethods

# Initialize forecaster
forecaster = ForecastingMethods()

# Run multiple forecasting methods
arima_result = forecaster.arima_forecast(values, forecast_steps=30)
auto_arima_result = forecaster.auto_arima_forecast(values, forecast_steps=30)
naive_result = forecaster.naive_forecast(values, method='last')

# Compare methods
comparison_df = forecaster.compare_methods(values, forecast_steps=30)
print(comparison_df)
```

### Anomaly Detection

```python
from src.anomaly_detection import AnomalyDetector, AutoencoderAnomalyDetector

# Statistical methods
detector = AnomalyDetector()
iso_result = detector.isolation_forest_detection(values)
stat_result = detector.statistical_detection(values, method='zscore')

# Deep learning method
autoencoder = AutoencoderAnomalyDetector(input_dim=1)
autoencoder.fit(values, epochs=100)
ae_result = autoencoder.detect_anomalies(values)
```

### Visualization

```python
from src.visualization import TimeSeriesVisualizer

# Create visualizer
viz = TimeSeriesVisualizer()

# Plot time series
viz.plot_time_series(values, title="Stock Price Data")

# Plot forecasts
forecasts = {
    'ARIMA': arima_result['predictions'],
    'GRU': gru_predictions,
    'Naive': naive_result['predictions']
}
viz.plot_forecast_comparison(values, forecasts)

# Plot anomalies
anomalies = {
    'Isolation Forest': iso_result['anomalies'],
    'Z-Score': stat_result['anomalies']
}
viz.plot_anomalies(values, anomalies)
```

## Configuration Options

### Data Configuration
```yaml
data:
  symbol: "NFLX"              # Stock symbol
  start_date: "2020-01-01"    # Start date
  end_date: "2023-01-01"      # End date
  column: "Close"             # Column to analyze
  use_synthetic: false        # Use synthetic data
```

### GRU Model Configuration
```yaml
gru_model:
  input_size: 1               # Input features
  hidden_size: 50             # Hidden units
  num_layers: 2               # Number of layers
  dropout: 0.2                # Dropout rate
  bidirectional: false        # Bidirectional GRU
  learning_rate: 0.001         # Learning rate
  epochs: 100                  # Training epochs
  batch_size: 32               # Batch size
```

### Forecasting Configuration
```yaml
forecasting:
  methods: ["ARIMA", "Auto-ARIMA", "Prophet", "Naive-last"]
  forecast_steps: 30
  arima:
    order: [1, 1, 1]
  prophet:
    seasonality_mode: "additive"
```

## Testing

Run the comprehensive test suite:

```bash
python -m pytest tests/ -v
```

Or run specific test modules:

```bash
python tests/test_timeseries.py
```

The test suite covers:
- Data preprocessing functionality
- GRU model training and inference
- Forecasting method accuracy
- Anomaly detection algorithms
- Integration testing

## Performance Optimization

### GPU Acceleration
The project automatically detects and uses GPU if available:

```python
# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

### Memory Optimization
- Use appropriate batch sizes for your hardware
- Enable gradient checkpointing for large models
- Use mixed precision training for faster training

### Parallel Processing
- Configure number of workers for data loading
- Use multiprocessing for hyperparameter tuning
- Enable parallel forecasting method execution

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **CUDA Out of Memory**: Reduce batch size or model size
   ```yaml
   gru_model:
     batch_size: 16  # Reduce from 32
     hidden_size: 25  # Reduce from 50
   ```

3. **Data Loading Issues**: Check internet connection for Yahoo Finance
   ```yaml
   data:
     use_synthetic: true  # Use synthetic data instead
   ```

4. **Prophet Installation**: On some systems, Prophet requires additional setup
   ```bash
   pip install prophet --no-deps
   pip install cmdstanpy
   ```

### Performance Issues

- **Slow Training**: Reduce model complexity or use GPU
- **Memory Issues**: Use smaller batch sizes or sequence lengths
- **Convergence Problems**: Adjust learning rate or add regularization

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `python -m pytest tests/`
5. Commit changes: `git commit -m "Add feature"`
6. Push to branch: `git push origin feature-name`
7. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **PyTorch**: Deep learning framework
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualization
- **Statsmodels**: Statistical modeling
- **Prophet**: Facebook's forecasting tool
- **Scikit-learn**: Machine learning utilities

## Citation

If you use this project in your research, please cite:

```bibtex
@software{timeseries_gru_analysis,
  title={Time Series Analysis with GRU and Modern Forecasting Methods},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/timeseries-analysis}
}
```

## Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the documentation
- Review the test cases for usage examples

---

**Note**: This project is for educational and research purposes. Always validate results and consider the limitations of time series forecasting in real-world applications.
# Time-Series-Analysis-with-GRU-and-Forecasting-Methods
