"""Main application for Time Series Analysis with GRU and modern forecasting methods."""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_preprocessing import TimeSeriesPreprocessor, generate_synthetic_data
from gru_model import GRUModel, GRUTrainer
from forecasting_methods import ForecastingMethods
from anomaly_detection import AnomalyDetector, AutoencoderAnomalyDetector
from visualization import TimeSeriesVisualizer

# Setup logging
def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration."""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = log_config.get('file', 'logs/timeseries_analysis.log')
    
    # Create logs directory
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


class TimeSeriesAnalysisApp:
    """Main application class for time series analysis."""
    
    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """Initialize the application.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        setup_logging(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.preprocessor = TimeSeriesPreprocessor(
            scaler_type=self.config['preprocessing']['scaler_type']
        )
        self.forecaster = ForecastingMethods()
        self.anomaly_detector = AnomalyDetector()
        self.visualizer = TimeSeriesVisualizer(
            style=self.config['visualization']['style'],
            figsize=tuple(self.config['visualization']['figsize'])
        )
        
        # Create output directories
        self._create_directories()
        
        # Data storage
        self.data = None
        self.scaled_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Results storage
        self.gru_results = None
        self.forecast_results = {}
        self.anomaly_results = {}
        
        self.logger.info("Time Series Analysis Application initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found. Using default configuration.")
            return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'data': {
                'symbol': 'NFLX',
                'start_date': '2020-01-01',
                'end_date': '2023-01-01',
                'column': 'Close',
                'use_synthetic': False
            },
            'preprocessing': {
                'scaler_type': 'minmax',
                'sequence_length': 30,
                'test_size': 0.2,
                'random_state': 42
            },
            'gru_model': {
                'input_size': 1,
                'hidden_size': 50,
                'num_layers': 2,
                'dropout': 0.2,
                'bidirectional': False,
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'epochs': 100,
                'batch_size': 32,
                'early_stopping_patience': 20
            },
            'forecasting': {
                'methods': ['ARIMA', 'Auto-ARIMA', 'Prophet', 'Naive-last'],
                'forecast_steps': 30
            },
            'anomaly_detection': {
                'methods': ['Isolation Forest', 'Statistical-zscore', 'Statistical-iqr']
            },
            'visualization': {
                'style': 'seaborn-v0_8',
                'figsize': [12, 8],
                'save_plots': True,
                'plot_format': 'png',
                'dpi': 300
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'logs/timeseries_analysis.log'
            },
            'output': {
                'models_dir': 'models',
                'plots_dir': 'plots',
                'results_dir': 'results',
                'logs_dir': 'logs'
            }
        }
    
    def _create_directories(self) -> None:
        """Create necessary output directories."""
        output_config = self.config['output']
        directories = [
            output_config['models_dir'],
            output_config['plots_dir'],
            output_config['results_dir'],
            output_config['logs_dir']
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def load_data(self) -> None:
        """Load and prepare data."""
        self.logger.info("Loading data...")
        
        data_config = self.config['data']
        
        if data_config.get('use_synthetic', False):
            # Generate synthetic data
            synthetic_config = data_config.get('synthetic', {})
            self.data = generate_synthetic_data(**synthetic_config)
            self.logger.info(f"Generated synthetic data with {len(self.data)} samples")
        else:
            # Load real stock data
            symbol = data_config['symbol']
            start_date = data_config['start_date']
            end_date = data_config['end_date']
            column = data_config['column']
            
            df = self.preprocessor.load_stock_data(symbol, start_date, end_date, column)
            self.data = df[column].values
            self.logger.info(f"Loaded {symbol} data from {start_date} to {end_date}")
        
        # Normalize data
        self.scaled_data = self.preprocessor.normalize_data(self.data.reshape(-1, 1))
        
        # Create sequences
        sequence_length = self.config['preprocessing']['sequence_length']
        X, y = self.preprocessor.create_sequences(self.scaled_data, sequence_length)
        
        # Split data
        test_size = self.config['preprocessing']['test_size']
        random_state = self.config['preprocessing']['random_state']
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocessor.train_test_split_sequences(
            X, y, test_size=test_size, random_state=random_state
        )
        
        self.logger.info(f"Data prepared: {len(self.X_train)} training samples, {len(self.X_test)} test samples")
    
    def train_gru_model(self) -> None:
        """Train the GRU model."""
        self.logger.info("Training GRU model...")
        
        gru_config = self.config['gru_model']
        
        # Create model
        model = GRUModel(
            input_size=gru_config['input_size'],
            hidden_size=gru_config['hidden_size'],
            num_layers=gru_config['num_layers'],
            dropout=gru_config['dropout'],
            bidirectional=gru_config['bidirectional']
        )
        
        # Create trainer
        trainer = GRUTrainer(
            model=model,
            learning_rate=gru_config['learning_rate'],
            weight_decay=gru_config['weight_decay']
        )
        
        # Prepare data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(self.X_train),
            torch.FloatTensor(self.y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(self.X_test),
            torch.FloatTensor(self.y_test)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=gru_config['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=gru_config['batch_size'],
            shuffle=False
        )
        
        # Train model
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=gru_config['epochs'],
            early_stopping_patience=gru_config['early_stopping_patience']
        )
        
        # Make predictions
        train_predictions = trainer.predict(train_loader)
        test_predictions = trainer.predict(val_loader)
        
        # Inverse transform predictions
        train_pred_scaled = np.concatenate([self.X_train[:, -1, :], train_predictions])
        test_pred_scaled = np.concatenate([self.X_test[:, -1, :], test_predictions])
        
        train_pred_original = self.preprocessor.inverse_transform(train_pred_scaled)
        test_pred_original = self.preprocessor.inverse_transform(test_pred_scaled)
        
        self.gru_results = {
            'model': model,
            'trainer': trainer,
            'train_predictions': train_pred_original,
            'test_predictions': test_pred_original,
            'history': history
        }
        
        # Save model
        model_path = os.path.join(self.config['output']['models_dir'], 'gru_model.pth')
        trainer.save_model(model_path)
        
        self.logger.info("GRU model training completed")
    
    def run_forecasting_methods(self) -> None:
        """Run multiple forecasting methods."""
        self.logger.info("Running forecasting methods...")
        
        forecast_config = self.config['forecasting']
        methods = forecast_config['methods']
        forecast_steps = forecast_config['forecast_steps']
        
        for method in methods:
            try:
                self.logger.info(f"Running {method} forecasting...")
                
                if method == 'ARIMA':
                    arima_config = forecast_config.get('arima', {})
                    result = self.forecaster.arima_forecast(
                        self.data,
                        order=tuple(arima_config.get('order', [1, 1, 1])),
                        forecast_steps=forecast_steps,
                        seasonal_order=arima_config.get('seasonal_order')
                    )
                
                elif method == 'Auto-ARIMA':
                    auto_arima_config = forecast_config.get('auto_arima', {})
                    result = self.forecaster.auto_arima_forecast(
                        self.data,
                        forecast_steps=forecast_steps,
                        seasonal=auto_arima_config.get('seasonal', True),
                        max_p=auto_arima_config.get('max_p', 5),
                        max_q=auto_arima_config.get('max_q', 5),
                        max_P=auto_arima_config.get('max_P', 2),
                        max_Q=auto_arima_config.get('max_Q', 2)
                    )
                
                elif method == 'Prophet':
                    # Convert to DataFrame for Prophet
                    dates = pd.date_range(start='2020-01-01', periods=len(self.data), freq='D')
                    df = pd.DataFrame({'ds': dates, 'y': self.data})
                    
                    prophet_config = forecast_config.get('prophet', {})
                    result = self.forecaster.prophet_forecast(
                        df,
                        forecast_steps=forecast_steps,
                        seasonality_mode=prophet_config.get('seasonality_mode', 'additive'),
                        yearly_seasonality=prophet_config.get('yearly_seasonality', True),
                        weekly_seasonality=prophet_config.get('weekly_seasonality', True),
                        daily_seasonality=prophet_config.get('daily_seasonality', False)
                    )
                
                elif method.startswith('Naive-'):
                    naive_method = method.split('-')[1].lower()
                    result = self.forecaster.naive_forecast(
                        self.data,
                        forecast_steps=forecast_steps,
                        method=naive_method
                    )
                
                else:
                    self.logger.warning(f"Unknown forecasting method: {method}")
                    continue
                
                self.forecast_results[method] = result
                self.logger.info(f"{method} forecasting completed")
                
            except Exception as e:
                self.logger.error(f"Error running {method}: {e}")
                continue
        
        self.logger.info("Forecasting methods completed")
    
    def run_anomaly_detection(self) -> None:
        """Run anomaly detection methods."""
        self.logger.info("Running anomaly detection methods...")
        
        anomaly_config = self.config['anomaly_detection']
        methods = anomaly_config['methods']
        
        for method in methods:
            try:
                self.logger.info(f"Running {method} anomaly detection...")
                
                if method == 'Isolation Forest':
                    iso_config = anomaly_config.get('isolation_forest', {})
                    result = self.anomaly_detector.isolation_forest_detection(
                        self.data,
                        contamination=iso_config.get('contamination', 0.1),
                        random_state=iso_config.get('random_state', 42)
                    )
                
                elif method.startswith('Statistical-'):
                    stat_method = method.split('-')[1].lower()
                    stat_config = anomaly_config.get('statistical', {})
                    result = self.anomaly_detector.statistical_detection(
                        self.data,
                        method=stat_method,
                        threshold=stat_config.get('threshold', 3.0),
                        window_size=stat_config.get('window_size')
                    )
                
                elif method == 'Autoencoder':
                    auto_config = anomaly_config.get('autoencoder', {})
                    detector = AutoencoderAnomalyDetector(
                        input_dim=1,
                        encoding_dim=auto_config.get('encoding_dim', 10),
                        hidden_dims=auto_config.get('hidden_dims', [64, 32])
                    )
                    
                    # Train autoencoder
                    detector.fit(
                        self.data,
                        epochs=auto_config.get('epochs', 100),
                        batch_size=auto_config.get('batch_size', 32),
                        learning_rate=auto_config.get('learning_rate', 0.001)
                    )
                    
                    # Detect anomalies
                    result = detector.detect_anomalies(
                        self.data,
                        threshold_percentile=auto_config.get('threshold_percentile', 95.0)
                    )
                
                else:
                    self.logger.warning(f"Unknown anomaly detection method: {method}")
                    continue
                
                self.anomaly_results[method] = result
                self.logger.info(f"{method} anomaly detection completed")
                
            except Exception as e:
                self.logger.error(f"Error running {method}: {e}")
                continue
        
        self.logger.info("Anomaly detection methods completed")
    
    def create_visualizations(self) -> None:
        """Create comprehensive visualizations."""
        self.logger.info("Creating visualizations...")
        
        plots_dir = self.config['output']['plots_dir']
        save_plots = self.config['visualization']['save_plots']
        
        # Plot original time series
        if save_plots:
            save_path = os.path.join(plots_dir, 'original_timeseries.png')
        else:
            save_path = None
        
        self.visualizer.plot_time_series(
            self.data,
            title="Original Time Series Data",
            save_path=save_path
        )
        
        # Plot GRU results
        if self.gru_results is not None:
            if save_plots:
                save_path = os.path.join(plots_dir, 'gru_forecast.png')
            else:
                save_path = None
            
            # Create predictions dictionary for visualization
            gru_predictions = {
                'GRU Train': self.gru_results['train_predictions'],
                'GRU Test': self.gru_results['test_predictions']
            }
            
            self.visualizer.plot_forecast_comparison(
                self.data,
                gru_predictions,
                title="GRU Model Forecast",
                save_path=save_path
            )
        
        # Plot forecasting comparison
        if self.forecast_results:
            if save_plots:
                save_path = os.path.join(plots_dir, 'forecast_comparison.png')
            else:
                save_path = None
            
            # Extract predictions
            forecast_predictions = {}
            for method, result in self.forecast_results.items():
                forecast_predictions[method] = result['predictions']
            
            self.visualizer.plot_forecast_comparison(
                self.data,
                forecast_predictions,
                title="Forecasting Methods Comparison",
                save_path=save_path
            )
        
        # Plot anomaly detection results
        if self.anomaly_results:
            if save_plots:
                save_path = os.path.join(plots_dir, 'anomaly_detection.png')
            else:
                save_path = None
            
            # Extract anomaly masks and scores
            anomalies = {}
            scores = {}
            for method, result in self.anomaly_results.items():
                anomalies[method] = result['anomalies']
                if 'scores' in result:
                    scores[method] = result['scores']
            
            self.visualizer.plot_anomalies(
                self.data,
                anomalies,
                scores,
                title="Anomaly Detection Results",
                save_path=save_path
            )
        
        # Create comprehensive dashboard
        if save_plots:
            save_path = os.path.join(plots_dir, 'comprehensive_dashboard.png')
        else:
            save_path = None
        
        self.visualizer.create_dashboard(
            self.data,
            forecasts=forecast_predictions if self.forecast_results else None,
            anomalies=anomalies if self.anomaly_results else None,
            title="Time Series Analysis Dashboard",
            save_path=save_path
        )
        
        self.logger.info("Visualizations created")
    
    def save_results(self) -> None:
        """Save analysis results."""
        self.logger.info("Saving results...")
        
        results_dir = self.config['output']['results_dir']
        
        # Save forecast results
        if self.forecast_results:
            forecast_summary = {}
            for method, result in self.forecast_results.items():
                forecast_summary[method] = {
                    'metrics': result['metrics'],
                    'predictions': result['predictions'].tolist()
                }
            
            import json
            with open(os.path.join(results_dir, 'forecast_results.json'), 'w') as f:
                json.dump(forecast_summary, f, indent=2)
        
        # Save anomaly detection results
        if self.anomaly_results:
            anomaly_summary = {}
            for method, result in self.anomaly_results.items():
                anomaly_summary[method] = {
                    'n_anomalies': int(result['n_anomalies']),
                    'anomaly_rate': float(result['anomaly_rate']),
                    'anomalies': result['anomalies'].tolist()
                }
            
            import json
            with open(os.path.join(results_dir, 'anomaly_results.json'), 'w') as f:
                json.dump(anomaly_summary, f, indent=2)
        
        self.logger.info("Results saved")
    
    def run_complete_analysis(self) -> None:
        """Run the complete time series analysis pipeline."""
        self.logger.info("Starting complete time series analysis...")
        
        try:
            # Load and prepare data
            self.load_data()
            
            # Train GRU model
            self.train_gru_model()
            
            # Run forecasting methods
            self.run_forecasting_methods()
            
            # Run anomaly detection
            self.run_anomaly_detection()
            
            # Create visualizations
            self.create_visualizations()
            
            # Save results
            self.save_results()
            
            self.logger.info("Complete time series analysis finished successfully!")
            
        except Exception as e:
            self.logger.error(f"Error in complete analysis: {e}")
            raise


def main():
    """Main function to run the time series analysis application."""
    app = TimeSeriesAnalysisApp()
    app.run_complete_analysis()


if __name__ == "__main__":
    main()
