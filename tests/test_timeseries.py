"""Unit tests for time series analysis components."""

import unittest
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_preprocessing import TimeSeriesPreprocessor, generate_synthetic_data
from gru_model import GRUModel, GRUTrainer
from forecasting_methods import ForecastingMethods
from anomaly_detection import AnomalyDetector


class TestDataPreprocessing(unittest.TestCase):
    """Test cases for data preprocessing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = TimeSeriesPreprocessor()
        self.test_data = np.random.randn(100, 1)
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        self.assertEqual(self.preprocessor.scaler_type, "minmax")
        self.assertFalse(self.preprocessor.is_fitted)
    
    def test_normalize_data(self):
        """Test data normalization."""
        normalized = self.preprocessor.normalize_data(self.test_data)
        
        # Check that data is normalized
        self.assertTrue(np.all(normalized >= 0))
        self.assertTrue(np.all(normalized <= 1))
        self.assertTrue(self.preprocessor.is_fitted)
    
    def test_inverse_transform(self):
        """Test inverse transformation."""
        normalized = self.preprocessor.normalize_data(self.test_data)
        original = self.preprocessor.inverse_transform(normalized)
        
        # Check that inverse transform recovers original data
        np.testing.assert_array_almost_equal(original, self.test_data, decimal=5)
    
    def test_create_sequences(self):
        """Test sequence creation."""
        sequence_length = 10
        X, y = self.preprocessor.create_sequences(self.test_data, sequence_length)
        
        # Check dimensions
        self.assertEqual(X.shape[1], sequence_length)
        self.assertEqual(X.shape[0], len(self.test_data) - sequence_length)
        self.assertEqual(len(y), len(self.test_data) - sequence_length)
    
    def test_train_test_split_sequences(self):
        """Test train-test split for sequences."""
        X = np.random.randn(100, 10, 1)
        y = np.random.randn(100, 1)
        
        X_train, X_test, y_train, y_test = self.preprocessor.train_test_split_sequences(
            X, y, test_size=0.2, random_state=42
        )
        
        # Check split sizes
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        data = generate_synthetic_data(n_samples=100, random_state=42)
        
        # Check data structure
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 100)
        self.assertIn('Close', data.columns)
        self.assertIn('Open', data.columns)
        self.assertIn('High', data.columns)
        self.assertIn('Low', data.columns)
        self.assertIn('Volume', data.columns)


class TestGRUModel(unittest.TestCase):
    """Test cases for GRU model functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = GRUModel(input_size=1, hidden_size=10, num_layers=1)
        self.test_input = torch.randn(32, 10, 1)  # batch_size=32, seq_len=10, input_size=1
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.input_size, 1)
        self.assertEqual(self.model.hidden_size, 10)
        self.assertEqual(self.model.num_layers, 1)
    
    def test_forward_pass(self):
        """Test forward pass."""
        output = self.model(self.test_input)
        
        # Check output shape
        self.assertEqual(output.shape, (32, 1))  # batch_size=32, output_size=1
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = GRUTrainer(self.model)
        
        self.assertIsInstance(trainer.model, GRUModel)
        self.assertIsNotNone(trainer.criterion)
        self.assertIsNotNone(trainer.optimizer)
    
    def test_model_parameters(self):
        """Test that model has trainable parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(total_params, 0)


class TestForecastingMethods(unittest.TestCase):
    """Test cases for forecasting methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.forecaster = ForecastingMethods()
        self.test_data = np.random.randn(100) + np.linspace(0, 10, 100)  # Data with trend
    
    def test_forecaster_initialization(self):
        """Test forecaster initialization."""
        self.assertEqual(len(self.forecaster.models), 0)
        self.assertEqual(len(self.forecaster.predictions), 0)
        self.assertEqual(len(self.forecaster.metrics), 0)
    
    def test_naive_forecast_last(self):
        """Test naive forecasting with last value method."""
        result = self.forecaster.naive_forecast(self.test_data, method='last')
        
        # Check result structure
        self.assertIn('method', result)
        self.assertIn('predictions', result)
        self.assertIn('metrics', result)
        self.assertEqual(result['method'], 'Naive-Last')
        
        # Check predictions are constant
        predictions = result['predictions']
        self.assertTrue(np.all(predictions == predictions[0]))
    
    def test_naive_forecast_mean(self):
        """Test naive forecasting with mean method."""
        result = self.forecaster.naive_forecast(self.test_data, method='mean')
        
        # Check predictions are constant and equal to mean
        predictions = result['predictions']
        expected_mean = np.mean(self.test_data)
        self.assertTrue(np.all(predictions == expected_mean))
    
    def test_naive_forecast_drift(self):
        """Test naive forecasting with drift method."""
        result = self.forecaster.naive_forecast(self.test_data, method='drift')
        
        # Check predictions show linear trend
        predictions = result['predictions']
        self.assertGreater(predictions[-1], predictions[0])
    
    def test_get_available_methods(self):
        """Test getting available forecasting methods."""
        methods = self.forecaster.get_available_methods()
        
        # Should always include naive methods
        self.assertIn('Naive-last', methods)
        self.assertIn('Naive-mean', methods)
        self.assertIn('Naive-drift', methods)
        self.assertIn('Naive-seasonal', methods)


class TestAnomalyDetector(unittest.TestCase):
    """Test cases for anomaly detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = AnomalyDetector()
        # Create test data with some outliers
        self.test_data = np.random.randn(100)
        self.test_data[50] = 10  # Add outlier
        self.test_data[75] = -10  # Add outlier
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        self.assertEqual(len(self.detector.models), 0)
        self.assertEqual(len(self.detector.anomalies), 0)
        self.assertEqual(len(self.detector.scores), 0)
    
    def test_isolation_forest_detection(self):
        """Test Isolation Forest anomaly detection."""
        result = self.detector.isolation_forest_detection(self.test_data)
        
        # Check result structure
        self.assertIn('method', result)
        self.assertIn('anomalies', result)
        self.assertIn('scores', result)
        self.assertEqual(result['method'], 'Isolation Forest')
        
        # Check that outliers are detected
        anomalies = result['anomalies']
        self.assertTrue(np.any(anomalies))
        self.assertGreater(result['n_anomalies'], 0)
    
    def test_statistical_detection_zscore(self):
        """Test statistical anomaly detection with z-score."""
        result = self.detector.statistical_detection(self.test_data, method='zscore')
        
        # Check result structure
        self.assertIn('method', result)
        self.assertIn('anomalies', result)
        self.assertIn('scores', result)
        self.assertEqual(result['method'], 'Statistical-Zscore')
        
        # Check that outliers are detected
        anomalies = result['anomalies']
        self.assertTrue(np.any(anomalies))
    
    def test_statistical_detection_iqr(self):
        """Test statistical anomaly detection with IQR."""
        result = self.detector.statistical_detection(self.test_data, method='iqr')
        
        # Check result structure
        self.assertIn('method', result)
        self.assertIn('anomalies', result)
        self.assertEqual(result['method'], 'Statistical-Iqr')
        
        # Check that outliers are detected
        anomalies = result['anomalies']
        self.assertTrue(np.any(anomalies))


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def test_complete_pipeline(self):
        """Test complete preprocessing and modeling pipeline."""
        # Generate synthetic data
        data = generate_synthetic_data(n_samples=200, random_state=42)
        values = data['Close'].values
        
        # Preprocessing
        preprocessor = TimeSeriesPreprocessor()
        scaled_data = preprocessor.normalize_data(values.reshape(-1, 1))
        
        # Create sequences
        X, y = preprocessor.create_sequences(scaled_data, sequence_length=10)
        X_train, X_test, y_train, y_test = preprocessor.train_test_split_sequences(X, y)
        
        # Create and test model
        model = GRUModel(input_size=1, hidden_size=20, num_layers=1)
        trainer = GRUTrainer(model)
        
        # Test forward pass
        test_input = torch.FloatTensor(X_train[:5])
        output = model(test_input)
        
        # Check output shape
        self.assertEqual(output.shape, (5, 1))
        
        # Test forecasting
        forecaster = ForecastingMethods()
        naive_result = forecaster.naive_forecast(values, method='last')
        
        # Check forecasting result
        self.assertIn('predictions', naive_result)
        self.assertIn('metrics', naive_result)
        
        # Test anomaly detection
        detector = AnomalyDetector()
        anomaly_result = detector.isolation_forest_detection(values)
        
        # Check anomaly detection result
        self.assertIn('anomalies', anomaly_result)
        self.assertIn('n_anomalies', anomaly_result)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDataPreprocessing,
        TestGRUModel,
        TestForecastingMethods,
        TestAnomalyDetector,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
