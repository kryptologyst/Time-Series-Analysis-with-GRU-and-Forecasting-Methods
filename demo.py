#!/usr/bin/env python3
"""Demonstration script for the Time Series Analysis project.

This script shows the complete functionality of the modernized time series analysis project.
"""

import sys
import os
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    """Run a demonstration of the time series analysis project."""
    print("=" * 60)
    print("TIME SERIES ANALYSIS PROJECT DEMONSTRATION")
    print("=" * 60)
    
    try:
        # Import modules
        print("\n1. Importing modules...")
        from data_preprocessing import TimeSeriesPreprocessor, generate_synthetic_data
        from gru_model import GRUModel, GRUTrainer
        from forecasting_methods import ForecastingMethods
        from anomaly_detection import AnomalyDetector
        from visualization import TimeSeriesVisualizer
        print("   ✓ All modules imported successfully!")
        
        # Generate synthetic data
        print("\n2. Generating synthetic data...")
        data = generate_synthetic_data(n_samples=500, random_state=42)
        values = data['Close'].values
        print(f"   ✓ Generated {len(values)} data points")
        
        # Preprocessing
        print("\n3. Preprocessing data...")
        preprocessor = TimeSeriesPreprocessor()
        scaled_data = preprocessor.normalize_data(values.reshape(-1, 1))
        X, y = preprocessor.create_sequences(scaled_data, sequence_length=20)
        X_train, X_test, y_train, y_test = preprocessor.train_test_split_sequences(X, y)
        print(f"   ✓ Created {len(X_train)} training and {len(X_test)} test sequences")
        
        # GRU Model
        print("\n4. Creating GRU model...")
        model = GRUModel(input_size=1, hidden_size=30, num_layers=1)
        trainer = GRUTrainer(model, learning_rate=0.001)
        print(f"   ✓ GRU model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Forecasting
        print("\n5. Testing forecasting methods...")
        forecaster = ForecastingMethods()
        naive_result = forecaster.naive_forecast(values, method='last', forecast_steps=10)
        print(f"   ✓ Naive forecasting completed (MSE: {naive_result['metrics']['mse']:.4f})")
        
        # Anomaly Detection
        print("\n6. Testing anomaly detection...")
        detector = AnomalyDetector()
        iso_result = detector.isolation_forest_detection(values, contamination=0.1)
        stat_result = detector.statistical_detection(values, method='zscore')
        print(f"   ✓ Isolation Forest: {iso_result['n_anomalies']} anomalies detected")
        print(f"   ✓ Statistical Z-score: {stat_result['n_anomalies']} anomalies detected")
        
        # Visualization
        print("\n7. Testing visualization...")
        viz = TimeSeriesVisualizer()
        print("   ✓ Visualization module initialized")
        
        # Configuration
        print("\n8. Testing configuration...")
        import yaml
        config_path = Path(__file__).parent / "config" / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print("   ✓ Configuration loaded successfully")
        else:
            print("   ⚠ Configuration file not found")
        
        # Testing
        print("\n9. Running unit tests...")
        try:
            import unittest
            from tests.test_timeseries import TestDataPreprocessing, TestGRUModel
            
            # Run a subset of tests
            test_suite = unittest.TestSuite()
            test_suite.addTest(unittest.makeSuite(TestDataPreprocessing))
            test_suite.addTest(unittest.makeSuite(TestGRUModel))
            
            runner = unittest.TextTestRunner(verbosity=0)
            result = runner.run(test_suite)
            
            if result.wasSuccessful():
                print("   ✓ Unit tests passed successfully")
            else:
                print(f"   ⚠ Some tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
        except Exception as e:
            print(f"   ⚠ Could not run tests: {e}")
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\nNext steps:")
        print("1. Run 'python main.py' for complete analysis")
        print("2. Run 'streamlit run streamlit_app.py' for web interface")
        print("3. Check 'notebooks/example_analysis.ipynb' for detailed examples")
        print("4. Modify 'config/config.yaml' to customize settings")
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Please install required dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
