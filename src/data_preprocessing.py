"""Data preprocessing and feature engineering utilities for time series analysis."""

import logging
from typing import Tuple, Optional, List, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import yfinance as yf

logger = logging.getLogger(__name__)


class TimeSeriesPreprocessor:
    """Handles preprocessing of time series data for machine learning models."""
    
    def __init__(self, scaler_type: str = "minmax") -> None:
        """Initialize the preprocessor.
        
        Args:
            scaler_type: Type of scaler to use ('minmax' or 'standard')
        """
        self.scaler_type = scaler_type
        self.scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()
        self.is_fitted = False
        
    def load_stock_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        column: str = "Close"
    ) -> pd.DataFrame:
        """Load stock data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'NFLX')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            column: Column to extract ('Close', 'Open', 'High', 'Low', 'Volume')
            
        Returns:
            DataFrame with stock data
        """
        try:
            logger.info(f"Loading stock data for {symbol} from {start_date} to {end_date}")
            df = yf.download(symbol, start=start_date, end=end_date)
            
            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")
                
            return df
        except Exception as e:
            logger.error(f"Error loading stock data: {e}")
            raise
    
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize the data using the configured scaler.
        
        Args:
            data: Input data array
            
        Returns:
            Normalized data array
        """
        if not self.is_fitted:
            scaled_data = self.scaler.fit_transform(data)
            self.is_fitted = True
        else:
            scaled_data = self.scaler.transform(data)
            
        return scaled_data
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform normalized data back to original scale.
        
        Args:
            data: Normalized data array
            
        Returns:
            Data in original scale
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transform")
        return self.scaler.inverse_transform(data)
    
    def create_sequences(
        self, 
        data: np.ndarray, 
        sequence_length: int = 30,
        target_column: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction.
        
        Args:
            data: Input time series data
            sequence_length: Length of input sequences
            target_column: Column index for target variable (if None, uses last column)
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        if target_column is None:
            target_column = data.shape[1] - 1
            
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length, target_column])
            
        return np.array(X), np.array(y)
    
    def train_test_split_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split sequences into train and test sets.
        
        Args:
            X: Input sequences
            y: Target values
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )


def generate_synthetic_data(
    n_samples: int = 1000,
    trend: float = 0.01,
    seasonality_period: int = 30,
    noise_level: float = 0.1,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """Generate synthetic time series data for testing.
    
    Args:
        n_samples: Number of data points to generate
        trend: Linear trend coefficient
        seasonality_period: Period of seasonal component
        noise_level: Standard deviation of noise
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic time series data
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate time index
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Generate components
    trend_component = np.linspace(0, trend * n_samples, n_samples)
    seasonal_component = 10 * np.sin(2 * np.pi * np.arange(n_samples) / seasonality_period)
    noise_component = np.random.normal(0, noise_level, n_samples)
    
    # Combine components
    values = 100 + trend_component + seasonal_component + noise_component
    
    return pd.DataFrame({
        'Date': dates,
        'Close': values,
        'Open': values + np.random.normal(0, 0.5, n_samples),
        'High': values + np.abs(np.random.normal(0, 1, n_samples)),
        'Low': values - np.abs(np.random.normal(0, 1, n_samples)),
        'Volume': np.random.randint(1000000, 10000000, n_samples)
    }).set_index('Date')
