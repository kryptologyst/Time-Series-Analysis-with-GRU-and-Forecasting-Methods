"""Multiple forecasting methods for time series analysis."""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

# Forecasting libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. ARIMA forecasting will be disabled.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("prophet not available. Prophet forecasting will be disabled.")

try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    warnings.warn("pmdarima not available. Auto-ARIMA will be disabled.")

logger = logging.getLogger(__name__)


class ForecastingMethods:
    """Collection of forecasting methods for time series analysis."""
    
    def __init__(self) -> None:
        """Initialize the forecasting methods."""
        self.models = {}
        self.predictions = {}
        self.metrics = {}
    
    def arima_forecast(
        self,
        data: np.ndarray,
        order: Tuple[int, int, int] = (1, 1, 1),
        forecast_steps: int = 30,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict[str, Any]:
        """ARIMA forecasting method.
        
        Args:
            data: Time series data
            order: ARIMA order (p, d, q)
            forecast_steps: Number of steps to forecast
            seasonal_order: Seasonal ARIMA order (P, D, Q, s)
            
        Returns:
            Dictionary with predictions and metrics
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ARIMA forecasting")
        
        try:
            # Fit ARIMA model
            if seasonal_order:
                model = ARIMA(data, order=order, seasonal_order=seasonal_order)
            else:
                model = ARIMA(data, order=order)
            
            fitted_model = model.fit()
            
            # Make predictions
            forecast = fitted_model.forecast(steps=forecast_steps)
            fitted_values = fitted_model.fittedvalues
            
            # Calculate metrics
            train_data = data[:-forecast_steps] if len(data) > forecast_steps else data
            test_data = data[-forecast_steps:] if len(data) > forecast_steps else data
            
            if len(test_data) > 0:
                mse = mean_squared_error(test_data, forecast[:len(test_data)])
                mae = mean_absolute_error(test_data, forecast[:len(test_data)])
                r2 = r2_score(test_data, forecast[:len(test_data)])
            else:
                mse = mae = r2 = np.nan
            
            result = {
                'method': 'ARIMA',
                'predictions': forecast,
                'fitted_values': fitted_values,
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                },
                'model': fitted_model
            }
            
            self.models['ARIMA'] = fitted_model
            self.predictions['ARIMA'] = forecast
            self.metrics['ARIMA'] = result['metrics']
            
            return result
            
        except Exception as e:
            logger.error(f"ARIMA forecasting failed: {e}")
            raise
    
    def auto_arima_forecast(
        self,
        data: np.ndarray,
        forecast_steps: int = 30,
        seasonal: bool = True,
        max_p: int = 5,
        max_q: int = 5,
        max_P: int = 2,
        max_Q: int = 2
    ) -> Dict[str, Any]:
        """Auto-ARIMA forecasting with automatic parameter selection.
        
        Args:
            data: Time series data
            forecast_steps: Number of steps to forecast
            seasonal: Whether to consider seasonal models
            max_p, max_q, max_P, max_Q: Maximum values for ARIMA parameters
            
        Returns:
            Dictionary with predictions and metrics
        """
        if not PMDARIMA_AVAILABLE:
            raise ImportError("pmdarima is required for Auto-ARIMA forecasting")
        
        try:
            # Fit auto-ARIMA model
            model = pm.auto_arima(
                data,
                seasonal=seasonal,
                max_p=max_p,
                max_q=max_q,
                max_P=max_P,
                max_Q=max_Q,
                suppress_warnings=True,
                stepwise=True
            )
            
            # Make predictions
            forecast = model.predict(n_periods=forecast_steps)
            fitted_values = model.predict_in_sample()
            
            # Calculate metrics
            train_data = data[:-forecast_steps] if len(data) > forecast_steps else data
            test_data = data[-forecast_steps:] if len(data) > forecast_steps else data
            
            if len(test_data) > 0:
                mse = mean_squared_error(test_data, forecast[:len(test_data)])
                mae = mean_absolute_error(test_data, forecast[:len(test_data)])
                r2 = r2_score(test_data, forecast[:len(test_data)])
            else:
                mse = mae = r2 = np.nan
            
            result = {
                'method': 'Auto-ARIMA',
                'predictions': forecast,
                'fitted_values': fitted_values,
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                },
                'model': model,
                'order': model.order,
                'seasonal_order': model.seasonal_order
            }
            
            self.models['Auto-ARIMA'] = model
            self.predictions['Auto-ARIMA'] = forecast
            self.metrics['Auto-ARIMA'] = result['metrics']
            
            return result
            
        except Exception as e:
            logger.error(f"Auto-ARIMA forecasting failed: {e}")
            raise
    
    def prophet_forecast(
        self,
        data: pd.DataFrame,
        forecast_steps: int = 30,
        seasonality_mode: str = 'additive',
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False
    ) -> Dict[str, Any]:
        """Prophet forecasting method.
        
        Args:
            data: DataFrame with 'ds' (date) and 'y' (value) columns
            forecast_steps: Number of steps to forecast
            seasonality_mode: 'additive' or 'multiplicative'
            yearly_seasonality: Whether to include yearly seasonality
            weekly_seasonality: Whether to include weekly seasonality
            daily_seasonality: Whether to include daily seasonality
            
        Returns:
            Dictionary with predictions and metrics
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("prophet is required for Prophet forecasting")
        
        try:
            # Prepare data for Prophet
            if 'ds' not in data.columns or 'y' not in data.columns:
                raise ValueError("Data must have 'ds' and 'y' columns for Prophet")
            
            # Split data for evaluation
            split_idx = len(data) - forecast_steps if len(data) > forecast_steps else len(data)
            train_data = data.iloc[:split_idx].copy()
            test_data = data.iloc[split_idx:].copy() if split_idx < len(data) else pd.DataFrame()
            
            # Initialize and fit Prophet model
            model = Prophet(
                seasonality_mode=seasonality_mode,
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality
            )
            
            model.fit(train_data)
            
            # Make predictions
            future = model.make_future_dataframe(periods=forecast_steps)
            forecast_df = model.predict(future)
            
            # Extract predictions
            predictions = forecast_df['yhat'].iloc[-forecast_steps:].values
            fitted_values = forecast_df['yhat'].iloc[:len(train_data)].values
            
            # Calculate metrics
            if len(test_data) > 0:
                test_predictions = forecast_df['yhat'].iloc[split_idx:split_idx+len(test_data)].values
                mse = mean_squared_error(test_data['y'], test_predictions)
                mae = mean_absolute_error(test_data['y'], test_predictions)
                r2 = r2_score(test_data['y'], test_predictions)
            else:
                mse = mae = r2 = np.nan
            
            result = {
                'method': 'Prophet',
                'predictions': predictions,
                'fitted_values': fitted_values,
                'forecast_df': forecast_df,
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                },
                'model': model
            }
            
            self.models['Prophet'] = model
            self.predictions['Prophet'] = predictions
            self.metrics['Prophet'] = result['metrics']
            
            return result
            
        except Exception as e:
            logger.error(f"Prophet forecasting failed: {e}")
            raise
    
    def naive_forecast(
        self,
        data: np.ndarray,
        forecast_steps: int = 30,
        method: str = 'last'
    ) -> Dict[str, Any]:
        """Naive forecasting methods for baseline comparison.
        
        Args:
            data: Time series data
            forecast_steps: Number of steps to forecast
            method: 'last', 'mean', 'drift', or 'seasonal'
            
        Returns:
            Dictionary with predictions and metrics
        """
        try:
            if method == 'last':
                # Last value
                last_value = data[-1]
                predictions = np.full(forecast_steps, last_value)
                
            elif method == 'mean':
                # Mean of historical data
                mean_value = np.mean(data)
                predictions = np.full(forecast_steps, mean_value)
                
            elif method == 'drift':
                # Linear trend continuation
                first_value = data[0]
                last_value = data[-1]
                slope = (last_value - first_value) / (len(data) - 1)
                predictions = np.array([last_value + slope * (i + 1) for i in range(forecast_steps)])
                
            elif method == 'seasonal':
                # Seasonal naive (assumes seasonal pattern)
                seasonal_period = 12  # Monthly seasonality
                if len(data) >= seasonal_period:
                    seasonal_values = data[-seasonal_period:]
                    predictions = np.tile(seasonal_values, (forecast_steps // seasonal_period + 1))[:forecast_steps]
                else:
                    # Fallback to last value if not enough data
                    last_value = data[-1]
                    predictions = np.full(forecast_steps, last_value)
            else:
                raise ValueError(f"Unknown naive method: {method}")
            
            # Calculate metrics
            train_data = data[:-forecast_steps] if len(data) > forecast_steps else data
            test_data = data[-forecast_steps:] if len(data) > forecast_steps else data
            
            if len(test_data) > 0:
                mse = mean_squared_error(test_data, predictions[:len(test_data)])
                mae = mean_absolute_error(test_data, predictions[:len(test_data)])
                r2 = r2_score(test_data, predictions[:len(test_data)])
            else:
                mse = mae = r2 = np.nan
            
            result = {
                'method': f'Naive-{method.title()}',
                'predictions': predictions,
                'fitted_values': data,  # For naive methods, fitted values are the original data
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                }
            }
            
            self.models[f'Naive-{method.title()}'] = None
            self.predictions[f'Naive-{method.title()}'] = predictions
            self.metrics[f'Naive-{method.title()}'] = result['metrics']
            
            return result
            
        except Exception as e:
            logger.error(f"Naive forecasting failed: {e}")
            raise
    
    def compare_methods(
        self,
        data: np.ndarray,
        methods: List[str] = None,
        forecast_steps: int = 30
    ) -> pd.DataFrame:
        """Compare multiple forecasting methods.
        
        Args:
            data: Time series data
            methods: List of methods to compare
            forecast_steps: Number of steps to forecast
            
        Returns:
            DataFrame with comparison results
        """
        if methods is None:
            methods = ['ARIMA', 'Auto-ARIMA', 'Prophet', 'Naive-last']
        
        results = []
        
        for method in methods:
            try:
                if method == 'ARIMA':
                    result = self.arima_forecast(data, forecast_steps=forecast_steps)
                elif method == 'Auto-ARIMA':
                    result = self.auto_arima_forecast(data, forecast_steps=forecast_steps)
                elif method == 'Prophet':
                    # Convert to DataFrame format for Prophet
                    dates = pd.date_range(start='2020-01-01', periods=len(data), freq='D')
                    df = pd.DataFrame({'ds': dates, 'y': data})
                    result = self.prophet_forecast(df, forecast_steps=forecast_steps)
                elif method.startswith('Naive-'):
                    naive_method = method.split('-')[1].lower()
                    result = self.naive_forecast(data, forecast_steps=forecast_steps, method=naive_method)
                else:
                    continue
                
                results.append({
                    'Method': method,
                    'MSE': result['metrics']['mse'],
                    'MAE': result['metrics']['mae'],
                    'R2': result['metrics']['r2']
                })
                
            except Exception as e:
                logger.warning(f"Failed to run {method}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def get_available_methods(self) -> List[str]:
        """Get list of available forecasting methods based on installed packages."""
        methods = ['Naive-last', 'Naive-mean', 'Naive-drift', 'Naive-seasonal']
        
        if STATSMODELS_AVAILABLE:
            methods.append('ARIMA')
        
        if PMDARIMA_AVAILABLE:
            methods.append('Auto-ARIMA')
        
        if PROPHET_AVAILABLE:
            methods.append('Prophet')
        
        return methods
