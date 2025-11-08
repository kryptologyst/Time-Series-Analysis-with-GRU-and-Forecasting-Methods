"""Advanced visualization utilities for time series analysis."""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import warnings

# Plotly imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive plots will be disabled.")

logger = logging.getLogger(__name__)


class TimeSeriesVisualizer:
    """Advanced visualization utilities for time series analysis."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)) -> None:
        """Initialize the visualizer.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        plt.style.use(style)
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    def plot_time_series(
        self,
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        title: str = "Time Series Plot",
        xlabel: str = "Time",
        ylabel: str = "Value",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Plot time series data.
        
        Args:
            data: Time series data
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            save_path: Path to save the plot
        """
        figsize = figsize or self.figsize
        plt.figure(figsize=figsize)
        
        if isinstance(data, pd.DataFrame):
            for i, column in enumerate(data.columns):
                plt.plot(data.index, data[column], label=column, color=self.colors[i % len(self.colors)])
            plt.legend()
        elif isinstance(data, pd.Series):
            plt.plot(data.index, data.values, label=data.name or 'Data')
        else:
            plt.plot(data, label='Data')
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_forecast_comparison(
        self,
        actual: np.ndarray,
        predictions: Dict[str, np.ndarray],
        title: str = "Forecast Comparison",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Plot comparison of different forecasting methods.
        
        Args:
            actual: Actual values
            predictions: Dictionary of method names and their predictions
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot
        """
        figsize = figsize or self.figsize
        plt.figure(figsize=figsize)
        
        # Plot actual data
        plt.plot(actual, label='Actual', color='black', linewidth=2, alpha=0.8)
        
        # Plot predictions
        for i, (method, pred) in enumerate(predictions.items()):
            plt.plot(pred, label=f'{method} Forecast', 
                    color=self.colors[i % len(self.colors)], alpha=0.7)
        
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_anomalies(
        self,
        data: np.ndarray,
        anomalies: Dict[str, np.ndarray],
        scores: Optional[Dict[str, np.ndarray]] = None,
        title: str = "Anomaly Detection Results",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Plot anomaly detection results.
        
        Args:
            data: Original time series data
            anomalies: Dictionary of anomaly detection results
            scores: Dictionary of anomaly scores (optional)
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot
        """
        n_methods = len(anomalies)
        figsize = figsize or (12, 3 * (n_methods + 1))
        
        fig, axes = plt.subplots(n_methods + 1, 1, figsize=figsize)
        
        if n_methods == 0:
            axes = [axes]
        
        # Plot original data
        axes[0].plot(data, label='Original Data', alpha=0.7, color='blue')
        axes[0].set_title('Original Time Series')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot anomaly detection results
        for i, (method, anomaly_mask) in enumerate(anomalies.items()):
            ax = axes[i + 1]
            
            # Plot data
            ax.plot(data, label='Data', alpha=0.7, color='blue')
            
            # Highlight anomalies
            anomaly_indices = np.where(anomaly_mask)[0]
            if len(anomaly_indices) > 0:
                ax.scatter(
                    anomaly_indices,
                    data[anomaly_indices],
                    color='red',
                    s=50,
                    label=f'Anomalies ({np.sum(anomaly_mask)})',
                    zorder=5
                )
            
            # Plot scores if available
            if scores and method in scores:
                ax2 = ax.twinx()
                ax2.plot(scores[method], color='orange', alpha=0.5, label='Anomaly Score')
                ax2.set_ylabel('Anomaly Score', color='orange')
                ax2.tick_params(axis='y', labelcolor='orange')
            
            ax.set_title(f'{method} Detection')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_decomposition(
        self,
        data: np.ndarray,
        period: int = 12,
        title: str = "Time Series Decomposition",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Plot time series decomposition.
        
        Args:
            data: Time series data
            period: Seasonal period
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot
        """
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            figsize = figsize or (12, 10)
            plt.figure(figsize=figsize)
            
            # Perform decomposition
            decomposition = seasonal_decompose(data, model='additive', period=period)
            
            # Plot components
            plt.subplot(4, 1, 1)
            plt.plot(decomposition.observed)
            plt.title('Original')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(4, 1, 2)
            plt.plot(decomposition.trend)
            plt.title('Trend')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(4, 1, 3)
            plt.plot(decomposition.seasonal)
            plt.title('Seasonal')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(4, 1, 4)
            plt.plot(decomposition.resid)
            plt.title('Residual')
            plt.grid(True, alpha=0.3)
            
            plt.suptitle(title)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
        except ImportError:
            logger.warning("statsmodels not available for decomposition")
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
    
    def plot_correlation_matrix(
        self,
        data: pd.DataFrame,
        title: str = "Correlation Matrix",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Plot correlation matrix heatmap.
        
        Args:
            data: DataFrame with numeric columns
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot
        """
        figsize = figsize or (10, 8)
        plt.figure(figsize=figsize)
        
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Create heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            fmt='.2f'
        )
        
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_performance(
        self,
        metrics: Dict[str, Dict[str, float]],
        title: str = "Model Performance Comparison",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Plot model performance comparison.
        
        Args:
            metrics: Dictionary of model metrics
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot
        """
        figsize = figsize or (12, 8)
        
        # Extract metrics
        models = list(metrics.keys())
        metric_names = list(next(iter(metrics.values())).keys())
        
        # Create subplots
        n_metrics = len(metric_names)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metric_names):
            values = [metrics[model][metric] for model in models]
            
            bars = axes[i].bar(models, values, color=self.colors[:len(models)])
            axes[i].set_title(f'{metric.upper()}')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_interactive_time_series(
        self,
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        title: str = "Interactive Time Series",
        save_path: Optional[str] = None
    ) -> None:
        """Create interactive time series plot using Plotly.
        
        Args:
            data: Time series data
            title: Plot title
            save_path: Path to save the plot
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for interactive plots")
            return
        
        fig = go.Figure()
        
        if isinstance(data, pd.DataFrame):
            for column in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[column],
                    mode='lines',
                    name=column
                ))
        elif isinstance(data, pd.Series):
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data.values,
                mode='lines',
                name=data.name or 'Data'
            ))
        else:
            fig.add_trace(go.Scatter(
                y=data,
                mode='lines',
                name='Data'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def plot_interactive_forecast(
        self,
        actual: np.ndarray,
        predictions: Dict[str, np.ndarray],
        title: str = "Interactive Forecast Comparison",
        save_path: Optional[str] = None
    ) -> None:
        """Create interactive forecast comparison plot.
        
        Args:
            actual: Actual values
            predictions: Dictionary of method names and their predictions
            title: Plot title
            save_path: Path to save the plot
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for interactive plots")
            return
        
        fig = go.Figure()
        
        # Add actual data
        fig.add_trace(go.Scatter(
            y=actual,
            mode='lines',
            name='Actual',
            line=dict(color='black', width=2)
        ))
        
        # Add predictions
        for method, pred in predictions.items():
            fig.add_trace(go.Scatter(
                y=pred,
                mode='lines',
                name=f'{method} Forecast',
                line=dict(dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def create_dashboard(
        self,
        data: np.ndarray,
        forecasts: Optional[Dict[str, np.ndarray]] = None,
        anomalies: Optional[Dict[str, np.ndarray]] = None,
        title: str = "Time Series Analysis Dashboard",
        save_path: Optional[str] = None
    ) -> None:
        """Create a comprehensive dashboard.
        
        Args:
            data: Time series data
            forecasts: Dictionary of forecasts (optional)
            anomalies: Dictionary of anomaly detection results (optional)
            title: Dashboard title
            save_path: Path to save the plot
        """
        n_plots = 1
        if forecasts:
            n_plots += 1
        if anomalies:
            n_plots += len(anomalies)
        
        figsize = (15, 4 * n_plots)
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
        
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Original time series
        axes[plot_idx].plot(data, label='Original Data', color='blue')
        axes[plot_idx].set_title('Original Time Series')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
        
        # Forecasts
        if forecasts:
            axes[plot_idx].plot(data, label='Actual', color='black', linewidth=2)
            for method, pred in forecasts.items():
                axes[plot_idx].plot(pred, label=f'{method} Forecast', alpha=0.7)
            axes[plot_idx].set_title('Forecast Comparison')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
        
        # Anomalies
        if anomalies:
            for method, anomaly_mask in anomalies.items():
                axes[plot_idx].plot(data, label='Data', alpha=0.7, color='blue')
                
                anomaly_indices = np.where(anomaly_mask)[0]
                if len(anomaly_indices) > 0:
                    axes[plot_idx].scatter(
                        anomaly_indices,
                        data[anomaly_indices],
                        color='red',
                        s=50,
                        label=f'Anomalies ({np.sum(anomaly_mask)})',
                        zorder=5
                    )
                
                axes[plot_idx].set_title(f'{method} Anomaly Detection')
                axes[plot_idx].legend()
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
