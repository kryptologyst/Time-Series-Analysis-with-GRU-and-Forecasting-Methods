"""Anomaly detection methods for time series analysis."""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings

# Deep learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Autoencoder anomaly detection will be disabled.")

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Collection of anomaly detection methods for time series data."""
    
    def __init__(self) -> None:
        """Initialize the anomaly detector."""
        self.models = {}
        self.anomalies = {}
        self.scores = {}
    
    def isolation_forest_detection(
        self,
        data: np.ndarray,
        contamination: float = 0.1,
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """Isolation Forest anomaly detection.
        
        Args:
            data: Time series data
            contamination: Expected proportion of anomalies
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with anomaly detection results
        """
        try:
            # Reshape data for Isolation Forest
            if data.ndim == 1:
                data_2d = data.reshape(-1, 1)
            else:
                data_2d = data
            
            # Fit Isolation Forest
            model = IsolationForest(
                contamination=contamination,
                random_state=random_state,
                n_estimators=100
            )
            
            anomaly_labels = model.fit_predict(data_2d)
            anomaly_scores = model.decision_function(data_2d)
            
            # Convert labels: -1 for anomalies, 1 for normal
            is_anomaly = anomaly_labels == -1
            
            result = {
                'method': 'Isolation Forest',
                'anomalies': is_anomaly,
                'scores': anomaly_scores,
                'model': model,
                'contamination': contamination,
                'n_anomalies': np.sum(is_anomaly),
                'anomaly_rate': np.mean(is_anomaly)
            }
            
            self.models['Isolation Forest'] = model
            self.anomalies['Isolation Forest'] = is_anomaly
            self.scores['Isolation Forest'] = anomaly_scores
            
            return result
            
        except Exception as e:
            logger.error(f"Isolation Forest detection failed: {e}")
            raise
    
    def statistical_detection(
        self,
        data: np.ndarray,
        method: str = 'zscore',
        threshold: float = 3.0,
        window_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Statistical anomaly detection methods.
        
        Args:
            data: Time series data
            method: 'zscore', 'iqr', or 'modified_zscore'
            threshold: Threshold for anomaly detection
            window_size: Window size for rolling statistics (if None, uses global stats)
            
        Returns:
            Dictionary with anomaly detection results
        """
        try:
            if method == 'zscore':
                if window_size is None:
                    # Global z-score
                    mean_val = np.mean(data)
                    std_val = np.std(data)
                    scores = np.abs((data - mean_val) / std_val)
                else:
                    # Rolling z-score
                    scores = []
                    for i in range(len(data)):
                        start_idx = max(0, i - window_size)
                        window_data = data[start_idx:i+1]
                        mean_val = np.mean(window_data)
                        std_val = np.std(window_data)
                        if std_val > 0:
                            z_score = np.abs((data[i] - mean_val) / std_val)
                        else:
                            z_score = 0
                        scores.append(z_score)
                    scores = np.array(scores)
                
            elif method == 'iqr':
                if window_size is None:
                    # Global IQR
                    q1 = np.percentile(data, 25)
                    q3 = np.percentile(data, 75)
                    iqr = q3 - q1
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr
                    is_anomaly = (data < lower_bound) | (data > upper_bound)
                    scores = np.where(is_anomaly, threshold, 0)
                else:
                    # Rolling IQR
                    scores = []
                    for i in range(len(data)):
                        start_idx = max(0, i - window_size)
                        window_data = data[start_idx:i+1]
                        q1 = np.percentile(window_data, 25)
                        q3 = np.percentile(window_data, 75)
                        iqr = q3 - q1
                        if iqr > 0:
                            lower_bound = q1 - threshold * iqr
                            upper_bound = q3 + threshold * iqr
                            is_outlier = (data[i] < lower_bound) | (data[i] > upper_bound)
                            score = threshold if is_outlier else 0
                        else:
                            score = 0
                        scores.append(score)
                    scores = np.array(scores)
                    is_anomaly = scores >= threshold
                
            elif method == 'modified_zscore':
                if window_size is None:
                    # Global modified z-score using median
                    median_val = np.median(data)
                    mad = np.median(np.abs(data - median_val))
                    scores = np.abs(0.6745 * (data - median_val) / mad) if mad > 0 else np.zeros_like(data)
                else:
                    # Rolling modified z-score
                    scores = []
                    for i in range(len(data)):
                        start_idx = max(0, i - window_size)
                        window_data = data[start_idx:i+1]
                        median_val = np.median(window_data)
                        mad = np.median(np.abs(window_data - median_val))
                        if mad > 0:
                            modified_z = np.abs(0.6745 * (data[i] - median_val) / mad)
                        else:
                            modified_z = 0
                        scores.append(modified_z)
                    scores = np.array(scores)
                
            else:
                raise ValueError(f"Unknown statistical method: {method}")
            
            # Determine anomalies based on threshold
            if method != 'iqr' or window_size is not None:
                is_anomaly = scores >= threshold
            
            result = {
                'method': f'Statistical-{method.title()}',
                'anomalies': is_anomaly,
                'scores': scores,
                'threshold': threshold,
                'n_anomalies': np.sum(is_anomaly),
                'anomaly_rate': np.mean(is_anomaly)
            }
            
            self.models[f'Statistical-{method.title()}'] = None
            self.anomalies[f'Statistical-{method.title()}'] = is_anomaly
            self.scores[f'Statistical-{method.title()}'] = scores
            
            return result
            
        except Exception as e:
            logger.error(f"Statistical detection failed: {e}")
            raise


class AutoencoderAnomalyDetector:
    """Autoencoder-based anomaly detection for time series."""
    
    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 10,
        hidden_dims: List[int] = None
    ) -> None:
        """Initialize the autoencoder.
        
        Args:
            input_dim: Input dimension
            encoding_dim: Encoding dimension
            hidden_dims: List of hidden layer dimensions
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for autoencoder anomaly detection")
        
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims or [64, 32]
        
        self.model = self._build_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def _build_model(self) -> nn.Module:
        """Build the autoencoder model."""
        
        class Autoencoder(nn.Module):
            def __init__(self, input_dim, encoding_dim, hidden_dims):
                super().__init__()
                
                # Encoder
                encoder_layers = []
                prev_dim = input_dim
                for hidden_dim in hidden_dims:
                    encoder_layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2)
                    ])
                    prev_dim = hidden_dim
                
                encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
                self.encoder = nn.Sequential(*encoder_layers)
                
                # Decoder
                decoder_layers = []
                prev_dim = encoding_dim
                for hidden_dim in reversed(hidden_dims):
                    decoder_layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2)
                    ])
                    prev_dim = hidden_dim
                
                decoder_layers.append(nn.Linear(prev_dim, input_dim))
                self.decoder = nn.Sequential(*decoder_layers)
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        return Autoencoder(self.input_dim, self.encoding_dim, self.hidden_dims)
    
    def fit(
        self,
        data: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """Train the autoencoder.
        
        Args:
            data: Training data
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            validation_split: Fraction of data for validation
            
        Returns:
            Training history
        """
        try:
            # Prepare data
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            # Normalize data
            scaled_data = self.scaler.fit_transform(data)
            self.is_fitted = True
            
            # Convert to tensors
            data_tensor = torch.FloatTensor(scaled_data)
            
            # Split data
            n_train = int(len(data_tensor) * (1 - validation_split))
            train_data = data_tensor[:n_train]
            val_data = data_tensor[n_train:] if validation_split > 0 else None
            
            # Create data loaders
            train_loader = DataLoader(
                TensorDataset(train_data, train_data),
                batch_size=batch_size,
                shuffle=True
            )
            
            val_loader = None
            if val_data is not None:
                val_loader = DataLoader(
                    TensorDataset(val_data, val_data),
                    batch_size=batch_size,
                    shuffle=False
                )
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10
            )
            
            # Training loop
            train_losses = []
            val_losses = []
            
            for epoch in range(epochs):
                # Training
                self.model.train()
                train_loss = 0.0
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                train_losses.append(train_loss)
                
                # Validation
                val_loss = 0.0
                if val_loader is not None:
                    self.model.eval()
                    with torch.no_grad():
                        for batch_x, batch_y in val_loader:
                            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                            outputs = self.model(batch_x)
                            loss = criterion(outputs, batch_y)
                            val_loss += loss.item()
                    
                    val_loss /= len(val_loader)
                    val_losses.append(val_loss)
                    scheduler.step(val_loss)
                
                if epoch % 20 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            return {
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            
        except Exception as e:
            logger.error(f"Autoencoder training failed: {e}")
            raise
    
    def detect_anomalies(
        self,
        data: np.ndarray,
        threshold_percentile: float = 95.0
    ) -> Dict[str, Any]:
        """Detect anomalies using the trained autoencoder.
        
        Args:
            data: Data to analyze
            threshold_percentile: Percentile for anomaly threshold
            
        Returns:
            Dictionary with anomaly detection results
        """
        if not self.is_fitted:
            raise ValueError("Autoencoder must be fitted before anomaly detection")
        
        try:
            # Prepare data
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            # Normalize data
            scaled_data = self.scaler.transform(data)
            
            # Convert to tensor
            data_tensor = torch.FloatTensor(scaled_data)
            data_loader = DataLoader(
                TensorDataset(data_tensor, data_tensor),
                batch_size=32,
                shuffle=False
            )
            
            # Get reconstruction errors
            self.model.eval()
            reconstruction_errors = []
            
            with torch.no_grad():
                for batch_x, _ in data_loader:
                    batch_x = batch_x.to(self.device)
                    reconstructed = self.model(batch_x)
                    error = torch.mean((batch_x - reconstructed) ** 2, dim=1)
                    reconstruction_errors.append(error.cpu().numpy())
            
            reconstruction_errors = np.concatenate(reconstruction_errors)
            
            # Determine threshold
            threshold = np.percentile(reconstruction_errors, threshold_percentile)
            
            # Detect anomalies
            is_anomaly = reconstruction_errors > threshold
            
            result = {
                'method': 'Autoencoder',
                'anomalies': is_anomaly,
                'scores': reconstruction_errors,
                'threshold': threshold,
                'n_anomalies': np.sum(is_anomaly),
                'anomaly_rate': np.mean(is_anomaly)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Autoencoder anomaly detection failed: {e}")
            raise


def plot_anomalies(
    data: np.ndarray,
    anomalies: Dict[str, np.ndarray],
    scores: Optional[Dict[str, np.ndarray]] = None,
    title: str = "Anomaly Detection Results"
) -> None:
    """Plot anomaly detection results.
    
    Args:
        data: Original time series data
        anomalies: Dictionary of anomaly detection results
        scores: Dictionary of anomaly scores (optional)
        title: Plot title
    """
    n_methods = len(anomalies)
    fig, axes = plt.subplots(n_methods + 1, 1, figsize=(12, 3 * (n_methods + 1)))
    
    if n_methods == 0:
        axes = [axes]
    
    # Plot original data
    axes[0].plot(data, label='Original Data', alpha=0.7)
    axes[0].set_title('Original Time Series')
    axes[0].legend()
    axes[0].grid(True)
    
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
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
