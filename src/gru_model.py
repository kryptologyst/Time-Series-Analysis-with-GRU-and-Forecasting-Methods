"""Enhanced GRU model with modern best practices and additional features."""

import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class GRUModel(nn.Module):
    """Enhanced GRU model for time series forecasting with dropout and batch normalization."""
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 50,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False
    ) -> None:
        """Initialize the GRU model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in GRU layers
            num_layers: Number of GRU layers
            dropout: Dropout rate for regularization
            bidirectional: Whether to use bidirectional GRU
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate output size based on bidirectional setting
        output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Fully connected layers with batch normalization
        self.batch_norm = nn.BatchNorm1d(output_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(output_size, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize model weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # GRU forward pass
        gru_out, _ = self.gru(x)
        
        # Take the last output
        last_output = gru_out[:, -1, :]
        
        # Apply batch normalization and dropout
        normalized = self.batch_norm(last_output)
        dropped = self.dropout_layer(normalized)
        
        # Final linear layer
        output = self.fc(dropped)
        
        return output


class GRUTrainer:
    """Trainer class for GRU model with training utilities."""
    
    def __init__(
        self,
        model: GRUModel,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        device: Optional[str] = None
    ) -> None:
        """Initialize the trainer.
        
        Args:
            model: GRU model to train
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            device: Device to use ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Tuple[float, Optional[float]]:
        """Train the model for one epoch.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            
        Returns:
            Tuple of (train_loss, val_loss)
        """
        self.model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            # Forward pass
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        val_loss = None
        if val_loader is not None:
            val_loss = self.validate(val_loader)
            self.scheduler.step(val_loss)
        
        return train_loss, val_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Validation loss
        """
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        return val_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        early_stopping_patience: int = 20,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, val_loss = self.train_epoch(train_loader, val_loader)
            
            self.train_losses.append(train_loss)
            if val_loss is not None:
                self.val_losses.append(val_loss)
            
            if verbose and epoch % 10 == 0:
                if val_loss is not None:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                else:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}")
            
            # Early stopping
            if val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }
    
    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """Make predictions on a dataset.
        
        Args:
            data_loader: Data loader for prediction
            
        Returns:
            Array of predictions
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                predictions.append(outputs.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def plot_training_history(self) -> None:
        """Plot training and validation loss curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        logger.info(f"Model loaded from {filepath}")
