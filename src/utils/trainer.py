"""Training module for emotion recognition."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Tuple, Optional
from tqdm import tqdm


class Trainer:
    """
    Trainer for multimodal LSTM emotion classifier.
    
    Args:
        model: The model to train
        device: Device to use for training (cpu or cuda)
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
    """
    
    def __init__(self, model, device='cpu', learning_rate=0.001, weight_decay=1e-5):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader: DataLoader, verbose: bool = True) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            verbose: Whether to show progress bar
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        iterator = tqdm(train_loader, desc="Training") if verbose else train_loader
        
        for batch_data, batch_labels in iterator:
            # Move data to device
            for modality in batch_data:
                batch_data[modality] = batch_data[modality].to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_data)
            loss = self.criterion(outputs, batch_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
            if verbose:
                iterator.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader, verbose: bool = True) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            verbose: Whether to show progress bar
        
        Returns:
            Tuple of (average_loss, accuracy, true_labels, predicted_labels)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        iterator = tqdm(val_loader, desc="Validation") if verbose else val_loader
        
        with torch.no_grad():
            for batch_data, batch_labels in iterator:
                # Move data to device
                for modality in batch_data:
                    batch_data[modality] = batch_data[modality].to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_labels)
                
                # Track metrics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy, np.array(all_labels), np.array(all_predictions)
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              num_epochs: int = 50, early_stopping_patience: int = 10,
              verbose: bool = True) -> Dict:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Number of epochs to train
            early_stopping_patience: Number of epochs to wait before early stopping
            verbose: Whether to show progress
        
        Returns:
            Dictionary with training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            if verbose:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, verbose)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            if verbose:
                print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            
            # Validate
            if val_loader is not None:
                val_loss, val_acc, _, _ = self.validate(val_loader, verbose)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                
                if verbose:
                    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
    
    def save_model(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': self.model.get_config()
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
