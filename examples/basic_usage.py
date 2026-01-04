"""
Example: Basic usage of multimodal emotion recognition with LSTM and EHOLF.

This example demonstrates how to:
1. Create synthetic multimodal data
2. Train an LSTM classifier
3. Optimize hyperparameters using EHOLF
"""

import numpy as np
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import MultimodalLSTMClassifier
from data import MultimodalDataLoader, MultimodalPreprocessor
from optimization import EHOLF
from utils import Config, calculate_metrics, print_metrics
from utils.trainer import Trainer


def generate_synthetic_data(n_samples=1000, sequence_length=50):
    """Generate synthetic multimodal emotion data for demonstration."""
    
    # Define modalities with different feature dimensions
    text_features = 300  # Text embedding dimension
    audio_features = 40  # Audio MFCC features
    visual_features = 128  # Visual features
    
    num_classes = 7  # Number of emotion classes
    
    # Generate synthetic data
    text_data = np.random.randn(n_samples, sequence_length, text_features)
    audio_data = np.random.randn(n_samples, sequence_length, audio_features)
    visual_data = np.random.randn(n_samples, sequence_length, visual_features)
    
    # Generate labels
    labels = np.random.randint(0, num_classes, n_samples)
    
    # Add some pattern to make it learnable
    for i in range(n_samples):
        label = labels[i]
        # Add class-specific patterns
        text_data[i] += label * 0.1
        audio_data[i] += label * 0.1
        visual_data[i] += label * 0.1
    
    data = {
        'text': text_data,
        'audio': audio_data,
        'visual': visual_data
    }
    
    return data, labels


def basic_training_example():
    """Example: Basic training without hyperparameter optimization."""
    
    print("="*70)
    print("EXAMPLE 1: Basic Training (Without EHOLF Optimization)")
    print("="*70)
    
    # Load configuration
    config = Config()
    
    # Generate synthetic data
    print("\n1. Generating synthetic multimodal data...")
    data, labels = generate_synthetic_data(n_samples=1000)
    
    # Split data
    split_idx = int(0.8 * len(labels))
    train_data = {k: v[:split_idx] for k, v in data.items()}
    train_labels = labels[:split_idx]
    val_data = {k: v[split_idx:] for k, v in data.items()}
    val_labels = labels[split_idx:]
    
    print(f"   Training samples: {len(train_labels)}")
    print(f"   Validation samples: {len(val_labels)}")
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    preprocessor = MultimodalPreprocessor(
        normalize=config.get('data.normalize'),
        max_sequence_length=config.get('data.max_sequence_length')
    )
    train_data = preprocessor.fit_transform(train_data)
    val_data = preprocessor.transform(val_data)
    
    # Create data loaders
    print("\n3. Creating data loaders...")
    data_loader = MultimodalDataLoader(
        batch_size=config.get('training.batch_size'),
        shuffle=True
    )
    train_loader, val_loader = data_loader.create_train_val_loaders(
        train_data, train_labels, val_data, val_labels
    )
    
    # Create model
    print("\n4. Creating LSTM model...")
    input_sizes = {
        'text': train_data['text'].shape[2],
        'audio': train_data['audio'].shape[2],
        'visual': train_data['visual'].shape[2]
    }
    
    model = MultimodalLSTMClassifier(
        input_sizes=input_sizes,
        hidden_size=config.get('model.hidden_size'),
        num_layers=config.get('model.num_layers'),
        num_classes=config.get('model.num_classes'),
        dropout=config.get('model.dropout'),
        bidirectional=config.get('model.bidirectional')
    )
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n5. Training model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device}")
    
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=config.get('training.learning_rate'),
        weight_decay=config.get('training.weight_decay')
    )
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,  # Reduced for demo
        early_stopping_patience=config.get('training.early_stopping_patience'),
        verbose=True
    )
    
    # Evaluate
    print("\n6. Evaluating model...")
    _, _, y_true, y_pred = trainer.validate(val_loader, verbose=False)
    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics, config.get('emotions'))
    
    return model, trainer, history


def eholf_optimization_example():
    """Example: Hyperparameter optimization using EHOLF."""
    
    print("\n\n" + "="*70)
    print("EXAMPLE 2: Hyperparameter Optimization with EHOLF")
    print("="*70)
    
    # Generate data
    print("\n1. Generating synthetic multimodal data...")
    data, labels = generate_synthetic_data(n_samples=500)  # Smaller dataset for faster optimization
    
    split_idx = int(0.8 * len(labels))
    train_data = {k: v[:split_idx] for k, v in data.items()}
    train_labels = labels[:split_idx]
    val_data = {k: v[split_idx:] for k, v in data.items()}
    val_labels = labels[split_idx:]
    
    # Preprocess
    preprocessor = MultimodalPreprocessor()
    train_data = preprocessor.fit_transform(train_data)
    val_data = preprocessor.transform(val_data)
    
    input_sizes = {
        'text': train_data['text'].shape[2],
        'audio': train_data['audio'].shape[2],
        'visual': train_data['visual'].shape[2]
    }
    
    # Define hyperparameter search space
    print("\n2. Defining hyperparameter search space...")
    param_space = {
        'hidden_size': {'type': 'int', 'min': 64, 'max': 256},
        'num_layers': {'type': 'int', 'min': 1, 'max': 3},
        'dropout': {'type': 'float', 'min': 0.1, 'max': 0.5},
        'learning_rate': {'type': 'log', 'min': 1e-4, 'max': 1e-2},
        'batch_size': {'type': 'categorical', 'values': [16, 32, 64]}
    }
    
    # Define objective function
    def objective_function(params):
        """Objective function for EHOLF optimization."""
        try:
            # Create model with current hyperparameters
            model = MultimodalLSTMClassifier(
                input_sizes=input_sizes,
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                num_classes=7,
                dropout=params['dropout'],
                bidirectional=True
            )
            
            # Create data loaders
            data_loader = MultimodalDataLoader(
                batch_size=int(params['batch_size']),
                shuffle=True
            )
            train_loader, val_loader = data_loader.create_train_val_loaders(
                train_data, train_labels, val_data, val_labels
            )
            
            # Train model
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            trainer = Trainer(
                model=model,
                device=device,
                learning_rate=params['learning_rate'],
                weight_decay=1e-5
            )
            
            # Train for fewer epochs in optimization
            trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=5,
                early_stopping_patience=3,
                verbose=False
            )
            
            # Evaluate and return validation accuracy as fitness
            _, accuracy, _, _ = trainer.validate(val_loader, verbose=False)
            
            return accuracy
        
        except Exception as e:
            print(f"   Error in objective function: {e}")
            return 0.0
    
    # Run EHOLF optimization
    print("\n3. Running EHOLF optimization...")
    print("   This may take a few minutes...")
    
    eholf = EHOLF(
        param_space=param_space,
        population_size=10,  # Reduced for demo
        generations=5,  # Reduced for demo
        mutation_rate=0.2,
        crossover_rate=0.7,
        elite_size=2
    )
    
    best_params, best_fitness = eholf.optimize(
        objective_function=objective_function,
        verbose=True
    )
    
    print("\n4. Optimization Results:")
    print(f"   Best Validation Accuracy: {best_fitness:.4f}")
    print("   Best Hyperparameters:")
    for param, value in best_params.items():
        print(f"      {param}: {value}")
    
    # Train final model with best hyperparameters
    print("\n5. Training final model with optimized hyperparameters...")
    
    final_model = MultimodalLSTMClassifier(
        input_sizes=input_sizes,
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        num_classes=7,
        dropout=best_params['dropout'],
        bidirectional=True
    )
    
    data_loader = MultimodalDataLoader(
        batch_size=int(best_params['batch_size']),
        shuffle=True
    )
    train_loader, val_loader = data_loader.create_train_val_loaders(
        train_data, train_labels, val_data, val_labels
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(
        model=final_model,
        device=device,
        learning_rate=best_params['learning_rate'],
        weight_decay=1e-5
    )
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
        early_stopping_patience=5,
        verbose=True
    )
    
    # Final evaluation
    print("\n6. Final Evaluation:")
    _, _, y_true, y_pred = trainer.validate(val_loader, verbose=False)
    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics)
    
    return best_params, final_model, eholf


if __name__ == '__main__':
    print("\nMultimodal Emotion Recognition with LSTM and EHOLF")
    print("="*70)
    print("\nThis example demonstrates the usage of the multimodal emotion")
    print("recognition system with LSTM classifier and EHOLF hyperparameter")
    print("optimization on synthetic data.\n")
    
    # Run basic training example
    model, trainer, history = basic_training_example()
    
    # Run EHOLF optimization example
    best_params, optimized_model, eholf = eholf_optimization_example()
    
    print("\n" + "="*70)
    print("Examples completed successfully!")
    print("="*70)
