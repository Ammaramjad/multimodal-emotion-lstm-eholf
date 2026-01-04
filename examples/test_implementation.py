"""
Simple test to validate the implementation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
from models import MultimodalLSTMClassifier
from data import MultimodalDataLoader, MultimodalPreprocessor
from optimization import EHOLF
from utils import Config, calculate_metrics
from utils.trainer import Trainer


def test_basic_components():
    """Test basic components."""
    print("="*50)
    print("Testing Basic Components")
    print("="*50)
    
    # Test config
    config = Config()
    print(f"✓ Config loaded")
    
    # Test model
    input_sizes = {'text': 50, 'audio': 20, 'visual': 30}
    model = MultimodalLSTMClassifier(
        input_sizes=input_sizes,
        hidden_size=32,
        num_layers=1,
        num_classes=7
    )
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test preprocessor
    data = {
        'text': np.random.randn(50, 20, 50),
        'audio': np.random.randn(50, 20, 20),
        'visual': np.random.randn(50, 20, 30)
    }
    preprocessor = MultimodalPreprocessor()
    processed_data = preprocessor.fit_transform(data)
    print(f"✓ Preprocessor works")
    
    # Test data loader
    labels = np.random.randint(0, 7, 50)
    loader = MultimodalDataLoader(batch_size=8)
    data_loader = loader.create_loader(processed_data, labels)
    print(f"✓ Data loader created")
    
    # Test EHOLF
    param_space = {
        'x': {'type': 'float', 'min': 0.0, 'max': 1.0}
    }
    eholf = EHOLF(param_space, population_size=5, generations=2)
    print(f"✓ EHOLF optimizer created")
    
    print("\n✅ All basic component tests passed!\n")


def test_training():
    """Test training workflow."""
    print("="*50)
    print("Testing Training Workflow")
    print("="*50)
    
    # Generate small data
    n_samples = 60
    seq_len = 15
    data = {
        'text': np.random.randn(n_samples, seq_len, 30),
        'audio': np.random.randn(n_samples, seq_len, 15),
    }
    labels = np.random.randint(0, 7, n_samples)
    
    # Add patterns to make it learnable
    for i in range(n_samples):
        data['text'][i] += labels[i] * 0.2
        data['audio'][i] += labels[i] * 0.2
    
    # Split
    split = int(0.8 * n_samples)
    train_data = {k: v[:split] for k, v in data.items()}
    train_labels = labels[:split]
    val_data = {k: v[split:] for k, v in data.items()}
    val_labels = labels[split:]
    
    print(f"Data: {len(train_labels)} train, {len(val_labels)} val samples")
    
    # Preprocess
    preprocessor = MultimodalPreprocessor(max_sequence_length=seq_len)
    train_data = preprocessor.fit_transform(train_data)
    val_data = preprocessor.transform(val_data)
    
    # Create loaders
    loader = MultimodalDataLoader(batch_size=8)
    train_loader, val_loader = loader.create_train_val_loaders(
        train_data, train_labels, val_data, val_labels
    )
    
    # Create model
    input_sizes = {k: v.shape[2] for k, v in train_data.items()}
    model = MultimodalLSTMClassifier(
        input_sizes=input_sizes,
        hidden_size=32,
        num_layers=1,
        num_classes=7,
        dropout=0.2
    )
    
    # Train
    trainer = Trainer(model=model, device='cpu', learning_rate=0.01)
    print("Training for 5 epochs...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5,
        early_stopping_patience=10,
        verbose=False
    )
    
    print(f"✓ Training completed")
    print(f"  Initial train loss: {history['train_losses'][0]:.4f}")
    print(f"  Final train loss: {history['train_losses'][-1]:.4f}")
    print(f"  Final val accuracy: {history['val_accuracies'][-1]:.4f}")
    
    # Evaluate
    _, acc, y_true, y_pred = trainer.validate(val_loader, verbose=False)
    metrics = calculate_metrics(y_true, y_pred)
    print(f"✓ Final accuracy: {metrics['accuracy']:.4f}")
    
    print("\n✅ Training workflow test passed!\n")


def test_eholf():
    """Test EHOLF optimization."""
    print("="*50)
    print("Testing EHOLF Optimization")
    print("="*50)
    
    # Simple optimization problem
    param_space = {
        'x': {'type': 'float', 'min': -5.0, 'max': 5.0},
        'y': {'type': 'int', 'min': -5, 'max': 5},
        'lr': {'type': 'log', 'min': 1e-4, 'max': 1e-1},
        'mode': {'type': 'categorical', 'values': ['a', 'b', 'c']}
    }
    
    # Objective: maximize negative squared distance from (2, 3)
    def objective(params):
        return -((params['x'] - 2.0)**2 + (params['y'] - 3.0)**2)
    
    eholf = EHOLF(
        param_space=param_space,
        population_size=8,
        generations=5,
        mutation_rate=0.3,
        crossover_rate=0.7
    )
    
    print("Running optimization...")
    best_params, best_fitness = eholf.optimize(objective, verbose=False)
    
    print(f"✓ Optimization completed")
    print(f"  Best fitness: {best_fitness:.4f}")
    print(f"  Best x: {best_params['x']:.2f} (target: 2.0)")
    print(f"  Best y: {best_params['y']} (target: 3)")
    print(f"  Parameter types correct: x={type(best_params['x']).__name__}, "
          f"y={type(best_params['y']).__name__}, lr={type(best_params['lr']).__name__}")
    
    # Verify types
    assert isinstance(best_params['x'], float)
    assert isinstance(best_params['y'], int)
    assert isinstance(best_params['lr'], float)
    
    print("\n✅ EHOLF optimization test passed!\n")


if __name__ == '__main__':
    print("\n" + "="*50)
    print("Multimodal Emotion Recognition Test Suite")
    print("="*50 + "\n")
    
    test_basic_components()
    test_training()
    test_eholf()
    
    print("="*50)
    print("✅ All tests passed successfully!")
    print("="*50)
