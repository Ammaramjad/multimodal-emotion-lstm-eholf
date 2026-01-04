# Quick Start Guide

This guide will help you get started with the Multimodal Emotion Recognition system.

## Installation

```bash
# Clone the repository
git clone https://github.com/Ammaramjad/multimodal-emotion-lstm-eholf.git
cd multimodal-emotion-lstm-eholf

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Quick Example

```python
import sys
sys.path.insert(0, 'src')

import numpy as np
import torch
from models import MultimodalLSTMClassifier
from data import MultimodalDataLoader, MultimodalPreprocessor
from utils.trainer import Trainer

# Prepare your multimodal data
# Shape: (n_samples, sequence_length, feature_dim)
data = {
    'text': np.random.randn(1000, 50, 300),    # Text embeddings
    'audio': np.random.randn(1000, 50, 40),    # Audio features (MFCC)
    'visual': np.random.randn(1000, 50, 128)   # Visual features
}
labels = np.random.randint(0, 7, 1000)  # Emotion labels (0-6)

# Split into train/val
split = int(0.8 * len(labels))
train_data = {k: v[:split] for k, v in data.items()}
train_labels = labels[:split]
val_data = {k: v[split:] for k, v in data.items()}
val_labels = labels[split:]

# Preprocess
preprocessor = MultimodalPreprocessor(normalize=True)
train_data = preprocessor.fit_transform(train_data)
val_data = preprocessor.transform(val_data)

# Create data loaders
loader = MultimodalDataLoader(batch_size=32)
train_loader, val_loader = loader.create_train_val_loaders(
    train_data, train_labels, val_data, val_labels
)

# Create model
input_sizes = {k: v.shape[2] for k, v in train_data.items()}
model = MultimodalLSTMClassifier(
    input_sizes=input_sizes,
    hidden_size=128,
    num_layers=2,
    num_classes=7,
    dropout=0.3
)

# Train
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainer = Trainer(model=model, device=device, learning_rate=0.001)
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    early_stopping_patience=10
)

# Save model
trainer.save_model('emotion_model.pth')
```

## Using EHOLF for Hyperparameter Optimization

```python
from optimization import EHOLF

# Define hyperparameter search space
param_space = {
    'hidden_size': {'type': 'int', 'min': 64, 'max': 256},
    'num_layers': {'type': 'int', 'min': 1, 'max': 3},
    'dropout': {'type': 'float', 'min': 0.1, 'max': 0.5},
    'learning_rate': {'type': 'log', 'min': 1e-4, 'max': 1e-2},
    'batch_size': {'type': 'categorical', 'values': [16, 32, 64]}
}

# Define objective function
def train_and_evaluate(params):
    # Create model with params
    model = MultimodalLSTMClassifier(
        input_sizes=input_sizes,
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    )
    
    # Create data loader with params
    loader = MultimodalDataLoader(batch_size=int(params['batch_size']))
    train_loader, val_loader = loader.create_train_val_loaders(
        train_data, train_labels, val_data, val_labels
    )
    
    # Train
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=params['learning_rate']
    )
    trainer.train(train_loader, val_loader, num_epochs=10, verbose=False)
    
    # Return validation accuracy
    _, accuracy, _, _ = trainer.validate(val_loader, verbose=False)
    return accuracy

# Run optimization
eholf = EHOLF(
    param_space=param_space,
    population_size=20,
    generations=10
)
best_params, best_score = eholf.optimize(train_and_evaluate)

print(f"Best validation accuracy: {best_score:.4f}")
print(f"Best hyperparameters: {best_params}")
```

## Running Examples

```bash
# Run the comprehensive example (takes a few minutes)
python examples/basic_usage.py

# Run quick tests
python examples/test_implementation.py
```

## Data Format

Your data should be organized as dictionaries with modality names as keys:

```python
data = {
    'text': np.array,     # (n_samples, seq_length, text_dim)
    'audio': np.array,    # (n_samples, seq_length, audio_dim)
    'visual': np.array    # (n_samples, seq_length, visual_dim)
}
labels = np.array        # (n_samples,) with values 0-6
```

### Emotion Labels

- 0: Neutral
- 1: Happy
- 2: Sad
- 3: Angry
- 4: Fear
- 5: Disgust
- 6: Surprise

## Tips

1. **Start small**: Test with a small dataset first to ensure everything works
2. **GPU acceleration**: Use CUDA if available for faster training
3. **Early stopping**: Enable early stopping to prevent overfitting
4. **Hyperparameter tuning**: Use EHOLF to find optimal hyperparameters
5. **Preprocessing**: Always normalize your features for better convergence

## Next Steps

- Check out the [full README](README.md) for detailed documentation
- Explore the [examples](examples/) directory for more usage patterns
- Read the code documentation for API details

## Troubleshooting

**Out of memory**: Reduce batch size or model hidden size

**Poor performance**: Try:
- Increasing model capacity (hidden_size, num_layers)
- Using EHOLF to optimize hyperparameters
- Training for more epochs
- Checking data preprocessing

**Slow training**: 
- Use GPU if available
- Increase batch size
- Reduce model complexity

For more help, open an issue on GitHub.
