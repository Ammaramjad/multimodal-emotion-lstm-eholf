# Multimodal Emotion Recognition with LSTM and EHOLF

A comprehensive framework for multimodal emotion recognition using LSTM-based deep learning and EHOLF (Evolved Hierarchical Optimization of Learned Features) hyperparameter optimization.

## Overview

This project implements a state-of-the-art multimodal emotion recognition system that:

- **Processes multiple modalities**: Combines text, audio, and visual features for emotion classification
- **Uses LSTM architecture**: Employs bidirectional LSTM networks to capture temporal dependencies
- **Optimizes hyperparameters**: Implements EHOLF, an evolutionary optimization algorithm for automated hyperparameter tuning
- **Supports 7 emotions**: Classifies neutral, happy, sad, angry, fear, disgust, and surprise

## Features

### 🎯 Multimodal Fusion
- Independent LSTM processing for each modality
- Late fusion strategy for combining multimodal features
- Flexible architecture supporting any combination of modalities

### 🧠 LSTM Classifier
- Bidirectional LSTM for better context understanding
- Configurable hidden size, layers, and dropout
- Batch normalization and regularization

### 🔧 EHOLF Optimization
- Evolutionary hyperparameter optimization
- Hierarchical search strategy
- Efficient population-based optimization
- Tournament selection and adaptive mutation

### 📊 Comprehensive Evaluation
- Accuracy, precision, recall, and F1-score metrics
- Per-class performance analysis
- Confusion matrix visualization support

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy, Pandas, Scikit-learn

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Ammaramjad/multimodal-emotion-lstm-eholf.git
cd multimodal-emotion-lstm-eholf
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from src.models import MultimodalLSTMClassifier
from src.data import MultimodalDataLoader, MultimodalPreprocessor
from src.utils import Config, calculate_metrics
from src.utils.trainer import Trainer
import torch

# Load configuration
config = Config()

# Define input sizes for each modality
input_sizes = {
    'text': 300,    # Text embedding dimension
    'audio': 40,    # Audio features (e.g., MFCC)
    'visual': 128   # Visual features
}

# Create model
model = MultimodalLSTMClassifier(
    input_sizes=input_sizes,
    hidden_size=128,
    num_layers=2,
    num_classes=7,
    dropout=0.3,
    bidirectional=True
)

# Prepare your data
# data should be a dict: {'text': array, 'audio': array, 'visual': array}
# labels should be a numpy array of emotion labels (0-6)

# Preprocess data
preprocessor = MultimodalPreprocessor(normalize=True, max_sequence_length=100)
train_data = preprocessor.fit_transform(train_data)
val_data = preprocessor.transform(val_data)

# Create data loaders
data_loader = MultimodalDataLoader(batch_size=32, shuffle=True)
train_loader, val_loader = data_loader.create_train_val_loaders(
    train_data, train_labels, val_data, val_labels
)

# Train model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainer = Trainer(model=model, device=device, learning_rate=0.001)
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    early_stopping_patience=10
)

# Evaluate
_, _, y_true, y_pred = trainer.validate(val_loader)
metrics = calculate_metrics(y_true, y_pred)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### Hyperparameter Optimization with EHOLF

```python
from src.optimization import EHOLF

# Define hyperparameter search space
param_space = {
    'hidden_size': {'type': 'int', 'min': 64, 'max': 256},
    'num_layers': {'type': 'int', 'min': 1, 'max': 3},
    'dropout': {'type': 'float', 'min': 0.1, 'max': 0.5},
    'learning_rate': {'type': 'log', 'min': 1e-4, 'max': 1e-2},
    'batch_size': {'type': 'categorical', 'values': [16, 32, 64]}
}

# Define objective function
def objective_function(params):
    # Create and train model with params
    # Return validation accuracy
    model = MultimodalLSTMClassifier(
        input_sizes=input_sizes,
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    )
    # ... train model ...
    return validation_accuracy

# Run optimization
eholf = EHOLF(
    param_space=param_space,
    population_size=20,
    generations=10,
    mutation_rate=0.2,
    crossover_rate=0.7
)

best_params, best_fitness = eholf.optimize(
    objective_function=objective_function,
    verbose=True
)

print(f"Best hyperparameters: {best_params}")
print(f"Best validation accuracy: {best_fitness:.4f}")
```

### Run Examples

Try the included example with synthetic data:

```bash
python examples/basic_usage.py
```

This will demonstrate:
1. Basic training without optimization
2. Hyperparameter optimization with EHOLF
3. Final model training with optimized parameters

## Project Structure

```
multimodal-emotion-lstm-eholf/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── lstm_classifier.py      # Multimodal LSTM model
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py          # Data loading utilities
│   │   └── preprocessor.py         # Data preprocessing
│   ├── optimization/
│   │   ├── __init__.py
│   │   └── eholf.py                # EHOLF optimizer
│   └── utils/
│       ├── __init__.py
│       ├── config.py               # Configuration management
│       ├── metrics.py              # Evaluation metrics
│       └── trainer.py              # Training utilities
├── examples/
│   └── basic_usage.py              # Example scripts
├── requirements.txt
├── README.md
└── LICENSE
```

## Configuration

The system uses a flexible configuration system. Default configuration:

```python
{
    'model': {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'bidirectional': True,
        'num_classes': 7
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 50,
        'weight_decay': 1e-5,
        'early_stopping_patience': 10
    },
    'data': {
        'max_sequence_length': 100,
        'normalize': True,
        'train_val_split': 0.2
    },
    'optimization': {
        'population_size': 20,
        'generations': 10,
        'mutation_rate': 0.2,
        'crossover_rate': 0.7,
        'elite_size': 2
    }
}
```

## Data Format

### Input Data Structure

Your data should be organized as follows:

```python
data = {
    'text': np.array,     # Shape: (n_samples, sequence_length, text_features)
    'audio': np.array,    # Shape: (n_samples, sequence_length, audio_features)
    'visual': np.array    # Shape: (n_samples, sequence_length, visual_features)
}
labels = np.array        # Shape: (n_samples,) with values 0-6
```

### Emotion Labels

- 0: Neutral
- 1: Happy
- 2: Sad
- 3: Angry
- 4: Fear
- 5: Disgust
- 6: Surprise

## EHOLF Algorithm

EHOLF (Evolved Hierarchical Optimization of Learned Features) is an evolutionary algorithm for hyperparameter optimization that:

1. **Initializes** a population of random hyperparameter configurations
2. **Evaluates** fitness (validation accuracy) for each configuration
3. **Selects** parents using tournament selection
4. **Generates** offspring through crossover and mutation
5. **Preserves** elite individuals across generations
6. **Repeats** until convergence or generation limit

### Key Features:
- **Hierarchical search**: Explores hyperparameter space at multiple scales
- **Adaptive mutation**: Gaussian mutation with parameter-specific strategies
- **Elite preservation**: Maintains best solutions across generations
- **Tournament selection**: Balances exploration and exploitation

## Performance

The system achieves competitive performance on emotion recognition tasks:

- Fast training with GPU support
- Efficient memory usage with batch processing
- Early stopping to prevent overfitting
- Learning rate scheduling for better convergence

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{multimodal_emotion_lstm_eholf,
  title={Multimodal Emotion Recognition with LSTM and EHOLF},
  author={Ammar Amjad},
  year={2026},
  url={https://github.com/Ammaramjad/multimodal-emotion-lstm-eholf}
}
```

## Acknowledgments

- PyTorch team for the deep learning framework
- Scikit-learn for evaluation metrics
- The emotion recognition research community

## Contact

For questions or issues, please open an issue on GitHub.
