"""Configuration management."""

import json
from typing import Dict, Any


class Config:
    """
    Configuration manager for the emotion recognition system.
    
    Args:
        config_dict (dict): Configuration dictionary
    """
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        self.config = config_dict or self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
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
            },
            'emotions': ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by key."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        def _deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict:
                    _deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        _deep_update(self.config, updates)
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.config.copy()
