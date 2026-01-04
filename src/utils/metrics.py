"""Evaluation metrics for emotion recognition."""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, Tuple


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     average: str = 'weighted') -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method for multi-class metrics
    
    Returns:
        Dictionary of metric names and values
    """
    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate precision, recall, and F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    
    # Calculate per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float], emotion_labels: list = None):
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics
        emotion_labels: List of emotion label names
    """
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    
    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    if 'precision_per_class' in metrics:
        print("\nPer-class Metrics:")
        print("-"*50)
        
        if emotion_labels is None:
            emotion_labels = [f"Class {i}" for i in range(len(metrics['precision_per_class']))]
        
        for i, label in enumerate(emotion_labels):
            if i < len(metrics['precision_per_class']):
                print(f"{label:12s} - P: {metrics['precision_per_class'][i]:.4f}, "
                      f"R: {metrics['recall_per_class'][i]:.4f}, "
                      f"F1: {metrics['f1_per_class'][i]:.4f}")
    
    print("="*50 + "\n")
