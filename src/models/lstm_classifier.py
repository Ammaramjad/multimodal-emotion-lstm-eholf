"""LSTM-based multimodal emotion classifier."""

import torch
import torch.nn as nn


class MultimodalLSTMClassifier(nn.Module):
    """
    Multimodal LSTM classifier for emotion recognition.
    
    This model processes multiple modalities (text, audio, visual) through
    separate LSTM layers and fuses them for emotion classification.
    
    Args:
        input_sizes (dict): Dictionary with modality names as keys and input dimensions as values
        hidden_size (int): Size of LSTM hidden state
        num_layers (int): Number of LSTM layers
        num_classes (int): Number of emotion classes
        dropout (float): Dropout rate
        bidirectional (bool): Whether to use bidirectional LSTM
    """
    
    def __init__(self, input_sizes, hidden_size=128, num_layers=2, 
                 num_classes=7, dropout=0.3, bidirectional=True):
        super(MultimodalLSTMClassifier, self).__init__()
        
        self.input_sizes = input_sizes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # Create LSTM for each modality
        self.modality_lstms = nn.ModuleDict()
        for modality, input_size in input_sizes.items():
            self.modality_lstms[modality] = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        
        # Calculate fusion layer input size
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        fusion_input_size = lstm_output_size * len(input_sizes)
        
        # Fusion and classification layers
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, inputs):
        """
        Forward pass through the model.
        
        Args:
            inputs (dict): Dictionary with modality names as keys and tensors as values
                          Each tensor shape: (batch_size, sequence_length, input_size)
        
        Returns:
            torch.Tensor: Logits for each class (batch_size, num_classes)
        """
        modality_outputs = []
        
        # Process each modality through its LSTM
        for modality, lstm in self.modality_lstms.items():
            if modality in inputs:
                x = inputs[modality]
                lstm_out, (hidden, cell) = lstm(x)
                
                # Use the last hidden state
                if self.bidirectional:
                    # Concatenate forward and backward hidden states
                    hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
                else:
                    hidden = hidden[-1]
                
                modality_outputs.append(hidden)
        
        # Concatenate all modality outputs
        fused = torch.cat(modality_outputs, dim=1)
        
        # Apply fusion layers
        fused = self.fusion(fused)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits
    
    def get_config(self):
        """Get model configuration."""
        return {
            'input_sizes': self.input_sizes,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional
        }
