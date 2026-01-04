"""
Improved BiLSTM with temporal attention pooling.
"""

import torch
import torch.nn as nn

class TemporalAttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, seq, mask=None):
        scores = self.attn(seq).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = torch.sum(seq * weights, dim=1)
        return context, weights

class BiLSTMAttentionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_size=128, num_layers=2, dropout=0.3, num_classes=4, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.attn_pool = TemporalAttentionPooling(hidden_size * self.num_directions)
        self.norm = nn.LayerNorm(hidden_size * self.num_directions)
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        out, _ = self.lstm(x)
        mask = None
        if lengths is not None:
            maxlen = x.size(1)
            device = x.device
            lengths = lengths.to(device)
            idx = torch.arange(maxlen, device=device).unsqueeze(0)
            mask = (idx < lengths.unsqueeze(1))
        context, weights = self.attn_pool(out, mask=mask)
        context = self.norm(context)
        context = self.dropout(context)
        logits = self.fc(context)
        return logits