"""
Quick runnable example:
- Creates synthetic data and runs EHOLF search + training to validate the pipeline.
"""

import numpy as np
from eholf_optimizer import EHOLFOptimizer
from train import MultimodalSequenceDataset, collate_fn, train_one_epoch, eval_model
from improved_model import BiLSTMAttentionClassifier
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

def make_synthetic_dataset(n_samples=300, seq_len=8, feat_dim=300, num_classes=4):
    X = [np.random.randn(seq_len, feat_dim).astype(np.float32) for _ in range(n_samples)]
    y = np.random.randint(0, num_classes, size=(n_samples,))
    return X, y

def evaluate_short(hparams, X_train, y_train, X_val, y_val, input_dim, num_classes, device='cpu'):
    train_ds = MultimodalSequenceDataset(X_train, y_train)
    val_ds = MultimodalSequenceDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    model = BiLSTMAttentionClassifier(input_dim=input_dim, hidden_size=int(hparams['hidden']), num_layers=1,
                                      dropout=float(hparams['dropout']), num_classes=num_classes)
    device = torch.device(device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(hparams['lr']), weight_decay=float(hparams['l2']))
    criterion = nn.CrossEntropyLoss()
    # one epoch quick
    train_one_epoch(model, train_loader, criterion, opt, device=device)
    val_loss, val_acc = eval_model(model, val_loader, criterion, device=device)
    return val_loss

def main():
    X, y = make_synthetic_dataset(n_samples=300, seq_len=8, feat_dim=300, num_classes=4)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    tr_idx = idx[:200]; val_idx = idx[200:250]; te_idx = idx[250:]
    X_tr = [X[i] for i in tr_idx]; y_tr = y[tr_idx]
    X_val = [X[i] for i in val_idx]; y_val = y[val_idx]
    X_te = [X[i] for i in te_idx]; y_te = y[te_idx]
    input_dim = X_tr[0].shape[1]; num_classes = 4

    bounds = {
        'lr': (1e-4, 1e-2, 'float'),
        'l2': (1e-6, 1e-3, 'float'),
        'dropout': (0.1, 0.5, 'float'),
        'hidden': (64, 256, 'int'),
    }
    opt = EHOLFOptimizer(bounds=bounds, pop_size=6, max_iter=8)
    def eval_fn(hp):
        return evaluate_short(hp, X_tr, y_tr, X_val, y_val, input_dim, num_classes, device='cpu')

    best_hp, best_score = opt.optimize(eval_fn, verbose=True)
    print("EHOLF best:", best_hp, "score:", best_score)

    # Train final
    model = BiLSTMAttentionClassifier(input_dim=input_dim, hidden_size=int(best_hp['hidden']), dropout=float(best_hp['dropout']), num_classes=num_classes)
    device = torch.device('cpu')
    model.to(device)
    optm = torch.optim.Adam(model.parameters(), lr=float(best_hp['lr']), weight_decay=float(best_hp['l2']))
    criterion = nn.CrossEntropyLoss()
    train_ds = MultimodalSequenceDataset([*X_tr, *X_val], np.concatenate([y_tr, y_val]))
    test_ds = MultimodalSequenceDataset(X_te, y_te)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    for epoch in range(1, 11):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optm, device=device)
        test_loss, test_acc = eval_model(model, test_loader, criterion, device=device)
        print(f"Epoch {epoch}: train_acc={train_acc:.3f} test_acc={test_acc:.3f}")

if __name__ == '__main__':
    main()