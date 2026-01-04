"""
Training and evaluation pipeline, plus a simple EHOLF evaluation function wrapper.
- MultimodalSequenceDataset (list of sequences)
- collate_fn for padding
- evaluate_fn generator to be used by EHOLF (short training)
- run_pipeline: preprocess caches, run EHOLF, train final model
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from improved_model import BiLSTMAttentionClassifier
from feature_cache import load_feature_cache, build_cache_from_list
from feature_extractors import TextFeatureExtractor, AudioFeatureExtractor, VideoFeatureExtractor
from parser_iemocap import load_iemocap_metadata
from utils import zscore_normalize
import os

class MultimodalSequenceDataset(Dataset):
    def __init__(self, X_sequences, y):
        self.X = [torch.tensor(x, dtype=torch.float32) for x in X_sequences]
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def collate_fn(batch):
    xs, ys = zip(*batch)
    lengths = [x.shape[0] for x in xs]
    maxlen = max(lengths)
    feat_dim = xs[0].shape[1]
    padded = torch.zeros(len(xs), maxlen, feat_dim, dtype=torch.float32)
    for i, x in enumerate(xs):
        padded[i, :x.shape[0], :] = x
    return padded, torch.stack(ys), torch.tensor(lengths)

def train_one_epoch(model, loader, criterion, optimizer, device='cpu'):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y, lengths in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total

def eval_model(model, loader, criterion, device='cpu'):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, lengths in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x, lengths)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total

def make_evaluate_fn(train_data, val_data, input_dim, num_classes, device='cpu', epochs=3):
    def evaluate(hparams):
        model = BiLSTMAttentionClassifier(input_dim=input_dim, hidden_size=int(hparams['hidden']),
                                          num_layers=1, dropout=float(hparams['dropout']), num_classes=num_classes)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=float(hparams['lr']), weight_decay=float(hparams['l2']))
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=collate_fn)
        best_val_loss = float('inf')
        for epoch in range(epochs):
            train_one_epoch(model, train_loader, criterion, optimizer, device=device)
            val_loss, val_acc = eval_model(model, val_loader, criterion, device=device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        return best_val_loss
    return evaluate

def example_extractor_fn(utterance, text_extractor=None, audio_extractor=None, video_extractor=None):
    """
    Given utterance metadata dict with 'wav', 'video', 'text', returns features dict and meta.
    Produces a single time-step fused vector (1, D). For sequence modelling, modify to per-frame vectors.
    """
    if text_extractor is None:
        text_extractor = TextFeatureExtractor(device='cpu')
    if audio_extractor is None:
        audio_extractor = AudioFeatureExtractor()
    if video_extractor is None:
        video_extractor = VideoFeatureExtractor(device='cpu')

    text_vec = text_extractor.encode_utterance(utterance.get('text', '') or "")
    audio_vec = audio_extractor.extract_frame_level(utterance.get('wav'))
    # sample some frames for video
    vid = utterance.get('video')
    if vid and os.path.exists(vid):
        import cv2
        cap = cv2.VideoCapture(vid)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames = []
        if total_frames > 0:
            sample_idxs = np.linspace(0, total_frames-1, num=min(4, total_frames), dtype=int)
            for idx in sample_idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret:
                    continue
                frames.append(frame)
        cap.release()
        if frames:
            feats = [video_extractor.extract_frame(f) for f in frames]
            video_vec = np.mean(feats, axis=0)
        else:
            video_vec = np.zeros(2048+10, dtype=float)
    else:
        video_vec = np.zeros(2048+10, dtype=float)

    fused = np.concatenate([text_vec, audio_vec, video_vec])
    return {'features': {'fused': fused}, 'meta': {'wav': utterance.get('wav'), 'video': vid, 'text': utterance.get('text')}}

def build_fused_dataset_from_cache(utterances, cache_dir, modal_key='fused'):
    X, y = [], []
    for u in utterances:
        uid = __import__('hashlib').md5((u.get('wav', '') + (u.get('video') or '')).encode('utf-8')).hexdigest()
        entry = load_feature_cache(uid, cache_dir)
        if entry is None:
            continue
        feat = entry['features'].get(modal_key)
        if feat is None:
            continue
        # produce a single time-step sequence (1, D)
        X.append(feat.reshape(1, -1))
        lab = u.get('label')
        if lab is None:
            lab = 0
        y.append(int(lab))
    return X, np.array(y)