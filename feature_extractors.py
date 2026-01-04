"""
Multimodal feature extractors:
- Text: BERT CLS -> PCA projection optionally
- Audio: MFCC(13)+delta+delta-delta + spectral descriptors aggregated
- Video: ResNet50 deep embedding (2048) + MediaPipe geometric descriptors (10)
"""

import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
import librosa
import cv2
from torchvision import models, transforms

# Import mediapipe_landmarks (optional)
try:
    from mediapipe_landmarks import frame_geometric_and_deep
    _HAS_MEDIAPIPE = True
except Exception:
    _HAS_MEDIAPIPE = False

class TextFeatureExtractor:
    def __init__(self, device='cpu', pca_components=12):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased").to(device)
        self.bert.eval()
        self.pca = PCA(n_components=pca_components)
        self._fit_pca = False

    def encode_utterance(self, text):
        toks = self.tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        toks = {k: v.to(self.device) for k, v in toks.items()}
        with torch.no_grad():
            out = self.bert(**toks, return_dict=True)
        cls = out.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
        return cls

    def fit_pca(self, cls_matrix):
        self.pca.fit(cls_matrix)
        self._fit_pca = True

    def transform(self, cls_vector):
        if not self._fit_pca:
            raise RuntimeError("PCA not fitted.")
        return self.pca.transform(cls_vector.reshape(1, -1)).squeeze()

class AudioFeatureExtractor:
    def __init__(self, sr=16000, n_mfcc=13):
        self.sr = sr
        self.n_mfcc = n_mfcc

    def extract_frame_level(self, audio_path):
        y, sr = librosa.load(audio_path, sr=self.sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        d1 = librosa.feature.delta(mfcc)
        d2 = librosa.feature.delta(mfcc, order=2)
        mfcc_all = np.vstack([mfcc, d1, d2])  # (39, T)
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_flux = librosa.onset.onset_strength(y=y, sr=sr)
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features = []
        for arr in [mfcc_all, spec_centroid, spec_flux[np.newaxis, :], spec_rolloff]:
            mean = np.mean(arr, axis=1)
            std = np.std(arr, axis=1)
            features.append(np.concatenate([mean, std]))
        feat = np.concatenate(features)
        return feat

class VideoResNetEmbedder:
    def __init__(self, device='cpu'):
        self.device = device
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.backbone = torch.nn.Sequential(*modules).to(device).eval()
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def deep_embedding(self, frame_bgr):
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.backbone(inp)
        feat = out.squeeze().cpu().numpy().reshape(-1)
        return feat

class VideoFeatureExtractor:
    def __init__(self, device='cpu'):
        self.embedder = VideoResNetEmbedder(device=device)

    def extract_frame(self, frame_bgr):
        deep = self.embedder.deep_embedding(frame_bgr)
        # geometric via mediapipe if available
        try:
            from mediapipe_landmarks import geometric_features_from_landmarks, extract_face_mesh_landmarks
            geom = geometric_features_from_landmarks(extract_face_mesh_landmarks(frame_bgr))
        except Exception:
            # fallback zeros
            geom = np.zeros(10, dtype=float)
        return np.concatenate([deep, geom])