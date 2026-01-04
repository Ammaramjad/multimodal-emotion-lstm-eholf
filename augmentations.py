"""
Data augmentation helpers for audio, video, text.
"""

import numpy as np
import librosa
import random
import cv2

def add_noise(y, snr_db=10):
    rms = np.sqrt(np.mean(y**2))
    noise_rms = rms / (10**(snr_db / 20.0))
    noise = np.random.normal(0, noise_rms, size=y.shape)
    return y + noise

def time_stretch(y, rate=1.0):
    return librosa.effects.time_stretch(y, rate)

def pitch_shift(y, sr, n_steps):
    return librosa.effects.pitch_shift(y, sr, n_steps)

def video_jitter_frame_drop(frames, drop_prob=0.1):
    out = []
    for f in frames:
        if random.random() < drop_prob:
            if len(out) > 0:
                out.append(out[-1])
            else:
                out.append(np.zeros_like(f))
        else:
            out.append(f)
    return out

def text_word_dropout(text, p=0.1):
    words = text.split()
    out = [w for w in words if random.random() > p]
    if not out:
        return text
    return " ".join(out)