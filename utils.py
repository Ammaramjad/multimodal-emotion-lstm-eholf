"""
Utility helpers: normalization, interpolation-based alignment, simple helpers
"""

import numpy as np

def zscore_normalize(X, eps=1e-8):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma[sigma < eps] = 1.0
    return (X - mu) / sigma, mu, sigma

def linear_interpolate_timestamps(src_times, src_feats, target_times):
    """
    src_times: (T_src,) timestamps (seconds)
    src_feats: (T_src, D)
    target_times: (T_tgt,)
    returns (T_tgt, D) interpolated features
    """
    src_times = np.asarray(src_times)
    src_feats = np.asarray(src_feats)
    target_times = np.asarray(target_times)
    D = src_feats.shape[1]
    out = np.zeros((len(target_times), D), dtype=float)
    for d in range(D):
        out[:, d] = np.interp(target_times, src_times, src_feats[:, d])
    return out