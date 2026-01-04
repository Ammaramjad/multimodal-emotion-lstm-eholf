"""
Feature caching utilities to save/load precomputed per-utterance features.
"""

import os
import time
import joblib
import hashlib

def _uid_from_path(path: str) -> str:
    return hashlib.md5(path.encode('utf-8')).hexdigest()

def save_feature_cache(utterance_id: str, features: dict, meta: dict, cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, f"{utterance_id}.pkl")
    payload = {'utterance_id': utterance_id, 'features': features, 'meta': meta, 'cached_at': time.time()}
    joblib.dump(payload, fname, compress=('gzip', 3))
    return fname

def load_feature_cache(utterance_id: str, cache_dir: str):
    fname = os.path.join(cache_dir, f"{utterance_id}.pkl")
    if not os.path.exists(fname):
        return None
    return joblib.load(fname)

def is_cache_fresh(utterance, cache_dir: str):
    uid = _uid_from_path(utterance.get('wav', '') + (utterance.get('video') or ''))
    entry = load_feature_cache(uid, cache_dir)
    if entry is None:
        return False
    meta = entry.get('meta', {})
    for key in ['wav', 'video', 'ann']:
        path = utterance.get(key)
        if not path:
            continue
        if not os.path.exists(path):
            return False
        cur_m = os.path.getmtime(path)
        cached_m = meta.get(f"{key}_mtime", None)
        if cached_m is None or abs(cur_m - cached_m) > 1.0:
            return False
    return True

def build_cache_from_list(utterance_list, extractor_fn, cache_dir: str, n_jobs: int = 1, overwrite=False):
    from joblib import Parallel, delayed
    def _process(u):
        uid = _uid_from_path(u.get('wav', '') + (u.get('video') or ''))
        out_path = os.path.join(cache_dir, f"{uid}.pkl")
        if os.path.exists(out_path) and (not overwrite):
            return out_path
        res = extractor_fn(u)
        meta = res.get('meta', {})
        for k in ['wav', 'video', 'ann']:
            if u.get(k) and os.path.exists(u.get(k)):
                meta[f"{k}_mtime"] = os.path.getmtime(u.get(k))
        save_feature_cache(uid, res.get('features', {}), meta, cache_dir)
        return out_path
    if n_jobs == 1:
        out = [_process(u) for u in utterance_list]
    else:
        out = Parallel(n_jobs=n_jobs)(delayed(_process)(u) for u in utterance_list)
    return out