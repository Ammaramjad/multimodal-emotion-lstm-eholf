"""
Time-resolved SHAP utilities (Kernel SHAP).
"""

import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

def compute_time_resolved_shap(model, background_X, samples_X, modality_slices, device='cpu', nsamples=100, flatten_mode='flatten'):
    n_bg, T, D = background_X.shape
    n_s = samples_X.shape[0]
    if flatten_mode == 'flatten':
        bg_flat = background_X.reshape(n_bg, T * D)
        samp_flat = samples_X.reshape(n_s, T * D)
        def predict_flat(x_flat):
            arr = np.asarray(x_flat)
            if arr.ndim == 2:
                arr3 = arr.reshape(arr.shape[0], T, D)
            elif arr.ndim == 3:
                arr3 = arr
            else:
                raise ValueError("Unexpected shape")
            import torch
            X = torch.tensor(arr3, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits = model(X)
                probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
            return probs
        explainer = shap.KernelExplainer(predict_flat, bg_flat, link="identity")
        shap_vals = explainer.shap_values(samp_flat, nsamples=nsamples)
        C = len(shap_vals)
        shap_per_class = []
        for c in range(C):
            arr = np.array(shap_vals[c]).reshape(n_s, T, D)
            shap_per_class.append(arr)
        base_values = explainer.expected_value
        return shap_per_class, base_values
    elif flatten_mode == 'timewise':
        shap_per_class = None
        base_values = None
        for t in range(T):
            bg_t = background_X[:, t, :]
            samp_t = samples_X[:, t, :]
            def predict_t(x_tab):
                N = x_tab.shape[0]
                arr = np.zeros((N, T, D), dtype=float)
                arr[:, t, :] = x_tab
                import torch
                X = torch.tensor(arr, dtype=torch.float32).to(device)
                with torch.no_grad():
                    logits = model(X)
                    probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
                return probs
            expl = shap.KernelExplainer(predict_t, bg_t, link="identity")
            vals = expl.shap_values(samp_t, nsamples=nsamples)
            C = len(vals)
            if shap_per_class is None:
                shap_per_class = [np.zeros((n_s, T, D), dtype=float) for _ in range(C)]
            for c in range(C):
                shap_per_class[c][:, t, :] = np.array(vals[c])
            base_values = expl.expected_value
        return shap_per_class, base_values
    else:
        raise ValueError("flatten_mode must be 'flatten' or 'timewise'")

def aggregate_modality_shap(shap_per_class, modality_slices):
    shap_mean = np.mean([np.abs(arr) for arr in shap_per_class], axis=0)
    n_s, T, D = shap_mean.shape
    out = {}
    for m, (s, e) in modality_slices.items():
        out[m] = np.sum(shap_mean[:, :, s:e], axis=2)
    return out

def plot_modality_timecourse(modality_importance: dict, sample_idx: int = 0, figsize=(10,4)):
    plt.figure(figsize=figsize)
    for m, arr in modality_importance.items():
        y = arr[sample_idx]
        plt.plot(y, label=m)
    plt.xlabel("Time-step")
    plt.ylabel("Aggregated |SHAP|")
    plt.legend()
    plt.title(f"Modality importance timecourse (sample {sample_idx})")
    plt.tight_layout()
    plt.show()

def plot_shap_heatmap(shap_array, t_idx=0, feature_names=None, vmax=None):
    arr = np.array(shap_array)
    if arr.ndim == 3:
        arr_t = np.mean(np.abs(arr[:, t_idx, :]), axis=0)
    else:
        arr_t = np.abs(arr[t_idx])
    plt.figure(figsize=(10, 3))
    sns.heatmap(arr_t[np.newaxis, :], cmap='viridis', cbar=True, vmax=vmax)
    if feature_names:
        plt.xticks(np.arange(len(feature_names))+0.5, feature_names, rotation=90)
    plt.yticks([])
    plt.title(f"Mean |SHAP| at time {t_idx}")
    plt.tight_layout()
    plt.show()