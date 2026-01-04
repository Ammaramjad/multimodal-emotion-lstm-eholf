"""
Kernel SHAP wrapper for the trained model.
"""

import shap
import numpy as np
import torch

def model_predict_wrapper(model, device='cpu'):
    model.to(device)
    model.eval()
    def predict(x_numpy):
        x = torch.tensor(x_numpy, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
        return probs
    return predict

def compute_kernel_shap(model, background_X, samples_X, device='cpu', nsamples=100):
    bg = background_X[:, -1, :].reshape(background_X.shape[0], -1)
    samp = samples_X[:, -1, :].reshape(samples_X.shape[0], -1)
    predict_fn = model_predict_wrapper(model, device=device)
    explainer = shap.KernelExplainer(predict_fn, bg, link="identity")
    shap_values = explainer.shap_values(samp, nsamples=nsamples)
    return shap_values