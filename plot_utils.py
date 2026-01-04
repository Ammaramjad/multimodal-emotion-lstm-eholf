"""
Simple plotting utilities used by SHAP/time-resolved modules.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_bar_modality_totals(modality_importance: dict, figsize=(6,4)):
    mods = list(modality_importance.keys())
    vals = [np.mean(v) for v in modality_importance.values()]
    plt.figure(figsize=figsize)
    sns.barplot(x=mods, y=vals)
    plt.ylabel("Mean aggregated |SHAP|")
    plt.title("Global modality importance")
    plt.tight_layout()
    plt.show()

def plot_time_heatmap(arr, x_labels=None, y_label="Feature index", title="Time heatmap", figsize=(10,4)):
    arr = np.asarray(arr)
    plt.figure(figsize=figsize)
    sns.heatmap(arr.T, cmap='viridis', cbar=True)
    if x_labels is not None:
        plt.xticks(np.arange(len(x_labels))+0.5, x_labels, rotation=90)
    plt.xlabel("Time-step")
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.show()