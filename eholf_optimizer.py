"""
EHOLF hyperparameter optimizer (single-process).
Implements Lévy Flight exploration and Adaptive Inertia Weight exploitation.
"""

import numpy as np
import math
import copy
import random

def levy_step(beta=1.5, size=1):
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
               (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1.0 / beta)
    u = np.random.normal(0, sigma_u, size=size)
    v = np.random.normal(0, 1, size=size)
    step = u / (np.abs(v) ** (1.0 / beta))
    return step

class EHOLFOptimizer:
    def __init__(self, bounds, pop_size=12, max_iter=50, aw_min=0.3, aw_max=0.9, gamma=2.0, seed=None):
        self.bounds = bounds
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.aw_min = aw_min
        self.aw_max = aw_max
        self.gamma = gamma
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def _sample_random(self):
        hp = {}
        for k, (lo, hi, typ) in self.bounds.items():
            if typ == 'int':
                hp[k] = int(np.random.randint(lo, hi + 1))
            else:
                hp[k] = float(np.random.uniform(lo, hi))
        return hp

    def _clip(self, hp):
        chp = {}
        for k, (lo, hi, typ) in self.bounds.items():
            v = hp[k]
            if v < lo: v = lo
            if v > hi: v = hi
            if typ == 'int': v = int(round(v))
            chp[k] = v
        return chp

    def optimize(self, evaluate_fn, verbose=True):
        pop = [self._sample_random() for _ in range(self.pop_size)]
        pop_scores = [evaluate_fn(p) for p in pop]
        best_idx = int(np.argmin(pop_scores))
        best = copy.deepcopy(pop[best_idx])
        best_score = pop_scores[best_idx]

        for t in range(1, self.max_iter + 1):
            AW = self.aw_min + (self.aw_max - self.aw_min) * math.exp(-self.gamma * (t / self.max_iter))
            for i in range(self.pop_size):
                if random.random() < 0.6:
                    s = levy_step(beta=1.5, size=len(self.bounds))
                    keys = list(self.bounds.keys())
                    k = random.randrange(self.pop_size)
                    theta_i = np.array([pop[i][kk] for kk in keys], dtype=float)
                    theta_k = np.array([pop[k][kk] for kk in keys], dtype=float)
                    theta_best = np.array([best[kk] for kk in keys], dtype=float)
                    new_theta = theta_i + s * (theta_i - theta_k) + s * (theta_best - theta_i)
                    candidate = {keys[j]: new_theta[j] for j in range(len(keys))}
                else:
                    keys = list(self.bounds.keys())
                    a, b = random.sample(range(self.pop_size), 2)
                    theta_a = np.array([pop[a][kk] for kk in keys], dtype=float)
                    theta_b = np.array([pop[b][kk] for kk in keys], dtype=float)
                    theta_best = np.array([best[kk] for kk in keys], dtype=float)
                    noise = np.random.normal(0, 1, size=len(keys))
                    new_theta = theta_best + AW * (theta_a - theta_b) + (1 - AW) * noise
                    candidate = {keys[j]: new_theta[j] for j in range(len(keys))}
                candidate = self._clip(candidate)
                score = evaluate_fn(candidate)
                if score < pop_scores[i]:
                    pop[i] = candidate
                    pop_scores[i] = score
                if pop_scores[i] < best_score:
                    best_score = pop_scores[i]
                    best = copy.deepcopy(pop[i])
            if verbose:
                print(f"Iter {t}/{self.max_iter} Best score: {best_score:.4f}")
        return best, best_score