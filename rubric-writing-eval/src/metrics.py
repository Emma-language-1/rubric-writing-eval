from __future__ import annotations

import numpy as np


def quadratic_weighted_kappa(y_true, y_pred, min_rating=1, max_rating=9) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("Ratings must be non-empty arrays with the same shape")
    n = max_rating - min_rating + 1
    observed = np.zeros((n, n), dtype=float)
    for a, b in zip(y_true, y_pred):
        observed[a - min_rating, b - min_rating] += 1
    hist_true = np.bincount(y_true - min_rating, minlength=n).astype(float)
    hist_pred = np.bincount(y_pred - min_rating, minlength=n).astype(float)
    expected = np.outer(hist_true, hist_pred) / y_true.size
    indices = np.arange(n)
    weights = ((indices[:, None] - indices[None, :]) ** 2) / ((n - 1) ** 2)
    denominator = float((weights * expected).sum())
    if denominator == 0:
        return 1.0 if np.array_equal(y_true, y_pred) else 0.0
    return 1.0 - float((weights * observed).sum()) / denominator


def agreement_summary(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    differences = np.abs(y_true - y_pred)
    return {
        "qwk": quadratic_weighted_kappa(y_true.astype(int), y_pred.astype(int)),
        "exact_agreement": float(np.mean(differences == 0)),
        "adjacent_agreement": float(np.mean(differences <= 1)),
        "mae": float(np.mean(differences)),
    }
