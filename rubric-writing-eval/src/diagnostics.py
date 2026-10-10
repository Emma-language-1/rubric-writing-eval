from __future__ import annotations

import numpy as np


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[: min(2, vt.shape[0])].T
    projected = centered @ components
    if projected.shape[1] == 1:
        projected = np.column_stack([projected[:, 0], np.zeros(len(projected))])
    return projected


def dbscan(points: np.ndarray, eps: float = 1.25, min_samples: int = 2) -> np.ndarray:
    labels = np.full(len(points), -99, dtype=int)
    cluster_id = 0

    def neighbours(index: int) -> list[int]:
        distances = np.linalg.norm(points - points[index], axis=1)
        return np.where(distances <= eps)[0].tolist()

    for i in range(len(points)):
        if labels[i] != -99:
            continue
        nearby = neighbours(i)
        if len(nearby) < min_samples:
            labels[i] = -1
            continue
        labels[i] = cluster_id
        seeds = [x for x in nearby if x != i]
        while seeds:
            j = seeds.pop()
            if labels[j] == -1:
                labels[j] = cluster_id
            if labels[j] != -99:
                continue
            labels[j] = cluster_id
            expanded = neighbours(j)
            if len(expanded) >= min_samples:
                seeds.extend(x for x in expanded if x not in seeds)
        cluster_id += 1
    return labels
