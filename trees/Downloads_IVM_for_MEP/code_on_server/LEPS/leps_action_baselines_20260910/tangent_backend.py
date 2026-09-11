"""Batched algebraic equivalent of ASE ImprovedTangentMethod for nonperiodic LEPS."""
import numpy as np


def improved_tangents(path, energies, k=1.0):
    path = np.asarray(path, dtype=float)
    energies = np.asarray(energies, dtype=float)
    if path.shape != (22, 6) or energies.shape != (22,):
        raise ValueError('unexpected LEPS band shape')
    edge = np.diff(path, axis=0)
    left, right = edge[:-1], edge[1:]
    up, mid, down = energies[2:], energies[1:-1], energies[:-2]
    hi = np.maximum(abs(up-mid), abs(down-mid))[:, None]
    lo = np.minimum(abs(up-mid), abs(down-mid))[:, None]
    tangent = np.where((up > down)[:, None], right*hi + left*lo, right*lo + left*hi)
    tangent = np.where(((up > mid) & (mid > down))[:, None], right, tangent)
    tangent = np.where(((up < mid) & (mid < down))[:, None], left, tangent)
    return tangent / np.linalg.norm(tangent, axis=1, keepdims=True)
