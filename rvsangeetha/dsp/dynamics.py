"""Level handling: peak normalisation, soft limiting, fades."""

from __future__ import annotations

import numpy as np


def peak_db(x: np.ndarray) -> float:
    p = float(np.max(np.abs(x))) if x.size else 0.0
    return 20.0 * np.log10(p) if p > 0 else -np.inf


def normalize(x: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    p = float(np.max(np.abs(x))) if x.size else 0.0
    if p <= 0:
        return x.copy()
    return (x * (10.0 ** (target_db / 20.0) / p)).astype(x.dtype, copy=False)


def soft_limit(x: np.ndarray, ceiling_db: float = -0.5,
               knee: float = 0.7) -> np.ndarray:
    """Soft-knee clipper: transparent below ``knee * ceiling``, smooth above."""
    c = 10.0 ** (ceiling_db / 20.0)
    k = knee * c
    ax = np.abs(x)
    over = ax > k
    y = x.copy()
    shaped = k + (c - k) * np.tanh((ax[over] - k) / (c - k))
    y[over] = np.sign(x[over]) * shaped
    return y.astype(x.dtype, copy=False)


def apply_gain(x: np.ndarray, gain_db: float) -> np.ndarray:
    return (x * 10.0 ** (gain_db / 20.0)).astype(x.dtype, copy=False)


def fade(x: np.ndarray, sr: float, fade_in_s: float = 0.0,
         fade_out_s: float = 0.0) -> np.ndarray:
    y = x.copy()
    n = y.shape[0]
    fi = min(n, int(fade_in_s * sr))
    fo = min(n, int(fade_out_s * sr))
    if fi > 0:
        ramp = np.linspace(0.0, 1.0, fi, dtype=y.dtype)
        y[:fi] = (y[:fi].T * ramp).T
    if fo > 0:
        ramp = np.linspace(1.0, 0.0, fo, dtype=y.dtype)
        y[n - fo:] = (y[n - fo:].T * ramp).T
    return y
