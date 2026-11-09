"""Impulsive-defect repair: dropout clicks, crackle and saturated peaks.

Clicks are found as derivative outliers against a local robust (MAD)
estimate, then the damaged span is rebuilt by cubic interpolation from the
clean samples either side -- the same idea as Audacity's Click Removal and
Clip Fix.  Mridangam strokes are themselves sharp transients, so detection
requires the spike to be far outside the local statistics before touching
anything (a real stroke raises the local MAD along with the peak).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def _repair_spans(x: np.ndarray, bad: np.ndarray, pad: int = 4) -> np.ndarray:
    """Cubic-interpolate every contiguous run of True in ``bad`` (1-D)."""
    y = x.copy()
    n = len(x)
    idx = np.flatnonzero(bad)
    if idx.size == 0:
        return y
    splits = np.flatnonzero(np.diff(idx) > 1) + 1
    for run in np.split(idx, splits):
        s, e = int(run[0]), int(run[-1]) + 1
        lo, hi = max(0, s - pad), min(n, e + pad)
        anchors = np.concatenate([np.arange(lo, s), np.arange(e, hi)])
        if anchors.size < 4:
            y[s:e] = 0.0
            continue
        spline = CubicSpline(anchors, y[anchors])
        y[s:e] = spline(np.arange(s, e))
    return y


def _detect_clicks(x: np.ndarray, sr: float, threshold: float,
                   max_click_ms: float) -> np.ndarray:
    d = np.abs(np.diff(x, prepend=x[0]))
    win = max(64, int(sr * 0.02))
    # local MAD of the derivative, block-wise for speed
    nblk = int(np.ceil(len(x) / win))
    pad_len = nblk * win - len(x)
    dp = np.pad(d, (0, pad_len)).reshape(nblk, win)
    med = np.median(dp, axis=1)
    mad = np.median(np.abs(dp - med[:, None]), axis=1) + 1e-9
    med_f = np.repeat(med, win)[: len(x)]
    mad_f = np.repeat(mad, win)[: len(x)]
    bad = d > med_f + threshold * mad_f
    # discard runs longer than a plausible click -- that is music, not a defect
    max_run = max(1, int(sr * max_click_ms / 1000.0))
    idx = np.flatnonzero(bad)
    if idx.size:
        splits = np.flatnonzero(np.diff(idx) > 1) + 1
        for run in np.split(idx, splits):
            if len(run) > max_run:
                bad[run] = False
    return bad


def remove_clicks(x: np.ndarray, sr: float, threshold: float = 14.0,
                  max_click_ms: float = 2.0) -> np.ndarray:
    """Remove short impulsive clicks.  ``threshold`` is in local MADs."""
    if x.ndim == 1:
        bad = _detect_clicks(x, sr, threshold, max_click_ms)
        return _repair_spans(x, bad)
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        ch = np.ascontiguousarray(x[:, c])
        out[:, c] = _repair_spans(ch, _detect_clicks(ch, sr, threshold,
                                                     max_click_ms))
    return out


def declip(x: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """Rebuild flattened peaks from tape or input-stage saturation."""
    def one(ch: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(ch)) if len(ch) else 0.0
        if peak <= 0:
            return ch.copy()
        bad = np.abs(ch) >= threshold * peak
        # only touch runs of 2+ samples; single touching samples are fine
        idx = np.flatnonzero(bad)
        if idx.size:
            keep = np.zeros_like(bad)
            splits = np.flatnonzero(np.diff(idx) > 1) + 1
            for run in np.split(idx, splits):
                if len(run) >= 2:
                    keep[run] = True
            bad = keep
        return _repair_spans(ch, bad)

    if x.ndim == 1:
        return one(x)
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        out[:, c] = one(np.ascontiguousarray(x[:, c]))
    return out
