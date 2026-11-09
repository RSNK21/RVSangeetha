"""Hum removal and tone-shaping filters (RBJ biquads + scipy IIR)."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfiltfilt, tf2sos


def _apply_sos(x: np.ndarray, sos: np.ndarray,
               zero_phase: bool = False) -> np.ndarray:
    """Offline we can afford zero-phase filtering for notches/high-pass,
    which avoids the long ring-in of narrow causal notches.  A long pad is
    needed because scipy's default is far shorter than a high-Q notch's
    settling time, which would leave hum ringing at both ends."""
    if not zero_phase:
        if x.ndim == 1:
            return sosfilt(sos, x).astype(x.dtype, copy=False)
        return sosfilt(sos, x, axis=0).astype(x.dtype, copy=False)
    padlen = min(x.shape[0] - 1, 60000)
    if x.ndim == 1:
        return sosfiltfilt(sos, x, padlen=padlen).astype(x.dtype, copy=False)
    return sosfiltfilt(sos, x, axis=0, padlen=padlen).astype(x.dtype,
                                                             copy=False)


def remove_hum(x: np.ndarray, sr: float, fundamental: float = 50.0,
               harmonics: int = 8, q: float = 35.0) -> np.ndarray:
    """Notch out mains hum and its harmonics.

    Indian mains is 50 Hz; cassette decks and worn pinch rollers also leak
    odd harmonics strongly, so a comb of narrow notches is used.  Q stays
    high so the veena's own low register (around 65-130 Hz for a D/E sarani)
    is not thinned out.
    """
    sections = []
    for k in range(1, harmonics + 1):
        f = fundamental * k
        if f >= sr / 2 * 0.98:
            break
        b, a = iirnotch(f, q, fs=sr)
        sections.append(tf2sos(b, a))
    if not sections:
        return x
    sos = np.vstack(sections)
    return _apply_sos(x, sos, zero_phase=True)


def rumble_filter(x: np.ndarray, sr: float, cutoff: float = 30.0,
                  order: int = 4) -> np.ndarray:
    """High-pass to drop turntable/transport rumble and handling thumps."""
    sos = butter(order, cutoff, btype="highpass", fs=sr, output="sos")
    return _apply_sos(x, sos, zero_phase=True)


def lowpass(x: np.ndarray, sr: float, cutoff: float = 16000.0,
            order: int = 4) -> np.ndarray:
    sos = butter(order, min(cutoff, sr / 2 * 0.98), btype="lowpass",
                 fs=sr, output="sos")
    return _apply_sos(x, sos)


def _rbj_shelf(sr: float, f0: float, gain_db: float, high: bool,
               slope: float = 0.9) -> np.ndarray:
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / sr
    cs, sn = np.cos(w0), np.sin(w0)
    alpha = sn / 2 * np.sqrt((a + 1 / a) * (1 / slope - 1) + 2)
    two_sqa = 2 * np.sqrt(a) * alpha
    if high:
        b0 = a * ((a + 1) + (a - 1) * cs + two_sqa)
        b1 = -2 * a * ((a - 1) + (a + 1) * cs)
        b2 = a * ((a + 1) + (a - 1) * cs - two_sqa)
        a0 = (a + 1) - (a - 1) * cs + two_sqa
        a1 = 2 * ((a - 1) - (a + 1) * cs)
        a2 = (a + 1) - (a - 1) * cs - two_sqa
    else:
        b0 = a * ((a + 1) - (a - 1) * cs + two_sqa)
        b1 = 2 * a * ((a - 1) - (a + 1) * cs)
        b2 = a * ((a + 1) - (a - 1) * cs - two_sqa)
        a0 = (a + 1) + (a - 1) * cs + two_sqa
        a1 = -2 * ((a - 1) + (a + 1) * cs)
        a2 = (a + 1) + (a - 1) * cs - two_sqa
    return tf2sos(np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0]))


def _rbj_peak(sr: float, f0: float, gain_db: float, q: float) -> np.ndarray:
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / sr
    cs, sn = np.cos(w0), np.sin(w0)
    alpha = sn / (2 * q)
    b = np.array([1 + alpha * a, -2 * cs, 1 - alpha * a])
    aa = np.array([1 + alpha / a, -2 * cs, 1 - alpha / a])
    return tf2sos(b / aa[0], aa / aa[0])


def high_shelf(x: np.ndarray, sr: float, f0: float = 7500.0,
               gain_db: float = 2.5) -> np.ndarray:
    """Restore top end dulled by tape self-erasure and azimuth drift."""
    return _apply_sos(x, _rbj_shelf(sr, f0, gain_db, high=True))


def low_shelf(x: np.ndarray, sr: float, f0: float = 120.0,
              gain_db: float = 0.0) -> np.ndarray:
    return _apply_sos(x, _rbj_shelf(sr, f0, gain_db, high=False))


def peaking_eq(x: np.ndarray, sr: float, f0: float, gain_db: float,
               q: float = 1.0) -> np.ndarray:
    return _apply_sos(x, _rbj_peak(sr, f0, gain_db, q))
