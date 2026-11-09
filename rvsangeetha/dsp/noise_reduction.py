"""Spectral noise gating, modelled on Audacity's Noise Reduction effect.

Two-pass workflow: capture a noise profile from a stretch of tape that
contains only hiss (lead-in, gap between items), then gate every STFT bin
whose energy stays near that profile.  Time smoothing (attack/release) and
frequency smoothing keep the gate from chattering, which matters for veena
recordings where the long sustain of a plucked string decays *into* the
hiss floor and a hard gate would chop the tail of every gamaka.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import stft, istft

_FFT = 2048
_HOP = 512


@dataclass
class NoiseProfile:
    """Per-bin mean log-magnitude of the sampled noise."""

    mean_db: np.ndarray  # shape (bins,)
    samplerate: float
    fft_size: int = _FFT


def _stft(x: np.ndarray, sr: float):
    return stft(x, fs=sr, nperseg=_FFT, noverlap=_FFT - _HOP, window="hann",
                padded=True, boundary="zeros")


def auto_noise_segment(x: np.ndarray, sr: float,
                       window_s: float = 0.5) -> tuple[int, int]:
    """Locate the quietest ``window_s`` stretch (mono view) for auto-profiling."""
    mono = x if x.ndim == 1 else x.mean(axis=1)
    win = max(1, int(window_s * sr))
    if len(mono) <= win:
        return 0, len(mono)
    hop = max(1, win // 4)
    sq = mono.astype(np.float64) ** 2
    csum = np.concatenate(([0.0], np.cumsum(sq)))
    starts = np.arange(0, len(mono) - win, hop)
    energy = csum[starts + win] - csum[starts]
    s = int(starts[int(np.argmin(energy))])
    return s, s + win


def estimate_noise_profile(noise: np.ndarray, sr: float) -> NoiseProfile:
    """Build a profile from audio that contains only the unwanted noise."""
    mono = noise if noise.ndim == 1 else noise.mean(axis=1)
    if len(mono) < _FFT:
        mono = np.pad(mono, (0, _FFT - len(mono)))
    _, _, z = _stft(mono, sr)
    mag = np.abs(z) + 1e-12
    mean_db = 20.0 * np.log10(mag).mean(axis=1)
    return NoiseProfile(mean_db=mean_db, samplerate=sr)


def _smooth_freq(gain: np.ndarray, bands: int) -> np.ndarray:
    if bands <= 0:
        return gain
    k = 2 * bands + 1
    kern = np.ones(k) / k
    pad = np.pad(gain, ((bands, bands), (0, 0)), mode="edge")
    out = np.empty_like(gain)
    for t in range(gain.shape[1]):
        out[:, t] = np.convolve(pad[:, t], kern, mode="valid")
    return out


def _smooth_time(gain: np.ndarray, sr: float,
                 attack_ms: float, release_ms: float) -> np.ndarray:
    frame_s = _HOP / sr
    a_att = float(np.exp(-frame_s / (max(attack_ms, 1.0) / 1000.0))) \
        if attack_ms > 0 else 0.0
    a_rel = float(np.exp(-frame_s / (max(release_ms, 1.0) / 1000.0))) \
        if release_ms > 0 else 0.0
    out = np.empty_like(gain)
    prev = gain[:, 0]
    out[:, 0] = prev
    for t in range(1, gain.shape[1]):
        g = gain[:, t]
        rising = g > prev  # gate opening: attack
        coeff = np.where(rising, a_att, a_rel)
        prev = coeff * prev + (1.0 - coeff) * g
        out[:, t] = prev
    return out


def _reduce_channel(x: np.ndarray, sr: float, profile: NoiseProfile,
                    reduction_db: float, sensitivity_db: float,
                    freq_smooth_bands: int,
                    attack_ms: float, release_ms: float) -> np.ndarray:
    _, _, z = _stft(x, sr)
    mag_db = 20.0 * np.log10(np.abs(z) + 1e-12)
    thresh = (profile.mean_db + sensitivity_db)[:, None]
    floor = 10.0 ** (-reduction_db / 20.0)
    gain = np.where(mag_db > thresh, 1.0, floor)
    gain = _smooth_freq(gain, freq_smooth_bands)
    gain = _smooth_time(gain, sr, attack_ms, release_ms)
    _, y = istft(z * gain, fs=sr, nperseg=_FFT, noverlap=_FFT - _HOP,
                 window="hann")
    y = y[: len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y.astype(x.dtype, copy=False)


def reduce_noise(x: np.ndarray, sr: float, profile: NoiseProfile,
                 reduction_db: float = 12.0, sensitivity_db: float = 6.0,
                 freq_smooth_bands: int = 3,
                 attack_ms: float = 20.0, release_ms: float = 100.0,
                 ) -> np.ndarray:
    """Gate ``x`` against ``profile``.  Works on mono or (n, ch) arrays."""
    if x.ndim == 1:
        return _reduce_channel(x, sr, profile, reduction_db, sensitivity_db,
                               freq_smooth_bands, attack_ms, release_ms)
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        out[:, c] = _reduce_channel(np.ascontiguousarray(x[:, c]), sr, profile,
                                    reduction_db, sensitivity_db,
                                    freq_smooth_bands, attack_ms, release_ms)
    return out
