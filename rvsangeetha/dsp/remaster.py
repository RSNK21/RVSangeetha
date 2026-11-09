"""The one-click remaster chain, voiced for cassette recordings of the veena.

Order matters: defects are repaired before broadband processing so the
noise profile is not polluted by clicks, hum is notched before the gate so
50 Hz energy does not hold gate bins open, and normalisation comes last.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import declick, dynamics, filters
from .noise_reduction import (NoiseProfile, auto_noise_segment,
                              estimate_noise_profile, reduce_noise)

Progress = Callable[[str, float], None]


@dataclass
class RemasterSettings:
    # defect repair
    declick_enabled: bool = True
    declick_threshold: float = 14.0
    declip_enabled: bool = True
    declip_threshold: float = 0.95
    # hum
    hum_enabled: bool = True
    hum_fundamental: float = 50.0   # Indian mains
    hum_harmonics: int = 8
    hum_q: float = 35.0
    # hiss
    nr_enabled: bool = True
    nr_reduction_db: float = 12.0
    nr_sensitivity_db: float = 6.0
    nr_freq_smooth: int = 3
    nr_attack_ms: float = 20.0
    nr_release_ms: float = 150.0
    # tone
    rumble_enabled: bool = True
    rumble_cutoff: float = 30.0
    shelf_enabled: bool = True
    shelf_freq: float = 7500.0
    shelf_gain_db: float = 2.5
    presence_enabled: bool = True
    presence_freq: float = 3200.0
    presence_gain_db: float = 1.5
    presence_q: float = 1.2
    # level
    normalize_enabled: bool = True
    normalize_db: float = -1.0
    limiter_enabled: bool = True
    limiter_ceiling_db: float = -0.3


def veena_preset() -> RemasterSettings:
    """Defaults voiced for veena + mridangam/ghatam/khanjira on cassette.

    - 50 Hz comb: Indian mains hum.
    - Moderate 12 dB hiss reduction with a long release so the decay of a
      plucked string and the shimmer of gamakas are not gated off.
    - Gentle 7.5 kHz shelf to recover sparkle lost to tape self-erasure,
      plus a small 3.2 kHz presence lift for the veena's attack and for
      the artist's occasional vocal passages.
    """
    return RemasterSettings()


def remaster(x: np.ndarray, sr: float, settings: RemasterSettings,
             noise_profile: Optional[NoiseProfile] = None,
             progress: Optional[Progress] = None) -> np.ndarray:
    def report(stage: str, frac: float) -> None:
        if progress:
            progress(stage, frac)

    y = x.astype(np.float32, copy=True)

    if settings.declick_enabled:
        report("Repairing clicks and dropouts", 0.05)
        y = declick.remove_clicks(y, sr, threshold=settings.declick_threshold)
    if settings.declip_enabled:
        report("Rebuilding saturated peaks", 0.20)
        y = declick.declip(y, threshold=settings.declip_threshold)
    if settings.hum_enabled:
        report("Removing mains hum", 0.30)
        y = filters.remove_hum(y, sr, settings.hum_fundamental,
                               settings.hum_harmonics, settings.hum_q)
    if settings.rumble_enabled:
        report("Filtering transport rumble", 0.40)
        y = filters.rumble_filter(y, sr, settings.rumble_cutoff)
    if settings.nr_enabled:
        report("Reducing tape hiss", 0.50)
        if noise_profile is None:
            s, e = auto_noise_segment(y, sr)
            noise_profile = estimate_noise_profile(y[s:e], sr)
        y = reduce_noise(y, sr, noise_profile,
                         reduction_db=settings.nr_reduction_db,
                         sensitivity_db=settings.nr_sensitivity_db,
                         freq_smooth_bands=settings.nr_freq_smooth,
                         attack_ms=settings.nr_attack_ms,
                         release_ms=settings.nr_release_ms)
    if settings.shelf_enabled:
        report("Restoring high frequencies", 0.75)
        y = filters.high_shelf(y, sr, settings.shelf_freq,
                               settings.shelf_gain_db)
    if settings.presence_enabled:
        report("Presence EQ", 0.82)
        y = filters.peaking_eq(y, sr, settings.presence_freq,
                               settings.presence_gain_db, settings.presence_q)
    if settings.limiter_enabled:
        y = dynamics.soft_limit(y, settings.limiter_ceiling_db)
    if settings.normalize_enabled:
        report("Normalising", 0.90)
        y = dynamics.normalize(y, settings.normalize_db)
    report("Done", 1.0)
    return y
