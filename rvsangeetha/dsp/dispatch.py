"""Shared effect dispatch used by the web server and the Pyodide bridge."""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import declick, dynamics, filters
from .noise_reduction import (NoiseProfile, auto_noise_segment,
                              estimate_noise_profile, reduce_noise)
from .remaster import remaster, veena_preset


def apply_effect(name: str, params: dict, x: np.ndarray, sr: float,
                 profile: Optional[NoiseProfile] = None) -> np.ndarray:
    """Apply effect ``name`` with ``params`` to ``x``; raises ValueError
    for unknown names."""
    p = params or {}

    if name == "noise_reduction":
        prof = profile
        if prof is None:
            a, b = auto_noise_segment(x, sr)
            prof = estimate_noise_profile(x[a:b], sr)
        return reduce_noise(
            x, sr, prof,
            reduction_db=float(p.get("reduction_db", 12)),
            sensitivity_db=float(p.get("sensitivity_db", 6)),
            freq_smooth_bands=int(p.get("freq_smooth", 3)),
            attack_ms=float(p.get("attack_ms", 20)),
            release_ms=float(p.get("release_ms", 150)))
    if name == "hum":
        return filters.remove_hum(x, sr, float(p.get("fundamental", 50)),
                                  int(p.get("harmonics", 8)),
                                  float(p.get("q", 35)))
    if name == "declick":
        return declick.remove_clicks(x, sr, float(p.get("threshold", 14)),
                                     float(p.get("max_click_ms", 2)))
    if name == "declip":
        return declick.declip(x, float(p.get("threshold", 0.95)))
    if name == "eq":
        out = filters.rumble_filter(x, sr, float(p.get("rumble_cutoff", 30)))
        out = filters.high_shelf(out, sr, float(p.get("shelf_freq", 7500)),
                                 float(p.get("shelf_gain_db", 2.5)))
        return filters.peaking_eq(out, sr,
                                  float(p.get("presence_freq", 3200)),
                                  float(p.get("presence_gain_db", 1.5)),
                                  float(p.get("presence_q", 1.2)))
    if name == "normalize":
        return dynamics.normalize(x, float(p.get("target_db", -1)))
    if name == "gain":
        return dynamics.apply_gain(x, float(p.get("gain_db", 0)))
    if name == "fade":
        return dynamics.fade(x, sr, float(p.get("fade_in_s", 0)),
                             float(p.get("fade_out_s", 0)))
    if name == "remaster":
        settings = veena_preset()
        for k, v in p.items():
            if hasattr(settings, k):
                cur = getattr(settings, k)
                setattr(settings, k,
                        bool(v) if isinstance(cur, bool) else type(cur)(v))
        return remaster(x, sr, settings, profile)
    raise ValueError(f"Unknown effect {name!r}")
