"""DSP regression tests on synthetic signals."""

import numpy as np
import pytest

from rvsangeetha.dsp import declick, dynamics, filters
from rvsangeetha.dsp.noise_reduction import (auto_noise_segment,
                                             estimate_noise_profile,
                                             reduce_noise)
from rvsangeetha.dsp.remaster import remaster, veena_preset

SR = 44100
RNG = np.random.default_rng(7)


def _tone(freq, dur=2.0, amp=0.5):
    t = np.arange(int(SR * dur)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _band_rms(x, lo, hi):
    spec = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), 1 / SR)
    sel = (f >= lo) & (f <= hi)
    return np.sqrt(np.mean(spec[sel] ** 2))


def test_hum_removal_notches_50hz_and_harmonics():
    sig = _tone(440) + _tone(50, amp=0.2) + _tone(150, amp=0.1)
    out = filters.remove_hum(sig, SR, fundamental=50.0)
    # measure at the notch centres; a narrow Q leaves the skirts alone
    assert _band_rms(out, 49.6, 50.4) < _band_rms(sig, 49.6, 50.4) * 0.05
    assert _band_rms(out, 149.6, 150.4) < _band_rms(sig, 149.6, 150.4) * 0.05
    # music content survives
    assert _band_rms(out, 430, 450) > _band_rms(sig, 430, 450) * 0.9


def test_noise_reduction_improves_snr():
    clean = _tone(330)
    noise = (0.02 * RNG.standard_normal(len(clean))).astype(np.float32)
    noisy = clean + noise
    # profile from a noise-only lead-in
    profile = estimate_noise_profile(noise[: SR // 2], SR)
    out = reduce_noise(noisy, SR, profile, reduction_db=15.0)
    # compare noise energy well away from the tone
    assert _band_rms(out, 5000, 10000) < _band_rms(noisy, 5000, 10000) * 0.4
    assert _band_rms(out, 320, 340) > _band_rms(noisy, 320, 340) * 0.7


def test_auto_noise_segment_finds_quiet_region():
    loud = _tone(220, dur=1.0)
    quiet = (0.001 * RNG.standard_normal(SR)).astype(np.float32)
    x = np.concatenate([loud, quiet, loud])
    s, e = auto_noise_segment(x, SR)
    assert SR <= s and e <= 2 * SR + int(0.5 * SR)


def test_click_removal_repairs_spike():
    sig = _tone(440)
    sig[10000:10004] += 0.9
    out = declick.remove_clicks(sig, SR)
    assert np.max(np.abs(out[9990:10014])) < 0.6
    # untouched elsewhere
    assert np.allclose(out[:5000], sig[:5000])


def test_declip_rebuilds_flat_tops():
    sig = _tone(440, amp=1.0)
    clipped = np.clip(sig, -0.6, 0.6) / 0.6 * 0.6
    out = declick.declip(clipped, threshold=0.95)
    # repaired peaks should exceed the clip ceiling
    assert np.max(np.abs(out)) > 0.61


def test_normalize_and_limiter():
    sig = _tone(440, amp=0.1)
    out = dynamics.normalize(sig, -1.0)
    assert dynamics.peak_db(out) == pytest.approx(-1.0, abs=0.05)
    lim = dynamics.soft_limit(np.ones(100, dtype=np.float32) * 2.0, -0.3)
    assert np.max(np.abs(lim)) <= 10 ** (-0.3 / 20) + 1e-6


def test_full_remaster_chain_stereo():
    mono = _tone(440) + _tone(50, amp=0.15) \
        + (0.01 * RNG.standard_normal(2 * SR)).astype(np.float32)
    x = np.stack([mono, mono], axis=1)
    stages = []
    out = remaster(x, SR, veena_preset(),
                   progress=lambda s, f: stages.append(s))
    assert out.shape == x.shape
    assert dynamics.peak_db(out) == pytest.approx(-1.0, abs=0.4)
    # hum relative to music must drop sharply (gain-invariant measure)
    hum_ratio_in = _band_rms(x[:, 0], 48, 52) / _band_rms(x[:, 0], 430, 450)
    hum_ratio_out = _band_rms(out[:, 0], 48, 52) / _band_rms(out[:, 0],
                                                             430, 450)
    assert hum_ratio_out < hum_ratio_in * 0.1
    assert "Reducing tape hiss" in stages


def test_document_edit_undo_roundtrip():
    from rvsangeetha.audio_document import AudioDocument
    doc = AudioDocument()
    doc.samples = np.stack([_tone(100, dur=1.0)] * 2, axis=1)
    doc.samplerate = SR
    n = doc.frames
    doc.cut(1000, 2000)
    assert doc.frames == n - 1000
    doc.paste(1000)
    assert doc.frames == n
    doc.add_split(5000)
    assert doc.markers == [5000]
    doc.undo()
    doc.undo()
    doc.undo()
    assert doc.frames == n and doc.markers == []
    doc.redo()
    assert doc.frames == n - 1000
