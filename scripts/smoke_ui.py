"""Headless UI smoke test: build the window, load audio, exercise edits,
apply an effect synchronously, and save a screenshot.

Run:  QT_QPA_PLATFORM=offscreen python scripts/smoke_ui.py out.png
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from PySide6.QtWidgets import QApplication

from rvsangeetha.dsp import filters
from rvsangeetha.ui.main_window import MainWindow

SR = 44100


def make_test_tape(path: Path) -> None:
    """Synthetic 'cassette': plucked-string-ish tones + hiss + 50 Hz hum."""
    rng = np.random.default_rng(3)
    t = np.arange(SR * 8) / SR
    sig = np.zeros_like(t, dtype=np.float32)
    for i, f in enumerate([294.0, 330.0, 392.0, 440.0]):  # sa ri ga pa -ish
        start = int(i * 1.8 * SR)
        n = int(1.6 * SR)
        tt = np.arange(n) / SR
        pluck = np.sin(2 * np.pi * f * tt) * np.exp(-2.2 * tt)
        pluck += 0.3 * np.sin(2 * np.pi * 2 * f * tt) * np.exp(-4 * tt)
        sig[start:start + n] += 0.5 * pluck.astype(np.float32)
    sig += 0.02 * rng.standard_normal(len(t)).astype(np.float32)  # hiss
    sig += 0.03 * np.sin(2 * np.pi * 50 * t).astype(np.float32)   # hum
    sf.write(str(path), np.stack([sig, sig], axis=1), SR)


def main() -> int:
    out_png = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("smoke.png")
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.show()

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "veena_side_a.wav"
        make_test_tape(wav)
        win.doc.load(wav)
        assert win.doc.frames == SR * 8, "load failed"

        # selection + noise profile from the quiet tail
        win.wave.set_selection(int(7.3 * SR), int(7.9 * SR))
        win.capture_profile()
        assert win.noise_profile is not None, "profile capture failed"

        # synchronous effect through the document (bypasses worker thread)
        before = win.doc.samples.copy()
        cleaned = filters.remove_hum(before, SR, 50.0)
        win.doc.replace_all(cleaned, "Hum Removal")
        assert win.doc.can_undo(), "undo stack empty after effect"

        # edits: split markers + cut + undo
        win.wave.set_cursor_frame(int(1.8 * SR))
        win.split_at_cursor()
        assert win.doc.markers, "split marker missing"
        n = win.doc.frames
        win.doc.cut(0, SR // 2)
        assert win.doc.frames == n - SR // 2, "cut failed"
        win.undo()
        assert win.doc.frames == n, "undo failed"

        # exports
        out_dir = Path(td) / "segments"
        out_dir.mkdir()
        files = win.doc.export_segments(out_dir, "kriti")
        assert len(files) == 2 and all(f.exists() for f in files), \
            "segment export failed"

        win.wave.set_selection(int(1.8 * SR), int(3.6 * SR))
        win._update_status()
        app.processEvents()
        pm = win.grab()
        pm.save(str(out_png))

    print(f"smoke OK — screenshot at {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
