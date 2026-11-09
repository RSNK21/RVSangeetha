"""Browser-side session for the Pyodide build.

Mirrors the web server's API but holds one in-memory document and avoids
soundfile (unavailable in WebAssembly): WAV encoding is plain numpy.
"""

from __future__ import annotations

import io
import json
import struct
import zipfile
from typing import Optional

import numpy as np

from rvsangeetha.dsp import dynamics
from rvsangeetha.dsp.dispatch import apply_effect
from rvsangeetha.dsp.noise_reduction import (NoiseProfile,
                                             estimate_noise_profile)

MAX_UNDO = 8


def _wav_bytes(x: np.ndarray, sr: int, bits: int = 16) -> bytes:
    """Minimal RIFF/WAVE PCM encoder for mono or (n, ch) float32."""
    if x.ndim == 1:
        x = x[:, None]
    n, ch = x.shape
    clipped = np.clip(x, -1.0, 1.0)
    if bits == 16:
        pcm = (clipped * 32767.0).astype("<i2").tobytes()
        bps = 2
    elif bits == 24:
        i32 = (clipped * 8388607.0).astype("<i4")
        # drop the high byte of each little-endian int32 -> packed 24-bit
        pcm = np.frombuffer(i32.tobytes(), dtype=np.uint8) \
                .reshape(-1, 4)[:, :3].tobytes()
        bps = 3
    else:
        raise ValueError("bits must be 16 or 24")
    byte_rate = sr * ch * bps
    data_len = n * ch * bps
    hdr = (b"RIFF" + struct.pack("<I", 36 + data_len) + b"WAVEfmt " +
           struct.pack("<IHHIIHH", 16, 1, ch, sr, byte_rate, ch * bps,
                       bps * 8) + b"data" + struct.pack("<I", data_len))
    return hdr + pcm


class LocalDoc:
    def __init__(self) -> None:
        self.samples = np.zeros((0, 1), dtype=np.float32)
        self.samplerate = 44100
        self.markers: list[int] = []
        self._undo: list[tuple[str, np.ndarray, list[int]]] = []
        self._redo: list[tuple[str, np.ndarray, list[int]]] = []
        self._clipboard: Optional[np.ndarray] = None

    frames = property(lambda s: s.samples.shape[0])
    channels = property(lambda s: s.samples.shape[1] if s.samples.ndim > 1
                        else 1)
    duration = property(lambda s: s.frames / s.samplerate)

    def _snapshot(self, label: str) -> None:
        self._undo.append((label, self.samples.copy(), list(self.markers)))
        if len(self._undo) > MAX_UNDO:
            self._undo.pop(0)
        self._redo.clear()

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> Optional[str]:
        if not self._undo:
            return None
        label, snap, marks = self._undo.pop()
        self._redo.append((label, self.samples, self.markers))
        self.samples, self.markers = snap, marks
        return label

    def redo(self) -> Optional[str]:
        if not self._redo:
            return None
        label, snap, marks = self._redo.pop()
        self._undo.append((label, self.samples, self.markers))
        self.samples, self.markers = snap, marks
        return label

    def _shift_markers(self, at: int, delta: int) -> None:
        self.markers = sorted({max(0, m + delta) if m >= at else m
                               for m in self.markers})

    def replace_all(self, samples: np.ndarray, label: str) -> None:
        self._snapshot(label)
        self.samples = np.ascontiguousarray(samples.astype(np.float32))

    def replace_range(self, start: int, end: int, samples: np.ndarray,
                      label: str) -> None:
        self._snapshot(label)
        out = self.samples.copy()
        out[start:end] = samples.reshape(end - start, self.channels)
        self.samples = out

    def delete(self, start: int, end: int, label: str = "Delete") -> None:
        self._snapshot(label)
        self.samples = np.ascontiguousarray(
            np.concatenate([self.samples[:start], self.samples[end:]]))
        self._shift_markers(start, -(end - start))

    def copy(self, start: int, end: int) -> None:
        self._clipboard = self.samples[start:end].copy()

    def cut(self, start: int, end: int) -> None:
        self.copy(start, end)
        self.delete(start, end, "Cut")

    def paste(self, at: int) -> bool:
        if self._clipboard is None or not len(self._clipboard):
            return False
        if self.frames and self._clipboard.shape[1] != self.channels:
            return False
        self._snapshot("Paste")
        self.samples = np.ascontiguousarray(np.concatenate(
            [self.samples[:at], self._clipboard, self.samples[at:]]))
        self._shift_markers(at, self._clipboard.shape[0])
        return True

    def trim(self, start: int, end: int) -> None:
        self._snapshot("Trim")
        self.samples = np.ascontiguousarray(self.samples[start:end])
        self.markers = sorted({min(max(0, m - start), end - start)
                               for m in self.markers
                               if start <= m <= end})

    def silence(self, start: int, end: int) -> None:
        self._snapshot("Silence")
        out = self.samples.copy()
        out[start:end] = 0
        self.samples = out

    def add_split(self, at: int) -> None:
        if 0 < at < self.frames and at not in self.markers:
            self.markers = sorted(self.markers + [at])

    def clear_splits(self) -> None:
        self.markers = []


_doc = LocalDoc()
_profile: Optional[NoiseProfile] = None
_name = "untitled"


def load(buf, channels: int, samplerate: int, name: str) -> str:
    """buf is interleaved float32 (memoryview from JS)."""
    global _doc, _profile, _name
    arr = np.frombuffer(bytes(buf), dtype=np.float32)
    _doc = LocalDoc()
    _doc.samples = np.ascontiguousarray(
        arr.reshape(-1, channels).astype(np.float32))
    _doc.samplerate = int(samplerate)
    _profile = None
    _name = name or "untitled"
    return info()


def info() -> str:
    d = _doc
    return json.dumps({
        "session": "local", "name": _name,
        "samplerate": d.samplerate, "channels": d.channels,
        "frames": d.frames, "duration": d.duration,
        "peak_db": None if not d.frames else round(
            float(dynamics.peak_db(d.samples)), 2),
        "markers": d.markers, "has_profile": _profile is not None,
        "can_undo": d.can_undo(), "can_redo": d.can_redo(),
    })


def peaks(cols: int) -> str:
    d = _doc
    cols = max(16, min(8000, int(cols)))
    mono = d.samples.mean(axis=1) if d.channels > 1 else d.samples[:, 0]
    if not len(mono):
        return json.dumps({"mins": [], "maxs": [],
                           "frames": 0, "duration": 0})
    idx = np.linspace(0, len(mono), cols + 1).astype(int)
    mins, maxs = [], []
    for i in range(cols):
        seg = mono[idx[i]:max(idx[i] + 1, idx[i + 1])]
        mins.append(round(float(seg.min()), 4))
        maxs.append(round(float(seg.max()), 4))
    return json.dumps({"mins": mins, "maxs": maxs, "frames": d.frames,
                       "duration": d.duration})


def _clamp(start: int, end) -> tuple[int, int]:
    start = max(0, min(int(start), _doc.frames))
    end = _doc.frames if end is None else max(start, min(int(end),
                                                         _doc.frames))
    return start, end


def edit(op: str, start: int, end: int, at: int) -> str:
    d = _doc
    start, end = _clamp(start, end)
    at = max(0, min(int(at), d.frames))
    if op == "cut":
        d.cut(start, end)
    elif op == "copy":
        d.copy(start, end)
    elif op == "delete":
        d.delete(start, end)
    elif op == "trim":
        d.trim(start, end)
    elif op == "silence":
        d.silence(start, end)
    elif op == "paste":
        if not d.paste(at):
            raise ValueError("Clipboard is empty or channel count differs.")
    elif op == "split":
        d.add_split(at)
    elif op == "clear_splits":
        d.clear_splits()
    else:
        raise ValueError(f"Unknown edit op {op!r}")
    return info()


def undo() -> str:
    _doc.undo()
    return info()


def redo() -> str:
    _doc.redo()
    return info()


def capture_profile(start: int, end: int) -> str:
    global _profile
    start, end = _clamp(start, end)
    seg = _doc.samples[start:end]
    if seg.shape[0] < 256:
        raise ValueError("Selection too short for a noise profile.")
    _profile = estimate_noise_profile(seg, _doc.samplerate)
    return info()


def effect(name: str, params_json: str, start: int, end) -> str:
    d = _doc
    start, end = _clamp(start, end)
    seg = d.samples[start:end].copy()
    out = apply_effect(name, json.loads(params_json), seg, d.samplerate,
                       _profile)
    label = name.replace("_", " ").title()
    if out.shape[0] == end - start and (start, end) != (0, d.frames):
        d.replace_range(start, end, out, label)
    else:
        d.replace_all(out, label)
    return info()


def slice_wav(start: int, end) -> bytes:
    start, end = _clamp(start, end)
    return _wav_bytes(_doc.samples[start:end], _doc.samplerate)


def export(fmt: str, start: int, end) -> bytes:
    start, end = _clamp(start, end)
    bits = {"wav16": 16, "wav24": 24}.get(fmt)
    if bits is None:
        raise ValueError(f"Format {fmt!r} is not available in the browser "
                         "app; use WAV.")
    return _wav_bytes(_doc.samples[start:end], _doc.samplerate, bits)


def export_segments() -> bytes:
    d = _doc
    if not d.markers:
        raise ValueError("No split markers set.")
    bounds = sorted({0, *[m for m in d.markers if 0 < m < d.frames],
                     d.frames})
    stem = _name.rsplit(".", 1)[0] or "segment"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i in range(len(bounds) - 1):
            z.writestr(f"{stem}_{i + 1:02d}.wav",
                       _wav_bytes(d.samples[bounds[i]:bounds[i + 1]],
                                  d.samplerate))
    return buf.getvalue()
