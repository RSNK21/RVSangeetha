"""Audio document: samples, selection-based edits, undo/redo, file I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

MAX_UNDO = 8  # full-buffer snapshots; capped to bound memory on long sides


class AudioDocument:
    """Float32 samples shaped (frames, channels) plus edit history."""

    def __init__(self) -> None:
        self.samples: np.ndarray = np.zeros((0, 1), dtype=np.float32)
        self.samplerate: int = 44100
        self.path: Optional[Path] = None
        self.dirty: bool = False
        self.markers: list[int] = []  # split points, in frames
        self._undo: list[tuple[str, np.ndarray, list[int]]] = []
        self._redo: list[tuple[str, np.ndarray, list[int]]] = []
        self._clipboard: Optional[np.ndarray] = None
        self.on_change: Optional[Callable[[], None]] = None

    # ------------------------------------------------------------- basics
    @property
    def frames(self) -> int:
        return self.samples.shape[0]

    @property
    def channels(self) -> int:
        return self.samples.shape[1]

    @property
    def duration(self) -> float:
        return self.frames / self.samplerate if self.samplerate else 0.0

    def _notify(self) -> None:
        if self.on_change:
            self.on_change()

    # ---------------------------------------------------------------- I/O
    def load(self, path: str | Path) -> None:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        self.samples = np.ascontiguousarray(data)
        self.samplerate = int(sr)
        self.path = Path(path)
        self.markers = []
        self._undo.clear()
        self._redo.clear()
        self.dirty = False
        self._notify()

    def export(self, path: str | Path, start: int = 0,
               end: Optional[int] = None, subtype: Optional[str] = None) -> None:
        end = self.frames if end is None else end
        seg = self.samples[start:end]
        p = str(path)
        kwargs = {"subtype": subtype} if subtype else {}
        sf.write(p, seg, self.samplerate, **kwargs)

    def export_segments(self, directory: str | Path, stem: str) -> list[Path]:
        """Write the regions between split markers as numbered files."""
        bounds = sorted({0, *[m for m in self.markers if 0 < m < self.frames],
                         self.frames})
        out = []
        directory = Path(directory)
        for i in range(len(bounds) - 1):
            p = directory / f"{stem}_{i + 1:02d}.wav"
            self.export(p, bounds[i], bounds[i + 1])
            out.append(p)
        return out

    # --------------------------------------------------------------- undo
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
        label, samples, markers = self._undo.pop()
        self._redo.append((label, self.samples, list(self.markers)))
        self.samples, self.markers = samples, markers
        self.dirty = True
        self._notify()
        return label

    def redo(self) -> Optional[str]:
        if not self._redo:
            return None
        label, samples, markers = self._redo.pop()
        self._undo.append((label, self.samples, list(self.markers)))
        self.samples, self.markers = samples, markers
        self.dirty = True
        self._notify()
        return label

    # -------------------------------------------------------------- edits
    def _shift_markers(self, at: int, delta: int) -> None:
        self.markers = sorted({max(0, m + delta) if m >= at else m
                               for m in self.markers})

    def replace_all(self, samples: np.ndarray, label: str) -> None:
        self._snapshot(label)
        self.samples = np.ascontiguousarray(samples.astype(np.float32,
                                                           copy=False))
        self.dirty = True
        self._notify()

    def replace_range(self, start: int, end: int, samples: np.ndarray,
                      label: str) -> None:
        """Same-length in-place replacement (effect applied to a selection)."""
        self._snapshot(label)
        self.samples[start:end] = samples.astype(np.float32, copy=False)
        self.dirty = True
        self._notify()

    def delete(self, start: int, end: int, label: str = "Delete") -> None:
        self._snapshot(label)
        self.samples = np.ascontiguousarray(
            np.delete(self.samples, np.s_[start:end], axis=0))
        self._shift_markers(end, start - end)
        self.dirty = True
        self._notify()

    def cut(self, start: int, end: int) -> None:
        self._clipboard = self.samples[start:end].copy()
        self.delete(start, end, "Cut")

    def copy(self, start: int, end: int) -> None:
        self._clipboard = self.samples[start:end].copy()

    def paste(self, at: int) -> bool:
        if self._clipboard is None or self._clipboard.shape[1] != self.channels:
            return False
        self._snapshot("Paste")
        self.samples = np.ascontiguousarray(
            np.insert(self.samples, at, self._clipboard, axis=0))
        self._shift_markers(at, self._clipboard.shape[0])
        self.dirty = True
        self._notify()
        return True

    def trim(self, start: int, end: int) -> None:
        self._snapshot("Trim")
        self.samples = np.ascontiguousarray(self.samples[start:end].copy())
        self.markers = sorted({min(max(m - start, 0), end - start)
                               for m in self.markers
                               if start < m < end})
        self.dirty = True
        self._notify()

    def silence(self, start: int, end: int) -> None:
        self._snapshot("Silence")
        self.samples[start:end] = 0.0
        self.dirty = True
        self._notify()

    def add_split(self, at: int) -> None:
        if 0 < at < self.frames and at not in self.markers:
            self._snapshot("Split")
            self.markers = sorted({*self.markers, at})
            self._notify()

    def clear_splits(self) -> None:
        if self.markers:
            self._snapshot("Clear splits")
            self.markers = []
            self._notify()
