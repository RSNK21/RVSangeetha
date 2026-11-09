"""Waveform view: min/max envelope rendering, selection, cursor, markers."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal

ENVELOPE_COLUMNS = 4000  # display resolution; keeps hour-long sides snappy

WAVE_COLOR = (105, 170, 255)
BG_COLOR = (16, 18, 24)
SELECTION_BRUSH = (255, 200, 80, 50)
MARKER_PEN = pg.mkPen((255, 110, 110), width=1, style=Qt.DashLine)
CURSOR_PEN = pg.mkPen((240, 240, 240), width=1)


def envelope(samples: np.ndarray, columns: int = ENVELOPE_COLUMNS):
    """Min/max per column of the mono mix, for fast drawing."""
    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    n = len(mono)
    if n == 0:
        return np.zeros(1), np.zeros(1), np.zeros(1)
    cols = min(columns, n)
    edges = np.linspace(0, n, cols + 1).astype(int)
    mins = np.empty(cols)
    maxs = np.empty(cols)
    for i in range(cols):
        seg = mono[edges[i]:max(edges[i] + 1, edges[i + 1])]
        mins[i] = seg.min()
        maxs[i] = seg.max()
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, mins, maxs


class WaveformView(pg.PlotWidget):
    """One track rendered as a filled min/max envelope.

    Click sets the cursor, click-drag makes a selection, wheel zooms.
    """

    selection_changed = Signal()
    cursor_moved = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent, background=BG_COLOR)
        self.setMenuEnabled(False)
        self.showGrid(x=True, y=False, alpha=0.15)
        self.getPlotItem().setMouseEnabled(x=True, y=False)
        self.setLimits(yMin=-1.05, yMax=1.05)
        self.setYRange(-1.0, 1.0)
        self.getAxis("left").setWidth(42)

        pen = pg.mkPen(WAVE_COLOR, width=1)
        brush = pg.mkBrush((*WAVE_COLOR, 90))
        self._curve_max = self.plot([], [], pen=pen)
        self._curve_min = self.plot([], [], pen=pen)
        self._fill = pg.FillBetweenItem(self._curve_min, self._curve_max,
                                        brush=brush)
        self.addItem(self._fill)

        self._region = pg.LinearRegionItem(brush=pg.mkBrush(*SELECTION_BRUSH),
                                           movable=True)
        self._region.setZValue(10)
        self._region.hide()
        self.addItem(self._region)
        self._region.sigRegionChanged.connect(
            lambda: self.selection_changed.emit())

        self._cursor = pg.InfiniteLine(pos=0, angle=90, pen=CURSOR_PEN,
                                       movable=False)
        self._cursor.setZValue(11)
        self.addItem(self._cursor)

        self._play_head = pg.InfiniteLine(pos=0, angle=90,
                                          pen=pg.mkPen((90, 255, 140),
                                                       width=1))
        self._play_head.setZValue(12)
        self._play_head.hide()
        self.addItem(self._play_head)

        self._marker_lines: list[pg.InfiniteLine] = []
        self._sr = 44100
        self._frames = 0
        self._dragging = False
        self._drag_start = 0.0

    # ------------------------------------------------------------ content
    def set_audio(self, samples: np.ndarray, sr: int) -> None:
        self._sr = sr
        self._frames = samples.shape[0]
        x, mins, maxs = envelope(samples)
        t = x / sr
        self._curve_min.setData(t, mins)
        self._curve_max.setData(t, maxs)
        self.setLimits(xMin=0, xMax=max(self._frames / sr, 1e-3))
        self.setXRange(0, max(self._frames / sr, 1e-3), padding=0)

    def set_markers(self, marker_frames: list[int]) -> None:
        for line in self._marker_lines:
            self.removeItem(line)
        self._marker_lines = []
        for m in marker_frames:
            line = pg.InfiniteLine(pos=m / self._sr, angle=90, pen=MARKER_PEN)
            line.setZValue(9)
            self.addItem(line)
            self._marker_lines.append(line)

    # ---------------------------------------------------------- selection
    def selection(self) -> Optional[tuple[int, int]]:
        if not self._region.isVisible():
            return None
        lo, hi = self._region.getRegion()
        s = max(0, int(lo * self._sr))
        e = min(self._frames, int(hi * self._sr))
        return (s, e) if e > s else None

    def set_selection(self, start: int, end: int) -> None:
        self._region.setRegion((start / self._sr, end / self._sr))
        self._region.show()
        self.selection_changed.emit()

    def clear_selection(self) -> None:
        self._region.hide()
        self.selection_changed.emit()

    def select_all(self) -> None:
        if self._frames:
            self.set_selection(0, self._frames)

    # ------------------------------------------------------------- cursor
    def cursor_frame(self) -> int:
        return max(0, min(self._frames,
                          int(self._cursor.value() * self._sr)))

    def set_cursor_frame(self, frame: int) -> None:
        self._cursor.setValue(frame / self._sr)
        self.cursor_moved.emit(frame)

    def set_play_head(self, frame: Optional[int]) -> None:
        if frame is None:
            self._play_head.hide()
        else:
            self._play_head.setValue(frame / self._sr)
            self._play_head.show()

    # -------------------------------------------------------------- mouse
    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            pos = self.getPlotItem().vb.mapSceneToView(ev.position())
            self._dragging = True
            self._drag_start = float(pos.x())
            self.set_cursor_frame(int(max(0.0, pos.x()) * self._sr))
            self.clear_selection()
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._dragging:
            pos = self.getPlotItem().vb.mapSceneToView(ev.position())
            a, b = sorted((self._drag_start, float(pos.x())))
            if (b - a) * self._sr > 32:  # ignore accidental micro-drags
                self.set_selection(int(max(0.0, a) * self._sr),
                                   int(min(self._frames / self._sr, b)
                                       * self._sr))
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._dragging and ev.button() == Qt.LeftButton:
            self._dragging = False
            ev.accept()
            return
        super().mouseReleaseEvent(ev)
