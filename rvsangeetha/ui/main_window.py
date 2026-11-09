"""RVSangeetha main window: transport, editing, restoration effects."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QFileDialog, QLabel, QMainWindow, QMessageBox,
                               QProgressDialog, QToolBar, QVBoxLayout,
                               QWidget)

from .. import APP_NAME, __version__
from ..audio_document import AudioDocument
from ..dsp import declick, dynamics, filters
from ..dsp.noise_reduction import (NoiseProfile, estimate_noise_profile,
                                   reduce_noise)
from ..dsp.remaster import RemasterSettings, remaster, veena_preset
from ..playback import Player
from . import dialogs
from .waveform import WaveformView

DARK_STYLE = """
QMainWindow, QDialog, QMessageBox { background: #14161c; color: #d8dce6; }
QMenuBar, QMenu { background: #1b1e26; color: #d8dce6; }
QMenuBar::item:selected, QMenu::item:selected { background: #2d3342; }
QToolBar { background: #1b1e26; border: none; spacing: 6px; }
QLabel { color: #d8dce6; }
QStatusBar { background: #1b1e26; color: #9aa3b5; }
QGroupBox { color: #d8dce6; border: 1px solid #2d3342; border-radius: 4px;
            margin-top: 8px; padding-top: 14px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; }
QPushButton { background: #2d3342; color: #d8dce6; border: none;
              padding: 6px 14px; border-radius: 3px; }
QPushButton:hover { background: #3a4156; }
QSlider::groove:horizontal { height: 4px; background: #2d3342; }
QSlider::handle:horizontal { width: 12px; background: #69aaff;
                             margin: -5px 0; border-radius: 6px; }
QSpinBox, QDoubleSpinBox { background: #1b1e26; color: #d8dce6;
                           border: 1px solid #2d3342; padding: 2px; }
QCheckBox { color: #d8dce6; }
"""


class _Worker(QThread):
    done = Signal(object)
    failed = Signal(str)
    progressed = Signal(str, float)

    def __init__(self, fn: Callable[..., np.ndarray], parent=None) -> None:
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            result = self._fn(lambda s, f: self.progressed.emit(s, f))
            self.done.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1200, 640)
        self.setStyleSheet(DARK_STYLE)

        self.doc = AudioDocument()
        self.doc.on_change = self._refresh
        self.player = Player()
        self.noise_profile: Optional[NoiseProfile] = None
        self._worker: Optional[_Worker] = None
        self._progress: Optional[QProgressDialog] = None

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        self.wave = WaveformView()
        self.wave.selection_changed.connect(self._update_status)
        self.wave.cursor_moved.connect(lambda _: self._update_status())
        layout.addWidget(self.wave)
        self.setCentralWidget(central)

        self._status = QLabel("Open a cassette transfer to begin "
                              "(File → Open…)")
        self.statusBar().addWidget(self._status, 1)

        self._play_timer = QTimer(self)
        self._play_timer.setInterval(50)
        self._play_timer.timeout.connect(self._tick_playhead)

        self._build_menus()

    # ------------------------------------------------------------- menus
    def _act(self, menu, text: str, slot, shortcut: Optional[str] = None
             ) -> QAction:
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.triggered.connect(slot)
        menu.addAction(a)
        return a

    def _build_menus(self) -> None:
        mb = self.menuBar()

        m = mb.addMenu("&File")
        self._act(m, "&Open…", self.open_file, "Ctrl+O")
        self._act(m, "&Save As…", self.save_as, "Ctrl+Shift+S")
        m.addSeparator()
        self._act(m, "Export &Selection…", self.export_selection)
        self._act(m, "Export Split Se&gments…", self.export_segments)
        m.addSeparator()
        self._act(m, "E&xit", self.close, "Ctrl+Q")

        m = mb.addMenu("&Edit")
        self._act(m, "&Undo", self.undo, "Ctrl+Z")
        self._act(m, "&Redo", self.redo, "Ctrl+Y")
        m.addSeparator()
        self._act(m, "Cu&t", self.cut, "Ctrl+X")
        self._act(m, "&Copy", self.copy, "Ctrl+C")
        self._act(m, "&Paste", self.paste, "Ctrl+V")
        self._act(m, "&Delete", self.delete, "Del")
        self._act(m, "Tri&m to Selection", self.trim, "Ctrl+T")
        self._act(m, "&Silence Selection", self.silence, "Ctrl+L")
        m.addSeparator()
        self._act(m, "Select &All", self.wave.select_all, "Ctrl+A")
        m.addSeparator()
        self._act(m, "Split at &Cursor", self.split_at_cursor, "Ctrl+I")
        self._act(m, "Clear Split &Markers", self.clear_splits)

        m = mb.addMenu("&Transport")
        self._act(m, "&Play / Stop", self.toggle_play, "Space")
        self._act(m, "&Stop", self.stop, "Esc")

        m = mb.addMenu("Effe&cts")
        self._act(m, "✦ One-Click Remaster (Veena Preset)…",
                  self.run_remaster, "Ctrl+R")
        m.addSeparator()
        self._act(m, "Capture &Noise Profile from Selection",
                  self.capture_profile, "Ctrl+N")
        self._act(m, "Noise Reduction (Tape &Hiss)…", self.run_nr)
        self._act(m, "Hum &Removal…", self.run_hum)
        self._act(m, "&Click / Dropout Repair…", self.run_declick)
        self._act(m, "&Declip (Distortion Repair)…", self.run_declip)
        m.addSeparator()
        self._act(m, "Tape Tone &EQ…", self.run_eq)
        self._act(m, "Nor&malize…", self.run_normalize)
        self._act(m, "&Amplify…", self.run_gain)
        self._act(m, "&Fade In / Out…", self.run_fade)

        m = mb.addMenu("&Help")
        self._act(m, "&About RVSangeetha", self.about)

        tb = QToolBar("Transport")
        tb.setMovable(False)
        self.addToolBar(tb)
        for text, slot in [("▶ Play", self.toggle_play), ("■ Stop", self.stop),
                           ("✦ Remaster", self.run_remaster)]:
            a = QAction(text, self)
            a.triggered.connect(slot)
            tb.addAction(a)

    # ----------------------------------------------------------- helpers
    def _refresh(self) -> None:
        self.wave.set_audio(self.doc.samples, self.doc.samplerate)
        self.wave.set_markers(self.doc.markers)
        self._update_status()

    def _update_status(self) -> None:
        if not self.doc.frames:
            return
        d = self.doc
        sel = self.wave.selection()
        parts = [f"{d.path.name if d.path else 'untitled'}",
                 f"{d.samplerate} Hz · {d.channels} ch · "
                 f"{d.duration:.1f} s",
                 f"peak {dynamics.peak_db(d.samples):.1f} dBFS"]
        if sel:
            parts.append(f"selection {sel[0] / d.samplerate:.2f}–"
                         f"{sel[1] / d.samplerate:.2f} s")
        if self.noise_profile is not None:
            parts.append("noise profile ✓")
        if d.markers:
            parts.append(f"{len(d.markers)} split(s)")
        self._status.setText("   |   ".join(parts))

    def _need_audio(self) -> bool:
        if not self.doc.frames:
            QMessageBox.information(self, APP_NAME,
                                    "Open an audio file first.")
            return True
        return False

    def _selection_or_all(self) -> tuple[int, int]:
        sel = self.wave.selection()
        return sel if sel else (0, self.doc.frames)

    def _run_job(self, label: str,
                 fn: Callable[[Callable[[str, float], None]], np.ndarray],
                 start: int, end: int) -> None:
        """Run ``fn`` on a worker thread, then write result into [start:end)."""
        self._progress = QProgressDialog(label, None, 0, 100, self)
        self._progress.setWindowModality(Qt.WindowModal)
        self._progress.setMinimumDuration(300)
        self._worker = _Worker(fn)
        self._worker.progressed.connect(
            lambda s, f: (self._progress.setLabelText(s),
                          self._progress.setValue(int(f * 100))))

        def finish(result: np.ndarray) -> None:
            self._progress.setValue(100)
            if result.shape[0] == end - start:
                self.doc.replace_range(start, end, result, label)
            else:
                self.doc.replace_all(result, label)
            self.statusBar().showMessage(f"{label}: done", 4000)

        def fail(tb: str) -> None:
            self._progress.cancel()
            QMessageBox.critical(self, APP_NAME, f"{label} failed:\n\n{tb}")

        self._worker.done.connect(finish)
        self._worker.failed.connect(fail)
        self._worker.start()

    # -------------------------------------------------------------- file
    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open cassette transfer", "",
            "Audio files (*.wav *.flac *.ogg *.mp3 *.aiff *.aif);;"
            "All files (*)")
        if not path:
            return
        try:
            self.doc.load(path)
        except Exception as e:
            QMessageBox.critical(self, APP_NAME, f"Could not open file:\n{e}")
            return
        self.noise_profile = None
        self.wave.clear_selection()
        self.setWindowTitle(f"{APP_NAME} — {Path(path).name}")

    def _save_dialog(self) -> Optional[str]:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save as", "",
            "WAV 16-bit (*.wav);;WAV 24-bit (*.wav);;FLAC (*.flac);;"
            "MP3 (*.mp3)")
        return path or None

    def save_as(self) -> None:
        if self._need_audio():
            return
        path = self._save_dialog()
        if not path:
            return
        subtype = "PCM_24" if path.lower().endswith(".wav") else None
        try:
            self.doc.export(path, subtype=subtype)
            self.statusBar().showMessage(f"Saved {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, APP_NAME, f"Save failed:\n{e}")

    def export_selection(self) -> None:
        if self._need_audio():
            return
        sel = self.wave.selection()
        if not sel:
            QMessageBox.information(self, APP_NAME, "Make a selection first.")
            return
        path = self._save_dialog()
        if path:
            try:
                self.doc.export(path, sel[0], sel[1])
                self.statusBar().showMessage(f"Exported {path}", 5000)
            except Exception as e:
                QMessageBox.critical(self, APP_NAME, f"Export failed:\n{e}")

    def export_segments(self) -> None:
        if self._need_audio():
            return
        if not self.doc.markers:
            QMessageBox.information(
                self, APP_NAME,
                "No split markers. Use Edit → Split at Cursor to mark item "
                "boundaries first.")
            return
        directory = QFileDialog.getExistingDirectory(self,
                                                     "Export segments to")
        if not directory:
            return
        stem = (self.doc.path.stem if self.doc.path else "segment")
        try:
            files = self.doc.export_segments(directory, stem)
            self.statusBar().showMessage(
                f"Exported {len(files)} segment(s) to {directory}", 6000)
        except Exception as e:
            QMessageBox.critical(self, APP_NAME, f"Export failed:\n{e}")

    # -------------------------------------------------------------- edit
    def undo(self) -> None:
        label = self.doc.undo()
        if label:
            self.statusBar().showMessage(f"Undid: {label}", 3000)

    def redo(self) -> None:
        label = self.doc.redo()
        if label:
            self.statusBar().showMessage(f"Redid: {label}", 3000)

    def _with_selection(self, fn: Callable[[int, int], None]) -> None:
        if self._need_audio():
            return
        sel = self.wave.selection()
        if not sel:
            QMessageBox.information(self, APP_NAME, "Make a selection first.")
            return
        fn(*sel)
        self.wave.clear_selection()

    def cut(self) -> None:
        self._with_selection(self.doc.cut)

    def copy(self) -> None:
        sel = self.wave.selection()
        if sel:
            self.doc.copy(*sel)
            self.statusBar().showMessage("Copied", 2000)

    def paste(self) -> None:
        if self._need_audio():
            return
        if not self.doc.paste(self.wave.cursor_frame()):
            QMessageBox.information(self, APP_NAME, "Clipboard is empty.")

    def delete(self) -> None:
        self._with_selection(self.doc.delete)

    def trim(self) -> None:
        self._with_selection(self.doc.trim)

    def silence(self) -> None:
        self._with_selection(self.doc.silence)

    def split_at_cursor(self) -> None:
        if self._need_audio():
            return
        self.doc.add_split(self.wave.cursor_frame())

    def clear_splits(self) -> None:
        self.doc.clear_splits()

    # --------------------------------------------------------- transport
    def toggle_play(self) -> None:
        if self.player.playing:
            self.stop()
            return
        if self._need_audio():
            return
        if not self.player.available:
            QMessageBox.information(self, APP_NAME,
                                    "No audio output device available.")
            return
        sel = self.wave.selection()
        start = sel[0] if sel else self.wave.cursor_frame()
        end = sel[1] if sel else None
        if self.player.play(self.doc.samples, self.doc.samplerate,
                            start, end):
            self._play_timer.start()

    def stop(self) -> None:
        self.player.stop()
        self._play_timer.stop()
        self.wave.set_play_head(None)

    def _tick_playhead(self) -> None:
        if self.player.playing:
            self.wave.set_play_head(self.player.position)
        else:
            self.stop()

    # ------------------------------------------------------------ effects
    def capture_profile(self) -> None:
        if self._need_audio():
            return
        sel = self.wave.selection()
        if not sel:
            QMessageBox.information(
                self, APP_NAME,
                "Select a stretch of pure tape hiss (lead-in or a gap "
                "between items), then capture the profile.")
            return
        seg = self.doc.samples[sel[0]:sel[1]]
        self.noise_profile = estimate_noise_profile(seg, self.doc.samplerate)
        self.statusBar().showMessage("Noise profile captured", 4000)
        self._update_status()

    def run_nr(self) -> None:
        if self._need_audio():
            return
        dlg = dialogs.noise_reduction_dialog(
            self, has_profile=self.noise_profile is not None)
        if not dlg.exec():
            return
        v = dlg.values()
        start, end = self._selection_or_all()
        seg = self.doc.samples[start:end].copy()
        sr = self.doc.samplerate
        profile = self.noise_profile

        def job(progress):
            progress("Analysing noise…", 0.1)
            p = profile
            if p is None:
                from ..dsp.noise_reduction import auto_noise_segment
                s, e = auto_noise_segment(seg, sr)
                p = estimate_noise_profile(seg[s:e], sr)
            progress("Reducing tape hiss…", 0.4)
            return reduce_noise(seg, sr, p,
                                reduction_db=v["reduction_db"],
                                sensitivity_db=v["sensitivity_db"],
                                freq_smooth_bands=int(v["freq_smooth"]),
                                attack_ms=v["attack_ms"],
                                release_ms=v["release_ms"])

        self._run_job("Noise Reduction", job, start, end)

    def run_hum(self) -> None:
        if self._need_audio():
            return
        dlg = dialogs.hum_dialog(self)
        if not dlg.exec():
            return
        v = dlg.values()
        start, end = self._selection_or_all()
        seg = self.doc.samples[start:end].copy()
        sr = self.doc.samplerate
        self._run_job(
            "Hum Removal",
            lambda progress: filters.remove_hum(
                seg, sr, v["fundamental"], int(v["harmonics"]), v["q"]),
            start, end)

    def run_declick(self) -> None:
        if self._need_audio():
            return
        dlg = dialogs.declick_dialog(self)
        if not dlg.exec():
            return
        v = dlg.values()
        start, end = self._selection_or_all()
        seg = self.doc.samples[start:end].copy()
        sr = self.doc.samplerate
        self._run_job(
            "Click Repair",
            lambda progress: declick.remove_clicks(
                seg, sr, v["threshold"], v["max_click_ms"]),
            start, end)

    def run_declip(self) -> None:
        if self._need_audio():
            return
        dlg = dialogs.declip_dialog(self)
        if not dlg.exec():
            return
        v = dlg.values()
        start, end = self._selection_or_all()
        seg = self.doc.samples[start:end].copy()
        self._run_job(
            "Declip",
            lambda progress: declick.declip(seg, v["threshold"]),
            start, end)

    def run_eq(self) -> None:
        if self._need_audio():
            return
        dlg = dialogs.eq_dialog(self)
        if not dlg.exec():
            return
        v = dlg.values()
        start, end = self._selection_or_all()
        seg = self.doc.samples[start:end].copy()
        sr = self.doc.samplerate

        def job(progress):
            progress("Rumble filter…", 0.2)
            y = filters.rumble_filter(seg, sr, v["rumble_cutoff"])
            progress("Brilliance shelf…", 0.5)
            y = filters.high_shelf(y, sr, v["shelf_freq"], v["shelf_gain_db"])
            progress("Presence…", 0.8)
            return filters.peaking_eq(y, sr, v["presence_freq"],
                                      v["presence_gain_db"], v["presence_q"])

        self._run_job("Tape Tone EQ", job, start, end)

    def run_normalize(self) -> None:
        if self._need_audio():
            return
        dlg = dialogs.normalize_dialog(self)
        if not dlg.exec():
            return
        v = dlg.values()
        start, end = self._selection_or_all()
        seg = self.doc.samples[start:end].copy()
        self._run_job("Normalize",
                      lambda progress: dynamics.normalize(seg,
                                                          v["target_db"]),
                      start, end)

    def run_gain(self) -> None:
        if self._need_audio():
            return
        dlg = dialogs.gain_dialog(self)
        if not dlg.exec():
            return
        v = dlg.values()
        start, end = self._selection_or_all()
        seg = self.doc.samples[start:end].copy()
        self._run_job("Amplify",
                      lambda progress: dynamics.apply_gain(seg, v["gain_db"]),
                      start, end)

    def run_fade(self) -> None:
        if self._need_audio():
            return
        dlg = dialogs.fade_dialog(self)
        if not dlg.exec():
            return
        v = dlg.values()
        start, end = self._selection_or_all()
        seg = self.doc.samples[start:end].copy()
        sr = self.doc.samplerate
        self._run_job("Fade",
                      lambda progress: dynamics.fade(seg, sr, v["fade_in_s"],
                                                     v["fade_out_s"]),
                      start, end)

    def run_remaster(self) -> None:
        if self._need_audio():
            return
        dlg = dialogs.RemasterDialog(
            veena_preset(), self, has_profile=self.noise_profile is not None)
        if not dlg.exec():
            return
        settings: RemasterSettings = dlg.settings()
        samples = self.doc.samples.copy()
        sr = self.doc.samplerate
        profile = self.noise_profile
        self._run_job(
            "Remaster",
            lambda progress: remaster(samples, sr, settings, profile,
                                      progress),
            0, self.doc.frames)

    # --------------------------------------------------------------- misc
    def about(self) -> None:
        QMessageBox.about(
            self, f"About {APP_NAME}",
            f"<b>{APP_NAME}</b> v{__version__}<br><br>"
            "Cassette restoration and remastering for Carnatic music, "
            "voiced for the Saraswati veena with mridangam, ghatam and "
            "khanjira accompaniment.<br><br>"
            "Workflow: open the transfer → capture a noise profile from "
            "tape hiss → One-Click Remaster → audition → fine-tune with "
            "individual effects → split items → export.")

    def closeEvent(self, event) -> None:
        self.stop()
        if self.doc.dirty:
            r = QMessageBox.question(
                self, APP_NAME,
                "There are unsaved changes. Quit anyway?",
                QMessageBox.Yes | QMessageBox.No)
            if r != QMessageBox.Yes:
                event.ignore()
                return
        event.accept()
