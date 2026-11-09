"""Effect parameter dialogs: generic slider/spinbox forms per effect."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QSlider, QSpinBox,
                               QVBoxLayout)

from ..dsp.remaster import RemasterSettings


@dataclass
class Param:
    key: str
    label: str
    lo: float
    hi: float
    default: float
    decimals: int = 1
    suffix: str = ""


class _SliderSpin(QHBoxLayout):
    """Linked slider + spinbox for one float parameter."""

    def __init__(self, p: Param) -> None:
        super().__init__()
        self._p = p
        self._scale = 10 ** p.decimals
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(p.lo * self._scale), int(p.hi * self._scale))
        if p.decimals == 0:
            self.spin: Any = QSpinBox()
            self.spin.setRange(int(p.lo), int(p.hi))
        else:
            self.spin = QDoubleSpinBox()
            self.spin.setRange(p.lo, p.hi)
            self.spin.setDecimals(p.decimals)
            self.spin.setSingleStep(1.0 / self._scale * 5)
        if p.suffix:
            self.spin.setSuffix(" " + p.suffix)
        self.addWidget(self.slider, 3)
        self.addWidget(self.spin, 1)
        self.slider.valueChanged.connect(
            lambda v: self.spin.setValue(v / self._scale
                                         if p.decimals else int(v)))
        self.spin.valueChanged.connect(
            lambda v: self.slider.setValue(int(float(v) * self._scale)))
        self.set_value(p.default)

    def set_value(self, v: float) -> None:
        self.spin.setValue(v if self._p.decimals else int(v))

    def value(self) -> float:
        return float(self.spin.value())


class EffectDialog(QDialog):
    """Build a parameter form from a list of :class:`Param`."""

    def __init__(self, title: str, params: list[Param], parent=None,
                 note: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{title} — RVSangeetha")
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)
        if note:
            lbl = QLabel(note)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color: #9ab;")
            layout.addWidget(lbl)
        form = QFormLayout()
        self._controls: dict[str, _SliderSpin] = {}
        for p in params:
            ctl = _SliderSpin(p)
            self._controls[p.key] = ctl
            form.addRow(p.label, ctl)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok
                                   | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> dict[str, float]:
        return {k: c.value() for k, c in self._controls.items()}


def noise_reduction_dialog(parent=None, has_profile: bool = False
                           ) -> EffectDialog:
    note = ("Noise profile: captured from current selection."
            if has_profile else
            "No profile captured — the quietest half-second of the recording "
            "will be used automatically. For best results, select a stretch "
            "of pure tape hiss (lead-in or inter-item gap) and use "
            "Effects → Capture Noise Profile first.")
    return EffectDialog("Noise Reduction (Tape Hiss)", [
        Param("reduction_db", "Reduction", 3, 30, 12, 1, "dB"),
        Param("sensitivity_db", "Sensitivity", 0, 18, 6, 1, "dB"),
        Param("freq_smooth", "Frequency smoothing", 0, 8, 3, 0, "bands"),
        Param("attack_ms", "Attack", 1, 200, 20, 0, "ms"),
        Param("release_ms", "Release", 20, 500, 150, 0, "ms"),
    ], parent, note)


def hum_dialog(parent=None) -> EffectDialog:
    return EffectDialog("Hum Removal", [
        Param("fundamental", "Mains frequency", 40, 70, 50, 1, "Hz"),
        Param("harmonics", "Harmonics", 1, 16, 8, 0),
        Param("q", "Notch Q (narrowness)", 10, 80, 35, 1),
    ], parent, "50 Hz for Indian mains; raise Q if the veena's low "
               "register sounds thin.")


def declick_dialog(parent=None) -> EffectDialog:
    return EffectDialog("Click / Dropout Repair", [
        Param("threshold", "Detection threshold", 6, 30, 14, 1, "MADs"),
        Param("max_click_ms", "Max click length", 0.5, 10, 2, 1, "ms"),
    ], parent, "Higher threshold = more conservative. Keep it high enough "
               "that mridangam strokes are never flagged.")


def declip_dialog(parent=None) -> EffectDialog:
    return EffectDialog("Declip (Distortion Repair)", [
        Param("threshold", "Clip ceiling", 0.80, 1.00, 0.95, 2, "× peak"),
    ], parent, "Rebuilds flattened waveform tops from tape or input-stage "
               "saturation.")


def eq_dialog(parent=None) -> EffectDialog:
    return EffectDialog("Tape Tone EQ", [
        Param("rumble_cutoff", "Rumble high-pass", 15, 120, 30, 0, "Hz"),
        Param("shelf_freq", "Brilliance shelf", 2000, 14000, 7500, 0, "Hz"),
        Param("shelf_gain_db", "Shelf gain", -6, 9, 2.5, 1, "dB"),
        Param("presence_freq", "Presence centre", 1000, 6000, 3200, 0, "Hz"),
        Param("presence_gain_db", "Presence gain", -6, 6, 1.5, 1, "dB"),
        Param("presence_q", "Presence Q", 0.3, 4, 1.2, 1),
    ], parent)


def normalize_dialog(parent=None) -> EffectDialog:
    return EffectDialog("Normalize", [
        Param("target_db", "Peak target", -12, 0, -1, 1, "dBFS"),
    ], parent)


def gain_dialog(parent=None) -> EffectDialog:
    return EffectDialog("Amplify", [
        Param("gain_db", "Gain", -24, 24, 0, 1, "dB"),
    ], parent)


def fade_dialog(parent=None) -> EffectDialog:
    return EffectDialog("Fade", [
        Param("fade_in_s", "Fade in", 0, 10, 0, 1, "s"),
        Param("fade_out_s", "Fade out", 0, 10, 0, 1, "s"),
    ], parent)


class RemasterDialog(QDialog):
    """Full manual control over the one-click remaster chain."""

    def __init__(self, settings: RemasterSettings, parent=None,
                 has_profile: bool = False) -> None:
        super().__init__(parent)
        self.setWindowTitle("Remaster — RVSangeetha")
        self.setMinimumWidth(520)
        self._boxes: dict[str, QCheckBox] = {}
        self._ctls: dict[str, _SliderSpin] = {}
        layout = QVBoxLayout(self)
        note = QLabel(
            "Voiced for cassette veena recitals: repairs defects first, "
            "notches 50 Hz mains, gates hiss with a long release so string "
            "decay and gamakas survive, then restores tape-dulled treble."
            + ("" if has_profile else
               " No noise profile captured — the quietest passage will be "
               "profiled automatically."))
        note.setWordWrap(True)
        note.setStyleSheet("color: #9ab;")
        layout.addWidget(note)

        def group(title: str, enabled_key: str, enabled: bool,
                  params: list[tuple[Param, float]]) -> None:
            box = QGroupBox(title)
            box.setCheckable(True)
            box.setChecked(enabled)
            self._boxes[enabled_key] = box
            form = QFormLayout(box)
            for p, current in params:
                ctl = _SliderSpin(p)
                ctl.set_value(current)
                self._ctls[p.key] = ctl
                form.addRow(p.label, ctl)
            layout.addWidget(box)

        s = settings
        group("Click && dropout repair", "declick_enabled", s.declick_enabled,
              [(Param("declick_threshold", "Threshold", 6, 30, 14, 1, "MADs"),
                s.declick_threshold)])
        group("Declip", "declip_enabled", s.declip_enabled,
              [(Param("declip_threshold", "Ceiling", 0.8, 1.0, 0.95, 2,
                      "× peak"), s.declip_threshold)])
        group("Mains hum removal", "hum_enabled", s.hum_enabled, [
            (Param("hum_fundamental", "Frequency", 40, 70, 50, 1, "Hz"),
             s.hum_fundamental),
            (Param("hum_harmonics", "Harmonics", 1, 16, 8, 0),
             s.hum_harmonics)])
        group("Tape hiss reduction", "nr_enabled", s.nr_enabled, [
            (Param("nr_reduction_db", "Reduction", 3, 30, 12, 1, "dB"),
             s.nr_reduction_db),
            (Param("nr_sensitivity_db", "Sensitivity", 0, 18, 6, 1, "dB"),
             s.nr_sensitivity_db),
            (Param("nr_release_ms", "Release", 20, 500, 150, 0, "ms"),
             s.nr_release_ms)])
        group("Tone restoration", "shelf_enabled", s.shelf_enabled, [
            (Param("shelf_freq", "Brilliance shelf", 2000, 14000, 7500, 0,
                   "Hz"), s.shelf_freq),
            (Param("shelf_gain_db", "Shelf gain", -6, 9, 2.5, 1, "dB"),
             s.shelf_gain_db),
            (Param("presence_gain_db", "Presence (3.2 kHz)", -6, 6, 1.5, 1,
                   "dB"), s.presence_gain_db)])
        group("Loudness", "normalize_enabled", s.normalize_enabled,
              [(Param("normalize_db", "Peak target", -12, 0, -1, 1, "dBFS"),
                s.normalize_db)])

        buttons = QDialogButtonBox(QDialogButtonBox.Ok
                                   | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def settings(self) -> RemasterSettings:
        s = RemasterSettings()
        for key, box in self._boxes.items():
            setattr(s, key, box.isChecked())
        s.rumble_enabled = s.hum_enabled
        s.presence_enabled = s.shelf_enabled
        s.limiter_enabled = s.normalize_enabled
        for key, ctl in self._ctls.items():
            v = ctl.value()
            if key in ("hum_harmonics",):
                setattr(s, key, int(v))
            else:
                setattr(s, key, v)
        return s
