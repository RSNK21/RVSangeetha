"""Playback through sounddevice; degrades gracefully with no audio device."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

try:
    import sounddevice as sd
    _SD_ERR = None
except Exception as e:  # pragma: no cover - import-time environment issue
    sd = None
    _SD_ERR = e


class Player:
    def __init__(self) -> None:
        self._stream = None
        self._data: Optional[np.ndarray] = None
        self._sr = 44100
        self._pos = 0
        self._end = 0
        self.on_position: Optional[Callable[[int], None]] = None

    @property
    def available(self) -> bool:
        return sd is not None

    @property
    def playing(self) -> bool:
        return self._stream is not None and self._stream.active

    @property
    def position(self) -> int:
        return self._pos

    def play(self, data: np.ndarray, sr: int, start: int = 0,
             end: Optional[int] = None) -> bool:
        if sd is None:
            return False
        self.stop()
        self._data = data
        self._sr = sr
        self._pos = start
        self._end = data.shape[0] if end is None else end

        def callback(out, frames, time_info, status):
            n = min(frames, self._end - self._pos)
            if n <= 0:
                out[:] = 0
                raise sd.CallbackStop
            out[:n] = self._data[self._pos:self._pos + n]
            if n < frames:
                out[n:] = 0
            self._pos += n
            if self.on_position:
                self.on_position(self._pos)

        try:
            self._stream = sd.OutputStream(
                samplerate=sr, channels=data.shape[1], dtype="float32",
                callback=callback)
            self._stream.start()
            return True
        except Exception:
            self._stream = None
            return False

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
