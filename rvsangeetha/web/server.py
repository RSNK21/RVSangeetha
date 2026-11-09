"""RVSangeetha web backend: the same DSP engine behind a REST API.

Sessions are in-memory: one :class:`AudioDocument` per uploaded transfer.
All processing endpoints mirror the desktop Effects menu.
"""

from __future__ import annotations

import io
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import APP_NAME, __version__
from ..audio_document import AudioDocument
from ..dsp import declick, dynamics, filters
from ..dsp.noise_reduction import (NoiseProfile, auto_noise_segment,
                                   estimate_noise_profile, reduce_noise)
from ..dsp.remaster import RemasterSettings, remaster, veena_preset

MAX_SESSIONS = 16
SESSION_TTL_S = 4 * 3600


@dataclass
class Session:
    doc: AudioDocument = field(default_factory=AudioDocument)
    profile: Optional[NoiseProfile] = None
    name: str = "untitled"
    last_used: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)


_sessions: dict[str, Session] = {}
_sessions_lock = threading.Lock()

app = FastAPI(title=APP_NAME, version=__version__)


def _get(sid: str) -> Session:
    s = _sessions.get(sid)
    if s is None:
        raise HTTPException(404, "Unknown session — upload a file first.")
    s.last_used = time.time()
    return s


def _evict() -> None:
    with _sessions_lock:
        now = time.time()
        stale = [k for k, s in _sessions.items()
                 if now - s.last_used > SESSION_TTL_S]
        for k in stale:
            _sessions.pop(k, None)
        while len(_sessions) >= MAX_SESSIONS:
            oldest = min(_sessions, key=lambda k: _sessions[k].last_used)
            _sessions.pop(oldest)


def _info(sid: str, s: Session) -> dict:
    d = s.doc
    return {
        "session": sid,
        "name": s.name,
        "samplerate": d.samplerate,
        "channels": d.channels,
        "frames": d.frames,
        "duration": d.duration,
        "peak_db": None if not d.frames else round(
            float(dynamics.peak_db(d.samples)), 2),
        "markers": d.markers,
        "has_profile": s.profile is not None,
        "can_undo": d.can_undo(),
        "can_redo": d.can_redo(),
    }


# ---------------------------------------------------------------- session
@app.post("/api/upload")
async def upload(file: UploadFile):
    _evict()
    raw = await file.read()
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception as e:
        raise HTTPException(400, f"Could not decode audio: {e}")
    sid = uuid.uuid4().hex[:12]
    s = Session(name=file.filename or "untitled")
    s.doc.samples = np.ascontiguousarray(data)
    s.doc.samplerate = int(sr)
    with _sessions_lock:
        _sessions[sid] = s
    return _info(sid, s)


@app.get("/api/{sid}/info")
def info(sid: str):
    return _info(sid, _get(sid))


@app.get("/api/{sid}/peaks")
def peaks(sid: str, cols: int = 2000):
    s = _get(sid)
    from ..ui.waveform import envelope
    cols = max(16, min(cols, 8000))
    _, mins, maxs = envelope(s.doc.samples, cols)
    return {"mins": np.round(mins, 4).tolist(),
            "maxs": np.round(maxs, 4).tolist(),
            "frames": s.doc.frames, "duration": s.doc.duration}


def _wav_bytes(samples: np.ndarray, sr: int, subtype: str = "PCM_16",
               fmt: str = "WAV") -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format=fmt, subtype=subtype)
    return buf.getvalue()


@app.get("/api/{sid}/audio.wav")
def audio(sid: str, start: int = 0, end: Optional[int] = None):
    s = _get(sid)
    with s.lock:
        end = s.doc.frames if end is None else min(end, s.doc.frames)
        data = _wav_bytes(s.doc.samples[start:end], s.doc.samplerate)
    return Response(data, media_type="audio/wav")


@app.get("/api/{sid}/export")
def export(sid: str, fmt: str = "wav16", start: int = 0,
           end: Optional[int] = None):
    s = _get(sid)
    formats = {"wav16": ("WAV", "PCM_16", "wav"),
               "wav24": ("WAV", "PCM_24", "wav"),
               "flac": ("FLAC", "PCM_16", "flac")}
    if fmt not in formats:
        raise HTTPException(400, f"Unknown format {fmt!r}")
    container, subtype, ext = formats[fmt]
    with s.lock:
        end = s.doc.frames if end is None else min(end, s.doc.frames)
        data = _wav_bytes(s.doc.samples[start:end], s.doc.samplerate,
                          subtype, container)
    stem = Path(s.name).stem or "rvsangeetha"
    return Response(data, media_type="application/octet-stream", headers={
        "Content-Disposition":
            f'attachment; filename="{stem}_remastered.{ext}"'})


@app.get("/api/{sid}/export_segments")
def export_segments(sid: str):
    s = _get(sid)
    with s.lock:
        d = s.doc
        if not d.markers:
            raise HTTPException(400, "No split markers set.")
        bounds = sorted({0, *[m for m in d.markers if 0 < m < d.frames],
                         d.frames})
        stem = Path(s.name).stem or "segment"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for i in range(len(bounds) - 1):
                z.writestr(f"{stem}_{i + 1:02d}.wav",
                           _wav_bytes(d.samples[bounds[i]:bounds[i + 1]],
                                      d.samplerate))
    return Response(buf.getvalue(), media_type="application/zip", headers={
        "Content-Disposition": f'attachment; filename="{stem}_segments.zip"'})


# ------------------------------------------------------------------ edits
class EditReq(BaseModel):
    op: str
    start: int = 0
    end: int = 0
    at: int = 0


@app.post("/api/{sid}/edit")
def edit(sid: str, req: EditReq):
    s = _get(sid)
    with s.lock:
        d = s.doc
        start = max(0, min(req.start, d.frames))
        end = max(start, min(req.end, d.frames))
        if req.op == "cut":
            d.cut(start, end)
        elif req.op == "copy":
            d.copy(start, end)
        elif req.op == "paste":
            if not d.paste(max(0, min(req.at, d.frames))):
                raise HTTPException(400, "Clipboard empty or channel "
                                         "mismatch.")
        elif req.op == "delete":
            d.delete(start, end)
        elif req.op == "trim":
            d.trim(start, end)
        elif req.op == "silence":
            d.silence(start, end)
        elif req.op == "split":
            d.add_split(max(0, min(req.at, d.frames)))
        elif req.op == "clear_splits":
            d.clear_splits()
        else:
            raise HTTPException(400, f"Unknown edit op {req.op!r}")
    return _info(sid, s)


@app.post("/api/{sid}/undo")
def undo(sid: str):
    s = _get(sid)
    with s.lock:
        label = s.doc.undo()
    out = _info(sid, s)
    out["undone"] = label
    return out


@app.post("/api/{sid}/redo")
def redo(sid: str):
    s = _get(sid)
    with s.lock:
        label = s.doc.redo()
    out = _info(sid, s)
    out["redone"] = label
    return out


# ---------------------------------------------------------------- effects
class ProfileReq(BaseModel):
    start: int
    end: int


@app.post("/api/{sid}/profile")
def capture_profile(sid: str, req: ProfileReq):
    s = _get(sid)
    with s.lock:
        seg = s.doc.samples[req.start:req.end]
        if seg.shape[0] < 256:
            raise HTTPException(400, "Selection too short for a noise "
                                     "profile.")
        s.profile = estimate_noise_profile(seg, s.doc.samplerate)
    return _info(sid, s)


class EffectReq(BaseModel):
    name: str
    params: dict = {}
    start: int = 0
    end: Optional[int] = None


def _apply_effect(s: Session, req: EffectReq) -> None:
    d = s.doc
    sr = d.samplerate
    start = max(0, min(req.start, d.frames))
    end = d.frames if req.end is None else max(start, min(req.end, d.frames))
    seg = d.samples[start:end].copy()
    p = req.params

    if req.name == "noise_reduction":
        profile = s.profile
        if profile is None:
            a, b = auto_noise_segment(seg, sr)
            profile = estimate_noise_profile(seg[a:b], sr)
        out = reduce_noise(
            seg, sr, profile,
            reduction_db=float(p.get("reduction_db", 12)),
            sensitivity_db=float(p.get("sensitivity_db", 6)),
            freq_smooth_bands=int(p.get("freq_smooth", 3)),
            attack_ms=float(p.get("attack_ms", 20)),
            release_ms=float(p.get("release_ms", 150)))
    elif req.name == "hum":
        out = filters.remove_hum(seg, sr,
                                 float(p.get("fundamental", 50)),
                                 int(p.get("harmonics", 8)),
                                 float(p.get("q", 35)))
    elif req.name == "declick":
        out = declick.remove_clicks(seg, sr,
                                    float(p.get("threshold", 14)),
                                    float(p.get("max_click_ms", 2)))
    elif req.name == "declip":
        out = declick.declip(seg, float(p.get("threshold", 0.95)))
    elif req.name == "eq":
        out = filters.rumble_filter(seg, sr,
                                    float(p.get("rumble_cutoff", 30)))
        out = filters.high_shelf(out, sr, float(p.get("shelf_freq", 7500)),
                                 float(p.get("shelf_gain_db", 2.5)))
        out = filters.peaking_eq(out, sr,
                                 float(p.get("presence_freq", 3200)),
                                 float(p.get("presence_gain_db", 1.5)),
                                 float(p.get("presence_q", 1.2)))
    elif req.name == "normalize":
        out = dynamics.normalize(seg, float(p.get("target_db", -1)))
    elif req.name == "gain":
        out = dynamics.apply_gain(seg, float(p.get("gain_db", 0)))
    elif req.name == "fade":
        out = dynamics.fade(seg, sr, float(p.get("fade_in_s", 0)),
                            float(p.get("fade_out_s", 0)))
    elif req.name == "remaster":
        settings = veena_preset()
        for k, v in p.items():
            if hasattr(settings, k):
                cur = getattr(settings, k)
                setattr(settings, k,
                        type(cur)(v) if not isinstance(cur, bool)
                        else bool(v))
        out = remaster(seg, sr, settings, s.profile)
    else:
        raise HTTPException(400, f"Unknown effect {req.name!r}")

    label = req.name.replace("_", " ").title()
    if out.shape[0] == end - start and (start, end) != (0, d.frames):
        d.replace_range(start, end, out, label)
    else:
        d.replace_all(out, label)


@app.post("/api/{sid}/effect")
def effect(sid: str, req: EffectReq):
    s = _get(sid)
    with s.lock:
        _apply_effect(s, req)
    return _info(sid, s)


# ----------------------------------------------------------------- static
_static = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(_static), html=True),
          name="static")


def main() -> None:
    import uvicorn
    uvicorn.run("rvsangeetha.web.server:app", host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
