"""Web API tests: upload → peaks → effects → edits → undo → export."""

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from rvsangeetha.web import server
from rvsangeetha.web.server import app

SR = 44100


@pytest.fixture()
def client():
    server._sessions.clear()
    return TestClient(app)


def _tape_bytes(dur=4.0):
    rng = np.random.default_rng(5)
    t = np.arange(int(SR * dur)) / SR
    sig = (0.5 * np.sin(2 * np.pi * 440 * t)
           + 0.03 * np.sin(2 * np.pi * 50 * t)
           + 0.02 * rng.standard_normal(len(t))).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, np.stack([sig, sig], axis=1), SR, format="WAV")
    return buf.getvalue()


def _upload(client):
    r = client.post("/api/upload", files={
        "file": ("side_a.wav", _tape_bytes(), "audio/wav")})
    assert r.status_code == 200, r.text
    return r.json()


def test_upload_info_peaks(client):
    info = _upload(client)
    sid = info["session"]
    assert info["samplerate"] == SR and info["channels"] == 2
    r = client.get(f"/api/{sid}/peaks?cols=500")
    pk = r.json()
    assert len(pk["mins"]) == 500 and len(pk["maxs"]) == 500
    assert min(pk["mins"]) < -0.1 and max(pk["maxs"]) > 0.1


def test_effect_hum_and_undo(client):
    info = _upload(client)
    sid = info["session"]
    r = client.post(f"/api/{sid}/effect",
                    json={"name": "hum", "params": {"fundamental": 50}})
    assert r.status_code == 200
    assert r.json()["can_undo"] is True
    r = client.post(f"/api/{sid}/undo")
    assert r.json()["undone"] == "Hum"


def test_profile_and_noise_reduction(client):
    info = _upload(client)
    sid = info["session"]
    r = client.post(f"/api/{sid}/profile",
                    json={"start": 0, "end": SR // 2})
    assert r.json()["has_profile"] is True
    r = client.post(f"/api/{sid}/effect",
                    json={"name": "noise_reduction",
                          "params": {"reduction_db": 15}})
    assert r.status_code == 200


def test_remaster_endpoint(client):
    info = _upload(client)
    sid = info["session"]
    r = client.post(f"/api/{sid}/effect",
                    json={"name": "remaster",
                          "params": {"nr_reduction_db": 10}})
    assert r.status_code == 200
    out = r.json()
    assert out["peak_db"] == pytest.approx(-1.0, abs=0.5)


def test_edit_cut_split_segments_export(client):
    info = _upload(client)
    sid = info["session"]
    frames = info["frames"]
    r = client.post(f"/api/{sid}/edit",
                    json={"op": "cut", "start": 0, "end": SR})
    assert r.json()["frames"] == frames - SR
    r = client.post(f"/api/{sid}/edit", json={"op": "split", "at": SR})
    assert r.json()["markers"] == [SR]
    r = client.get(f"/api/{sid}/export_segments")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    import zipfile
    z = zipfile.ZipFile(io.BytesIO(r.content))
    assert len(z.namelist()) == 2


def test_export_formats(client):
    info = _upload(client)
    sid = info["session"]
    for fmt in ("wav16", "wav24", "flac"):
        r = client.get(f"/api/{sid}/export?fmt={fmt}")
        assert r.status_code == 200 and len(r.content) > 1000
    data, sr = sf.read(io.BytesIO(
        client.get(f"/api/{sid}/export?fmt=wav24").content))
    assert sr == SR and data.shape[1] == 2


def test_audio_stream_selection(client):
    info = _upload(client)
    sid = info["session"]
    r = client.get(f"/api/{sid}/audio.wav?start=0&end={SR}")
    data, sr = sf.read(io.BytesIO(r.content))
    assert len(data) == SR


def test_unknown_session_404(client):
    assert client.get("/api/nope/info").status_code == 404


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "RVSangeetha" in r.text
