"""Browser E2E check of the web app: upload, effect, split, screenshot.

Run: PYTHONPATH=. python scripts/smoke_web.py docs/screenshot_web.png
(Requires the server NOT already running; it starts its own on :8765.)
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from playwright.sync_api import sync_playwright

SR = 44100
PORT = 8765


def make_tape(path: Path) -> None:
    rng = np.random.default_rng(3)
    t = np.arange(SR * 8) / SR
    sig = np.zeros_like(t, dtype=np.float32)
    for i, f in enumerate([294.0, 330.0, 392.0, 440.0]):
        s = int(i * 1.8 * SR)
        n = int(1.6 * SR)
        tt = np.arange(n) / SR
        pluck = np.sin(2 * np.pi * f * tt) * np.exp(-2.2 * tt)
        sig[s:s + n] += 0.5 * pluck.astype(np.float32)
    sig += 0.02 * rng.standard_normal(len(t)).astype(np.float32)
    sig += 0.03 * np.sin(2 * np.pi * 50 * t).astype(np.float32)
    sf.write(str(path), np.stack([sig, sig], axis=1), SR)


def main() -> int:
    out_png = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("web.png")
    from rvsangeetha.web.server import app
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT,
                            log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "veena_side_a.wav"
        make_tape(wav)
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{PORT}/")
            assert "RVSangeetha" in page.title()

            page.set_input_files("#file", str(wav))
            page.wait_for_selector("#editor", state="visible", timeout=15000)
            page.wait_for_function(
                "document.getElementById('status').textContent"
                ".includes('veena_side_a')")

            # drag a selection on the quiet tail and capture a profile
            box = page.locator("#wave").bounding_box()
            y = box["y"] + box["height"] / 2
            page.mouse.move(box["x"] + box["width"] * 0.92, y)
            page.mouse.down()
            page.mouse.move(box["x"] + box["width"] * 0.99, y, steps=5)
            page.mouse.up()
            page.click("#profile")
            page.wait_for_function(
                "document.getElementById('profbadge').textContent"
                ".includes('profile')")

            # run hum removal on the whole file (clears selection first)
            page.mouse.click(box["x"] + 2, y)
            page.click("#hum")
            page.wait_for_function(
                "document.getElementById('status').textContent"
                ".includes('done')", timeout=60000)

            # split at cursor
            page.mouse.click(box["x"] + box["width"] * 0.25, y)
            page.click("#split")
            page.wait_for_function(
                "document.getElementById('status').textContent"
                ".includes('split')")

            # undo must re-enable after effect; export endpoint must stream
            assert page.locator("#undo").is_enabled()
            sid = page.evaluate("sid")
            resp = page.request.get(
                f"http://127.0.0.1:{PORT}/api/{sid}/export?fmt=wav16")
            assert resp.ok and len(resp.body()) > 100000, "export failed"

            page.screenshot(path=str(out_png), full_page=True)
            assert not errors, f"JS errors: {errors}"
            browser.close()

    print(f"web smoke OK — screenshot at {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
