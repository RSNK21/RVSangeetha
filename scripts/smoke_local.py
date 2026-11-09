"""Smoke-test the pure-browser (Pyodide) build served as static files."""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from playwright.sync_api import sync_playwright

SITE = Path("/tmp/_site")
WAV = Path("/tmp/veena_test.wav")


def make_wav():
    sr = 22050
    t = np.arange(int(sr * 3)) / sr
    tone = 0.4 * np.sin(2 * np.pi * 330 * t) * np.exp(-t / 2)
    hum = 0.05 * np.sin(2 * np.pi * 50 * t)
    rng = np.random.default_rng(0)
    hiss = 0.02 * rng.standard_normal(len(t))
    sf.write(WAV, (tone + hum + hiss).astype(np.float32), sr)


def main():
    make_wav()
    srv = subprocess.Popen(
        [sys.executable, "-m", "http.server", "8765", "-d", str(SITE)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("dialog", lambda d: (errors.append("dialog: " +
                                                       d.message),
                                         d.dismiss()))
            page.goto("http://localhost:8765/")
            page.wait_for_function(
                "document.getElementById('drophint').textContent"
                ".includes('browser')", timeout=15000)
            print("local mode detected OK")
            assert page.is_hidden("#expflac"), "FLAC button should be hidden"

            page.set_input_files("#file", str(WAV))
            page.wait_for_selector("#editor", state="visible",
                                   timeout=240000)
            status = page.text_content("#status")
            print("loaded:", status)
            # browsers may resample on decode; just check it loaded sanely
            assert "Hz" in status and "3.0 s" in status

            # one-click remaster
            page.click("#remaster")
            page.wait_for_function(
                "document.getElementById('status').textContent"
                ".includes('done')", timeout=240000)
            print("remaster:", page.text_content("#status"))

            # undo / redo
            page.click("#undo")
            page.wait_for_timeout(500)
            page.click("#redo")
            page.wait_for_timeout(500)

            # export wav16
            with page.expect_download(timeout=60000) as dl:
                page.click("#exp16")
            path = dl.value.path()
            data, sr = sf.read(path)
            print(f"export: {dl.value.suggested_filename} "
                  f"{len(data)/sr:.2f}s @ {sr} Hz")
            assert sr >= 22050 and len(data) > 0

            # split + segments zip
            page.click("#wave", position={"x": 200, "y": 100})
            page.click("#split")
            page.wait_for_timeout(500)
            with page.expect_download(timeout=60000) as dl2:
                page.click("#expseg")
            print("segments:", dl2.value.suggested_filename)

            page.screenshot(path="/tmp/web_local.png", full_page=True)
            browser.close()
            if errors:
                print("PAGE ERRORS:", errors)
                sys.exit(1)
    finally:
        srv.terminate()
    print("LOCAL BROWSER SMOKE OK")


if __name__ == "__main__":
    main()
