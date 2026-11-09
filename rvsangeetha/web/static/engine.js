/* RVSangeetha local engine: runs the Python DSP in the browser via Pyodide.
   Used when no backend is reachable (e.g. the GitHub Pages build). */
"use strict";

const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
const PY_FILES = [
  "rvsangeetha/__init__.py",
  "rvsangeetha/dsp/__init__.py",
  "rvsangeetha/dsp/declick.py",
  "rvsangeetha/dsp/dynamics.py",
  "rvsangeetha/dsp/filters.py",
  "rvsangeetha/dsp/noise_reduction.py",
  "rvsangeetha/dsp/remaster.py",
  "rvsangeetha/dsp/dispatch.py",
  "local_session.py",
];

function makeLocalEngine() {
  let pyodide = null, sess = null;
  const blobUrls = [];

  async function boot(progress) {
    if (sess) return;
    progress("Loading Python runtime (first visit only)…");
    if (!window.loadPyodide) {
      await new Promise((ok, err) => {
        const s = document.createElement("script");
        s.src = PYODIDE_URL + "pyodide.js";
        s.onload = ok; s.onerror = () => err(new Error("Could not load Pyodide"));
        document.head.appendChild(s);
      });
    }
    pyodide = await loadPyodide({ indexURL: PYODIDE_URL });
    progress("Loading NumPy + SciPy…");
    await pyodide.loadPackage(["numpy", "scipy"]);
    progress("Loading RVSangeetha DSP…");
    for (const f of PY_FILES) {
      const res = await fetch("py/" + f);
      if (!res.ok) throw new Error("Missing " + f);
      const path = "/lib/" + f;
      pyodide.FS.mkdirTree(path.slice(0, path.lastIndexOf("/")));
      pyodide.FS.writeFile(path, await res.text());
    }
    pyodide.runPython("import sys; sys.path.insert(0, '/lib')");
    sess = pyodide.pyimport("local_session");
  }

  function pyErr(e) {
    const m = String(e.message || e);
    // surface only the final python error line
    const lines = m.trim().split("\n");
    const last = lines[lines.length - 1];
    return new Error(last.replace(/^\w+Error: /, ""));
  }

  async function decode(file) {
    const ac = new (window.AudioContext || window.webkitAudioContext)();
    try {
      const buf = await ac.decodeAudioData(await file.arrayBuffer());
      const ch = buf.numberOfChannels, n = buf.length;
      const inter = new Float32Array(n * ch);
      for (let c = 0; c < ch; c++) {
        const d = buf.getChannelData(c);
        for (let i = 0; i < n; i++) inter[i * ch + c] = d[i];
      }
      return { inter, ch, sr: buf.sampleRate };
    } finally { ac.close(); }
  }

  function freeBlobUrls() {
    while (blobUrls.length) URL.revokeObjectURL(blobUrls.pop());
  }
  function blobUrl(bytes, type) {
    const u = URL.createObjectURL(new Blob([bytes.toJs
      ? bytes.toJs() : bytes], { type }));
    blobUrls.push(u);
    return u;
  }
  function pyBytes(proxy) {
    const js = proxy.toJs();          // Uint8Array copy
    proxy.destroy && proxy.destroy();
    return js;
  }

  return {
    local: true,

    async upload(file, progress) {
      await boot(progress);
      progress("Decoding audio…");
      const { inter, ch, sr } = await decode(file);
      try {
        const buf = pyodide.toPy(inter);   // memoryview on the python side
        try { return JSON.parse(sess.load(buf, ch, sr, file.name)); }
        finally { buf.destroy && buf.destroy(); }
      } catch (e) { throw pyErr(e); }
    },

    // JSON API mirror of the server routes
    async call(path, body) {
      try {
        if (path.endsWith("/info")) return JSON.parse(sess.info());
        const mPk = path.match(/\/peaks\?cols=(\d+)/);
        if (mPk) return JSON.parse(sess.peaks(parseInt(mPk[1], 10)));
        if (path.endsWith("/undo")) return JSON.parse(sess.undo());
        if (path.endsWith("/redo")) return JSON.parse(sess.redo());
        if (path.endsWith("/edit"))
          return JSON.parse(sess.edit(body.op, body.start, body.end, body.at));
        if (path.endsWith("/profile"))
          return JSON.parse(sess.capture_profile(body.start, body.end));
        if (path.endsWith("/effect"))
          return JSON.parse(sess.effect(body.name,
            JSON.stringify(body.params || {}), body.start, body.end));
        throw new Error("Unknown call " + path);
      } catch (e) { throw pyErr(e); }
    },

    async audioUrl(start, end) {
      freeBlobUrls();
      return blobUrl(pyBytes(sess.slice_wav(start, end)), "audio/wav");
    },

    async export(fmt, start, end, stem) {
      try {
        const bytes = pyBytes(sess.export(fmt, start, end));
        const ext = fmt === "flac" ? "flac" : "wav";
        return { url: blobUrl(bytes, "application/octet-stream"),
                 filename: `${stem}_remastered.${ext}` };
      } catch (e) { throw pyErr(e); }
    },

    async exportSegments(stem) {
      try {
        const bytes = pyBytes(sess.export_segments());
        return { url: blobUrl(bytes, "application/zip"),
                 filename: `${stem}_segments.zip` };
      } catch (e) { throw pyErr(e); }
    },
  };
}

window.makeLocalEngine = makeLocalEngine;
