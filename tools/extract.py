"""Pull real signals, spectra and facts from the 40 CWRU recordings into ../data.js.

Run from anywhere:  python3 tools/extract.py
Uses the same loader, resampling and features as the Bearing Collective experiment.
"""
import json, pathlib, sys
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly, welch, butter, sosfiltfilt, hilbert

SITE = pathlib.Path(__file__).resolve().parents[1]
IOC = SITE.parent
sys.path.insert(0, str(IOC))
from bearing_fl.data import read_de            # noqa: E402
from bearing_fl.features import fft_features   # noqa: E402

RAW = IOC / "data" / "raw"
FS = 12000
bench = json.load(open(IOC / "artifacts" / "benchmark.json"))
recs = bench["recordings"]
CLASS = ["Normal", "Inner Race", "Outer Race", "Ball"]

def load(fid):
    x, _ = read_de(RAW / f"{fid}.mat")
    rec = next(r for r in recs if r["file_id"] == fid)
    if rec["source_rate"] == 48000:                      # Normal baseline: 48 kHz → 12 kHz, as in the experiment
        x = resample_poly(x, 1, 4)
    m = loadmat(RAW / f"{fid}.mat")
    rpm = [v for k, v in m.items() if k.endswith("RPM")]
    return x, int(np.ravel(rpm[0])[0]) if rpm else None

def r(a, d=4): return [round(float(v), d) for v in a]

# ---------- file grid + per-file facts ----------
files = []
for rec in recs:
    x, rpm = load(rec["file_id"])
    files.append({"id": rec["file_id"], "load": rec["load_hp"], "cls": rec["label"], "dia": rec["diameter"],
                  "rpm": rpm, "sec": round(len(x) / FS, 2), "mb": round(rec["download_bytes"] / 1e6, 2),
                  "rms": round(float(np.sqrt(np.mean((x - x.mean()) ** 2))), 4),
                  "kurt": round(float(np.mean(((x - x.mean()) / x.std()) ** 4)), 2), "sha": rec["sha256"][:12]})

# ---------- one recording per class at 0 HP, 0.007" ----------
pick = {"Normal": 97, "Inner Race": 105, "Outer Race": 130, "Ball": 118}
sig, spec, env, feats = {}, {}, {}, {}
rpm0 = None
sos = butter(4, [1500, 5000], btype="bandpass", fs=FS, output="sos")
for name, fid in pick.items():
    x, rpm = load(fid); rpm0 = rpm0 or rpm
    x = x - x.mean()
    s0 = 24000
    sig[name] = r(x[s0:s0 + 1200])                        # 0.1 s of real signal
    f, p = welch(x, fs=FS, nperseg=4096)
    db = 10 * np.log10(p + 1e-14)
    edges = np.linspace(0, len(f) - 1, 241).astype(int)
    spec[name] = r([db[a:b].mean() for a, b in zip(edges[:-1], edges[1:])], 2)
    e = np.abs(hilbert(sosfiltfilt(sos, x))); e -= e.mean()
    seg = 2 ** 14; n = len(e) // seg
    P = np.mean([np.abs(np.fft.rfft(e[i * seg:(i + 1) * seg] * np.hanning(seg))) ** 2 for i in range(n)], axis=0)
    fe = np.fft.rfftfreq(seg, 1 / FS)
    keep = fe <= 500
    Pk = P[keep] / P[keep][fe[keep] > 20].max()
    env[name] = {"f": r(fe[keep], 2), "p": r(Pk, 4)}
    w = np.lib.stride_tricks.sliding_window_view(x, 2048)[::1024][:60]
    feats[name] = r(fft_features(w).mean(axis=0), 4)

# characteristic frequencies for the drive-end SKF 6205-2RS (multiples of shaft speed, from the CWRU site)
shaft = rpm0 / 60
fault_hz = {"shaft": shaft, "BPFI": 5.4152 * shaft, "BPFO": 3.5848 * shaft, "BSF": 4.7135 * shaft}

# ---------- severity: inner race at 0 HP, three sizes ----------
sev = {}
for dia, fid in [(0.007, 105), (0.014, 169), (0.021, 209)]:
    x, _ = load(fid); x = x - x.mean()
    sev[str(dia)] = {"sig": r(x[24000:25200]), "rms": round(float(np.sqrt(np.mean(x * x))), 4)}

# ---------- windowing + split illustration on a real recording ----------
x105, _ = load(105)
win_demo = r(x105[0:12288:4] - x105[0:12288].mean())      # 1.02 s, decimated for drawing

# ---------- 1-second real clips (stored as int, value × 1000) for listening and dragging ----------
def clip(fid, start=24000):
    x, _ = load(fid); x = x - x.mean()
    return [int(round(v * 1000)) for v in x[start:start + FS]]
clips = {name: clip(fid) for name, fid in pick.items()}
sev_clips = {str(d): clip(fid) for d, fid in [(0.007, 105), (0.014, 169), (0.021, 209)]}

out = {"fs": FS, "clips": clips, "sev_clips": sev_clips, "files": files, "sig": sig, "spec": spec, "env": env, "feats": feats, "fault_hz": {k: round(v, 1) for k, v in fault_hz.items()},
       "rpm0": rpm0, "sev": sev, "win_demo": win_demo, "len105": len(x105),
       "split": {"train": .55, "val": .20, "test": .25, "gap": 2048}, "window": 2048, "hop": 1024}
(SITE / "data.js").write_text("window.DS = " + json.dumps(out, separators=(",", ":")) + ";")
print("ok", len(files), "files;", {k: v for k, v in out["fault_hz"].items()}, "rpm", rpm0)
# quick check: where are the envelope peaks?
for name in pick:
    f = np.array(env[name]["f"]); p = np.array(env[name]["p"]); m = (f > 50) & (f < 400)
    print(name, "envelope peak at", f[m][np.argmax(p[m])], "Hz")
print("rpms by load:", sorted({(f["load"], f["rpm"] or 0) for f in files}))
print("IR rms by size:", {k: v["rms"] for k, v in sev.items()})
