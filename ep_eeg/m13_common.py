"""M13 shared pieces: lead geometry, hold-out patterns, spherical-spline interpolation,
the reconstruction score, data rows, the E seal and a synthetic dev generator.

Nothing here reads the E session. Real data comes from the frozen eval layer through
ep_eeg.data.load_subject (T runs 0-3 train, T runs 4-5 test); the dev generator is
only for harness checks and its receipts are marked dev=true.
"""
from __future__ import annotations
import hashlib, json, os, platform, re, subprocess, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEADS = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
         "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]
N_LEADS = 22
# standard_1005 positions (m), copied from MNE 1.13 so the geometry cannot drift with MNE versions
XYZ = np.array([
    [0.00031, 0.05851, 0.06646], [-0.06018, 0.02272, 0.05554], [-0.03406, 0.02601, 0.07999],
    [0.00038, 0.02739, 0.08867], [0.03478, 0.02644, 0.07881], [0.06229, 0.02372, 0.05563],
    [-0.08028, -0.01376, 0.02916], [-0.06536, -0.01163, 0.06436], [-0.03616, -0.00998, 0.08975],
    [0.0004, -0.00917, 0.10024], [0.03767, -0.00962, 0.08841], [0.06712, -0.0109, 0.06358],
    [0.08346, -0.01278, 0.02921], [-0.06356, -0.04701, 0.06562], [-0.03551, -0.04729, 0.09132],
    [0.00039, -0.04732, 0.09943], [0.03838, -0.04707, 0.0907], [0.06661, -0.04664, 0.06558],
    [-0.02862, -0.08052, 0.07544], [0.00032, -0.08111, 0.08262], [0.03192, -0.08049, 0.07672],
    [0.00022, -0.10218, 0.05061]])
M1_XYZ = np.array([-0.08608, -0.04499, -0.06799])        # left mastoid: IV-2a reference
HEAD_CENTER = np.array([0.00082, -0.01623, -0.00116])     # LSQ sphere fit to all 1005 points
HEAD_R = 0.0986
STEP = 5           # row grid: every 5th sample of the 750-sample segment (50 Hz)
RIDGE_STEP = 10    # ridge readouts are fitted on every 10th training sample


def unit(xyz):
    v = xyz - HEAD_CENTER
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ---------------------------------------------------------------- patterns
def patterns():
    """P1: each lead alone (22). P3: each lead with its two nearest leads (Euclidean, standard_1005),
    duplicates removed in lead order. Returns dict name -> list of sorted index tuples."""
    D = np.linalg.norm(XYZ[:, None] - XYZ[None], axis=-1)
    p1 = [(c,) for c in range(N_LEADS)]
    p3, seen = [], set()
    for c in range(N_LEADS):
        nn = [j for j in np.argsort(D[c]) if j != c][:2]
        t = tuple(sorted([c, *map(int, nn)]))
        if t not in seen:
            seen.add(t); p3.append(t)
    return {"P1": p1, "P3": p3}


def all_patterns():
    p = patterns(); return [("P1", t) for t in p["P1"]] + [("P3", t) for t in p["P3"]]


def derangement(seed, n=N_LEADS):
    """Seeded permutation with no fixed point: lead c is placed at the scalp position of lead pi[c]."""
    rng = np.random.default_rng(7000 + seed)
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)): return p


# ---------------------------------------------------------------- spherical spline (Perrin et al. 1989)
def _g(cosang, m=4, n_terms=50):
    from numpy.polynomial import legendre
    n = np.arange(1, n_terms + 1)
    coef = np.r_[0.0, (2 * n + 1) / (n * (n + 1)) ** m / (4 * np.pi)]
    return legendre.legval(np.clip(cosang, -1, 1), coef)


def spline_matrix(hidden, xyz=XYZ, lam=1e-5):
    """Linear map (n_hidden x n_visible) from visible to hidden leads for positions xyz.
    Fixed by geometry alone: no data enter it."""
    hid = list(hidden); vis = [c for c in range(N_LEADS) if c not in hid]
    U = unit(xyz)
    G = _g(U[vis] @ U[vis].T) + lam * np.eye(len(vis))
    A = np.zeros((len(vis) + 1, len(vis) + 1)); A[:-1, :-1] = G; A[:-1, -1] = 1; A[-1, :-1] = 1
    Ainv = np.linalg.pinv(A)[:, :-1]                      # (nv+1, nv): [c; c0] = Ainv @ v
    Gh = np.c_[_g(U[hid] @ U[vis].T), np.ones(len(hid))]  # (nh, nv+1)
    return Gh @ Ainv


# ---------------------------------------------------------------- score
def r2_cols(yhat, y):
    """R^2 per column on the evaluation rows (SST around the evaluation mean)."""
    yhat = np.asarray(yhat, float); y = np.asarray(y, float)
    return 1 - ((yhat - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


# ---------------------------------------------------------------- data rows
def rows(seg, step=STEP, offset=0):
    """(n,22,750) z-scored segments -> (n*ceil(750/step), 22) sample rows."""
    return np.ascontiguousarray(seg[:, :, offset::step].transpose(0, 2, 1).reshape(-1, seg.shape[1]))


def load_real(subject, split="t04_val45"):
    """t04_val45 (official): train runs 0-3, test runs 4-5.  dev_r02_r3 (E0.6 fix round): of the
    run 0-3 training trials (run order, 48 per run), train on runs 0-2 and test on run 3; runs 4-5 unused.
    Normalisation statistics are the frozen layer's (fit on runs 0-3) in both cases."""
    from ep_eeg.data import load_subject
    d = load_subject(subject)
    tr = d["train"]["seg"].astype(np.float64)
    if split == "t04_val45":
        return {"train": tr, "test": d["val"]["seg"].astype(np.float64),
                "source": f"A{subject:02d}T runs 0-3 / 4-5 (frozen eval layer)"}
    if split == "dev_r02_r3":
        assert len(tr) == 192
        return {"train": tr[:144], "test": tr[144:], "source": f"A{subject:02d}T runs 0-2 / 3 (E0.6 dev split)"}
    raise ValueError(split)


def synth_subject(seed, n_train=192, n_test=96, n_src=40, noise=0.1):
    """DEV ONLY. Point sources inside the head sphere, 1/d^2 gain, referenced to the left mastoid,
    AR(2) rhythms 8-30 Hz plus broadband sources, sensor noise; then the frozen band-pass and a
    z-score with training statistics. Has real spatial structure, so the controls should order."""
    from scipy.signal import lfilter, sosfiltfilt
    from eval.preprocess import _SOS, FS
    rng = np.random.default_rng(9000 + seed)
    d = rng.normal(size=(n_src, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    d[:, 2] = np.abs(d[:, 2]) * 0.8 + 0.2; d /= np.linalg.norm(d, axis=1, keepdims=True)
    src = HEAD_CENTER + d * HEAD_R * rng.uniform(0.55, 0.8, size=(n_src, 1))
    def gain(e): return 1.0 / np.linalg.norm(e[None] - src, axis=-1) ** 2
    Gm = np.stack([gain(e) for e in XYZ]) - gain(M1_XYZ)[None]          # (22, n_src)
    T = 750 + 500
    def sim(n):
        out = np.empty((n, N_LEADS, 750))
        for i in range(n):
            S = np.empty((n_src, T))
            for s in range(n_src):
                e = rng.normal(size=T)
                if s % 2 == 0:
                    f = rng.uniform(8, 30); r = 0.97
                    a = [1, -2 * r * np.cos(2 * np.pi * f / FS), r * r]
                    S[s] = lfilter([1], a, e)
                else:
                    S[s] = lfilter([1], [1, -0.95], e)
            X = Gm @ S + noise * np.std(Gm @ S) * rng.normal(size=(N_LEADS, T))
            out[i] = sosfiltfilt(_SOS, X, axis=1)[:, 250:1000]
        return out
    tr, te = sim(n_train), sim(n_test)
    m = tr.transpose(1, 0, 2).reshape(N_LEADS, -1).mean(1)[None, :, None]
    s = tr.transpose(1, 0, 2).reshape(N_LEADS, -1).std(1)[None, :, None]
    return {"train": (tr - m) / s, "test": (te - m) / s, "source": f"synthetic dev seed {seed}"}


# ---------------------------------------------------------------- receipts
E_PATTERN = re.compile(r"A0\dE", re.I)


def e_seal():
    roots = [Path(os.environ.get("MNE_DATA", "/workspace/mne_data")), ROOT / "cache", ROOT / "data"]
    hits = [str(p) for r in roots if r.exists() for p in r.rglob("*") if E_PATTERN.search(p.name)]
    return {"method": "filename", "roots": [str(r) for r in roots], "matches": hits}


def git(*a, cwd=ROOT):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True).stdout.strip()


def env_info():
    import cadence, scipy, sklearn
    cad = Path(cadence.__file__).resolve().parents[2]
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "cadence_version": cadence.__version__, "cadence_commit": git("rev-parse", "HEAD", cwd=cad),
            "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "python": platform.python_version(), "cpu": os.cpu_count(), "loadavg": os.getloadavg()}


def sha(*arrays):
    h = hashlib.sha256()
    for a in arrays: h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _default(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    raise TypeError(type(o))


def write_receipt(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(payload, default=_default, ensure_ascii=False))
    tmp.replace(path); return path


def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
