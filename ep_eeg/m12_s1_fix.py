"""M12 S1 one fix round (reports/M12.md S1.6, S1.10, amendments 6-8). Selection on dev seed 101 (dev-val split),
grid beta {0.01, 0.04, 0.1} x eta {0.02, 0.05, 0.1, 0.5} x normalize {0, 0.999} x (a, b) of the non-output free neurons
{N1, N2}; then the official seeds rerun with the chosen setting, marked 'after fix'. Implemented by wrapping m12_s1
(the first-round code is not modified)."""
from __future__ import annotations
import dataclasses, itertools, json, os, sys, time
from pathlib import Path
import numpy as np
from . import m12_s1 as S
from . import m12_common as C

GRID = list(itertools.product((0.01, 0.04, 0.1), (0.02, 0.05, 0.1, 0.5), (0.0, 0.999), ("N1", "N2")))
SETTING = {}
_b1, _b2, _ep, _t1d, _t2d, _wr = S.build_t1, S.build_t2, S.ep_step, S.t1_data, S.t2_data, C.write_receipt


def _cfg(g):
    if SETTING:
        g["lr"].config = dataclasses.replace(g["lr"].config, beta=SETTING["beta"], normalize=SETTING["normalize"])
    return g


def build_t1(H, init_seed, hid_ab=S.NEUR["N2"], out_ab=S.NEUR["N1"], backend="torch64", eta=S.ETA, odd=False):
    ab = S.NEUR[SETTING["ab"]] if SETTING else hid_ab
    return _cfg(_b1(H, init_seed, hid_ab=ab, out_ab=out_ab, backend=backend, eta=eta, odd=odd))


def build_t2(init_seed, ab=S.NEUR["N2"], out_ab=S.NEUR["N2"], backend="torch64", eta=S.ETA):
    a2 = S.NEUR[SETTING["ab"]] if SETTING else ab
    g = _b2(init_seed, ab=a2, out_ab=out_ab, backend=backend, eta=eta)
    return _cfg(g) if backend == "torch64" else g


def ep_step(g, D, tgt, eta=S.ETA):
    if not SETTING or SETTING["normalize"] == 0.0: return _ep(g, D, tgt, eta)
    lr = g["lr"]; free = lr.free(D)
    plus = lr.nudged(D, free, tgt, 1.0); minus = lr.nudged(D, free, tgt, -1.0)
    lr.update(free, plus, minus)          # RMS normalisation lives in update (amendment 8)
    return free, [int(getattr(s, "steps", -1) or -1) for s in (free, plus, minus)]


def _swap(d, keys):
    d = dict(d)
    for k in keys: d[k + "_te"] = d[k + "_dv"]
    return d


def write_receipt(out_dir, name, payload, dev=False):
    payload = dict(payload); payload["fix_setting"] = dict(SETTING); payload["config"] = {**payload["config"], "beta": SETTING["beta"], "normalize": SETTING["normalize"],
                                                                                          "free_ab": S.NEUR[SETTING["ab"]]}
    if SETTING.get("select"): payload["evaluated_on"] = "dev-val split of seed 101 (selection only)"
    return _wr(out_dir, name, payload, dev=dev)


def install(setting):
    SETTING.clear(); SETTING.update(setting)
    S.build_t1, S.build_t2, S.ep_step = build_t1, build_t2, ep_step
    C.write_receipt = write_receipt
    if setting.get("select"):
        S.t1_data = lambda seed: _swap(_t1d(seed), ("x", "y", "t"))
        S.t2_data = lambda seed: _swap(_t2d(seed), ("X", "y", "logv", "Z", "in"))


def main():
    # python -m ep_eeg.m12_s1_fix select <task> <H|0> <grid index>   |   official <task> <arm> <seed> <H|0> <setting json>
    mode = sys.argv[1]
    if mode == "launch": return launch(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    C.e_seal(); commit = C.run_commit()
    if mode == "select":
        task, H, gi = sys.argv[2], int(sys.argv[3]) or None, int(sys.argv[4]); beta, eta, nz, ab = GRID[gi]
        install({"beta": beta, "eta": eta, "normalize": nz, "ab": ab, "select": True, "grid_index": gi})
        S.run_job(task, "ep", 101, H=H, eta=eta, tag=f"fixsel_g{gi}", dev=True, out_dir=str(S.S1 / commit / "fixsel" / f"g{gi}"),
                  ckpt_dir=f"/workspace/m12_ckpt/S1/fixsel/g{gi}")
    else:
        task, arm, seed, H = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]) or None; st = json.loads(sys.argv[6])
        install({**st, "select": False})
        S.run_job(task, arm, seed, H=H, eta=st["eta"], tag="fix", out_dir=str(S.S1 / commit / "fix"), ckpt_dir="/workspace/m12_ckpt/S1/fix")


if __name__ == "__main__":
    main()


def launch(jobfile, P=4, logname="fix"):
    """Run argv lines from `jobfile` (python -m ep_eeg.m12_s1_fix ...), at most P at once; PIDs tracked in a file."""
    import subprocess
    logd = Path("/workspace/m12_logs/S1") / logname; logd.mkdir(parents=True, exist_ok=True)
    pending = [l.split() for l in Path(jobfile).read_text().splitlines() if l.strip()]; running = {}
    while pending or running:
        for pid, (p, nm) in list(running.items()):
            if p.poll() is not None: print(time.strftime("%H:%M:%S"), "finished", nm, "rc", p.returncode, flush=True); running.pop(pid)
        while pending and len(running) < P:
            args = pending.pop(0); nm = "_".join(args[:5])
            p = subprocess.Popen([sys.executable, "-m", "ep_eeg.m12_s1_fix", *args], stdout=open(logd / f"{nm}.log", "a"),
                                 stderr=subprocess.STDOUT, cwd="/workspace/EP-EEG")
            running[p.pid] = (p, nm); print(time.strftime("%H:%M:%S"), "started", nm, p.pid, flush=True)
        (logd / "pids.json").write_text(json.dumps({str(k): v[1] for k, v in running.items()}))
        time.sleep(5)
    print("all done", flush=True)
