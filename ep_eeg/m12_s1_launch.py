"""Launch the S1 jobs, at most P concurrent subprocesses (2 threads each), PIDs tracked in a file; resumable
(finished receipts are skipped, unfinished jobs resume from their checkpoints)."""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path

def jobs(tag):
    J = [("T2", a, s, None) for a in ("ift", "ep", "shuffled") for s in (0, 1, 2)]
    J += [("T1", "ep", s, H) for H in (32, 8) for s in (0, 1, 2)]
    J += [("T2", "random", s, None) for s in (0, 1, 2)]
    J += [("T1", a, s, H) for a in ("random", "randodd") for H in (8, 32) for s in (0, 1, 2)]
    return J

def main():
    tag = sys.argv[1]; P = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    extra = sys.argv[3:]
    logd = Path("/workspace/m12_logs/S1") / tag; logd.mkdir(parents=True, exist_ok=True)
    pending = jobs(tag); running = {}
    while pending or running:
        for pid, (p, j) in list(running.items()):
            if p.poll() is not None:
                print(time.strftime("%H:%M:%S"), "finished", j, "rc", p.returncode, flush=True); running.pop(pid)
        while pending and len(running) < P:
            task, arm, seed, H = pending.pop(0)
            name = f"{task}_{arm}" + (f"_H{H}" if H else "") + f"_s{seed}"
            cmd = [sys.executable, "-m", "ep_eeg.m12_s1", "run", "--task", task, "--arm", arm, "--seed", str(seed), "--tag", tag] + (["--H", str(H)] if H else []) + extra
            p = subprocess.Popen(cmd, stdout=open(logd / f"{name}.log", "a"), stderr=subprocess.STDOUT, cwd="/workspace/EP-EEG")
            running[p.pid] = (p, name); print(time.strftime("%H:%M:%S"), "started", name, p.pid, flush=True)
            (logd / "pids.json").write_text(json.dumps({str(k): v[1] for k, v in running.items()}))
        (logd / "pids.json").write_text(json.dumps({str(k): v[1] for k, v in running.items()}))
        time.sleep(5)
    print("all done", flush=True)

if __name__ == "__main__":
    main()
