"""M12 S0 A3 (reports/M12.md S0.3 + amendment 1, N8/N9): throughput of one qualified EP update at H = 512, input 1760, batch 64.

Graph: 1760 linear sources -> 512 hidden (one-way, dense, W_in ~ N(0, 1/1760)); hidden <-> hidden dense symmetric N(0, 1/512);
hidden <-> 4 outputs N(0, 1/512); hidden and outputs N2 = (a, b) = (1, 1); free -> free scaled to weighted rho_free = 0.9;
bias 0. Drive x ~ N(0, 1), batch 64, cross-entropy nudge (T = 0.1), beta 0.04, tolerance 3e-3, qualified, damping 3, cold start.
Per repetition: t_free, t_plus, t_minus, t_contrast; t_u = their sum. IFT update (f64 only): tight free phase (1e-11) +
free-block adjoint solve. 3 warm-up + 20 measured repetitions, all 23 raw timings kept.
Thread configurations: 8 threads x 1 job; 2 threads x 4 concurrent jobs (each job times itself).
Secondary (informational, not used for U): the S3 draft architecture (entry K=8 filters x 80 positions reading 22 components,
tied across time; K pooling neurons; within-layer full reciprocity; 4 motors).
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

REPS, WARM = (2, 1) if os.environ.get("M12_BENCH_DEV_SHORT") else (20, 3)
NIN, H, NO, B = 1760, 512, 4, 64


def worker(backend, threads, tag, arch, out_dir, dev):
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = str(threads)
    import numpy as np, warnings
    import torch
    torch.set_num_threads(threads)
    from . import m12_common as C
    import cadence as cd
    from cadence.learning import Learner, LearnerConfig, evenodd_neuron_model, time_shared_entry, tie_rates
    C.e_seal()
    rng = np.random.default_rng(0)
    pre, post, eff, kind, ties = [], [], [], [], []
    if arch == "h512":
        ns, nfree = NIN, H + NO
        hid = np.arange(ns, ns + H); out = np.arange(ns + H, ns + H + NO)
        P, Q = np.meshgrid(np.arange(ns), hid, indexing="ij")
        pre.append(P.ravel()); post.append(Q.ravel()); eff.append(rng.normal(0, 1 / np.sqrt(NIN), P.size)); kind.append(np.zeros(P.size, bool))
        iu, ju = np.triu_indices(H, 1); w = rng.normal(0, 1 / np.sqrt(H), len(iu))
        pre += [hid[iu], hid[ju]]; post += [hid[ju], hid[iu]]; eff += [w, w]; kind += [np.ones(len(iu), bool)] * 2
        P, Q = np.meshgrid(hid, out, indexing="ij"); w = rng.normal(0, 1 / np.sqrt(H), P.size)
        pre += [P.ravel(), Q.ravel()]; post += [Q.ravel(), P.ravel()]; eff += [w, w]; kind += [np.ones(P.size, bool)] * 2
        groups = None
    else:  # S3 draft: 22 x 80 sources, 8 filters x 80 positions, 8 pools, 4 motors
        Cn, T, K = 22, 80, 8; ns = Cn * T
        p0, q0, tie, entry = time_shared_entry(Cn, T, K)
        shared = rng.normal(0, 1 / np.sqrt(Cn), K * Cn)
        pre.append(p0); post.append(q0); eff.append(shared[tie]); kind.append(np.zeros(len(p0), bool)); ties.append(tie)
        ent = np.arange(ns, ns + K * T); pool = np.arange(ns + K * T, ns + K * T + K); out = np.arange(ns + K * T + K, ns + K * T + K + NO)
        def recip(a_, b_, scale):
            P_, Q_ = np.meshgrid(a_, b_, indexing="ij"); w_ = rng.normal(0, scale, P_.size)
            pre.extend([P_.ravel(), Q_.ravel()]); post.extend([Q_.ravel(), P_.ravel()]); eff.extend([w_, w_]); kind.extend([np.ones(P_.size, bool)] * 2)
            ties.extend([-np.ones(P_.size, np.int64)] * 2)
        iu, ju = np.triu_indices(len(ent), 1); w = rng.normal(0, 1 / np.sqrt(len(ent)), len(iu))
        pre += [ent[iu], ent[ju]]; post += [ent[ju], ent[iu]]; eff += [w, w]; kind += [np.ones(len(iu), bool)] * 2; ties += [-np.ones(len(iu), np.int64)] * 2
        iu, ju = np.triu_indices(K, 1); w = rng.normal(0, 1 / np.sqrt(K), len(iu))
        pre += [pool[iu], pool[ju]]; post += [pool[ju], pool[iu]]; eff += [w, w]; kind += [np.ones(len(iu), bool)] * 2; ties += [-np.ones(len(iu), np.int64)] * 2
        for k in range(K):
            recip(entry[k], pool[k:k + 1], 1 / np.sqrt(T))
        recip(pool, out, 1 / np.sqrt(K))
        nfree = K * T + K + NO
        groups = np.concatenate(ties)
    pre, post, eff, kind = map(np.concatenate, (pre, post, eff, kind))
    n = ns + nfree
    conn = cd.Connectome.from_synapses(n, pre=pre, post=post)
    order = np.lexsort((pre, post)); assert np.array_equal(conn.pre, pre[order]) and np.array_equal(conn.post, post[order])
    eff, kind = eff[order], kind[order]
    if groups is not None: groups = groups[order]
    a = np.ones(n); b = np.r_[np.zeros(ns), np.ones(nfree)]
    model = evenodd_neuron_model(a, b, linear=range(ns), dt=0.5); lip = model.lipschitz_per_neuron(n)
    rowm = np.bincount(conn.post, np.abs(eff) * lip[conn.pre] * kind, minlength=n)[ns:].max()
    eff = np.where(kind, eff * 0.9 / rowm, eff)
    kw = {"cpu64": {}, "torch64": {"backend": "torch", "device": "cpu", "precision": "float64"},
          "torch32": {"backend": "torch", "device": "cpu", "precision": "float32"}}[backend]
    brain = cd.NeuralGraph(conn, model, efficacy=eff, bias=np.zeros(n), dense_limit=4096, **kw)
    cert = cd.certificate(brain, sources=range(ns))
    lc = LearnerConfig(beta=0.04, eta=0.0, eta_bias=0.0, centered=True, free_steps=50000, nudged_steps=50000, tolerance=3e-3,
                       nudge="cross_entropy", temperature=0.1, normalize=0.0, normalize_floor=1e-3, momentum=0.0, decay=0.0,
                       scale_cap=1e9, qualified=True, damping=3, surprise=False)
    pn = np.zeros(n, bool); pn[ns:] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = Learner(brain, list(out), config=lc, plastic_synapses=np.ones(len(eff), bool), plastic_neurons=pn, reciprocal=True,
                     tie_groups=groups, synapse_rate=None if groups is None else tie_rates(groups))
    X = np.zeros((B, n)); X[:, :ns] = rng.normal(size=(B, ns)); y = rng.integers(0, NO, B)
    rows = []
    for rep in range(WARM + REPS):
        la = os.getloadavg()[0]
        t0 = time.perf_counter(); free = lr.free(X); t1 = time.perf_counter(); tgt = lr.targets(y)
        plus = lr.nudged(X, free, tgt, sign=1.0); t2 = time.perf_counter(); minus = lr.nudged(X, free, tgt, sign=-1.0); t3 = time.perf_counter()
        e, nb = lr.contrast(free, plus, minus); np.asarray(e); t4 = time.perf_counter()
        rows.append({"rep": rep, "warmup": rep < WARM, "loadavg1": la, "t_free": t1 - t0, "t_plus": t2 - t1, "t_minus": t3 - t2,
                     "t_contrast": t4 - t3, "t_u": t4 - t0, "steps": [int(getattr(s, "steps", -1) or -1) for s in (free, plus, minus)]})
    ift_rows = []
    if backend != "torch32" and arch == "h512":
        from .ift import loss_grad_s
        for rep in range(WARM + REPS):
            t0 = time.perf_counter()
            eq = lr.brain.equilibrate(X, budget=50000, tolerance=1e-11, damping=3)
            s = np.asarray(eq.state.activation); v = np.asarray(eq.state.v)
            sp = model.slope_at(v)[:, ns:]; gs, _ = loss_grad_s(lr, s, y); W = np.asarray(lr.brain.dense())[ns:, ns:]   # W[pre, post]
            JT = np.eye(nfree)[None] - sp[:, :, None] * W[None]
            lam = np.linalg.solve(JT, (sp * gs[:, ns:])[..., None])[..., 0]
            g_edges = (s[:, conn.pre] * np.pad(lam, ((0, 0), (ns, 0)))[:, conn.post]).mean(0)
            t1 = time.perf_counter()
            ift_rows.append({"rep": rep, "warmup": rep < WARM, "t_ift": t1 - t0, "converged": bool(np.all(eq.converged)), "steps": int(eq.steps)})
    med = lambda key, rs: float(np.median([r[key] for r in rs if not r["warmup"]]))
    res = {"arch": arch, "backend": backend, "threads": threads, "tag": tag, "n": n, "sources": ns, "free": nfree, "synapses": int(conn.synapses),
           "batch": B, "certificate": cert.to_dict(), "blocked": bool(brain._blocked), "fused_numba_used": False,
           "config": {"beta": 0.04, "tolerance": 3e-3, "nudge": "cross_entropy", "temperature": 0.1, "damping": 3, "qualified": True,
                      "a": 1.0, "b": 1.0, "dt": 0.5, "rho_free": 0.9, "init": "W_in N(0,1/1760); hidden, readout N(0,1/512) symmetric; bias 0"},
           "weights_sha256": C.sha(eff), "inputs_sha256": C.sha(X, y), "rows": rows, "ift_rows": ift_rows,
           "median": {k: med(k, rows) for k in ("t_free", "t_plus", "t_minus", "t_contrast", "t_u")} | ({"t_ift": med("t_ift", ift_rows)} if ift_rows else {})}
    C.write_receipt(out_dir, f"{arch}_{backend}_t{threads}_{tag}", res, dev=dev)
    print(json.dumps({"arch": arch, "backend": backend, "threads": threads, "tag": tag, **res["median"]}), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--worker", nargs=4, metavar=("BACKEND", "THREADS", "TAG", "ARCH"))
    ap.add_argument("--dev", action="store_true"); ap.add_argument("--out", default=None)
    ap.add_argument("--backends", default="cpu64,torch64,torch32"); ap.add_argument("--configs", default="8x1,2x4")
    a = ap.parse_args()
    from . import m12_common as C
    out = Path(a.out) if a.out else C.OUT / "S0" / ("dev" if a.dev else "A3") / C.run_commit()
    if a.worker:
        bk, th, tag, arch = a.worker; worker(bk, int(th), tag, arch, out, a.dev); return
    C.e_seal()
    if not a.dev: C.start_clock("m12_s0_bench")
    t0 = time.time()
    for bk in a.backends.split(","):
        for cfg in a.configs.split(","):
            th, jobs = map(int, cfg.split("x"))
            names = [f"h512_{bk}_t{th}_j{jobs}_{i}" for i in range(jobs)]
            if all((out / f"{nm}.json").exists() for nm in names):
                continue  # resume
            procs = [subprocess.Popen([sys.executable, "-m", "ep_eeg.m12_s0_bench", "--worker", bk, str(th), f"j{jobs}_{i}", "h512", "--out", str(out)]
                                      + (["--dev"] if a.dev else []), cwd=C.ROOT) for i in range(jobs)]
            print(f"{bk} {cfg}: pids {[p.pid for p in procs]}", flush=True)
            for p in procs:
                if p.wait() != 0: raise SystemExit(f"worker {p.pid} failed")
    for bk in ("cpu64", "torch64"):
        if not (out / f"s3draft_{bk}_t8_j1_0.json").exists():
            p = subprocess.run([sys.executable, "-m", "ep_eeg.m12_s0_bench", "--worker", bk, "8", "j1_0", "s3draft", "--out", str(out)] + (["--dev"] if a.dev else []), cwd=C.ROOT)
            if p.returncode != 0: print(f"s3draft {bk} failed (informational only)", flush=True)
    if not a.dev:
        C.ledger("M12-S0", "A3_bench", "NA", 1, time.time() - t0, "done", "A3 throughput H=512 / 1760 / batch 64 on cpu64, torch64, torch32 (8x1, 2x4); GPU not available")


if __name__ == "__main__":
    main()
