"""M14 tables: grid selection on T validation (--select), seeds at the selection (--seeds), E gate (--gate).

Selection rule (reports/M14.md 1.3): per arm, the (eta, U) with the highest 9-subject mean T-validation accuracy
on seed 0. Writes receipts/M14/selected.json.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "receipts" / "M14"
ARMS = ("A", "B", "C", "nolake", "C_nolake", "randlake")


def load(d, pattern="*.json"):
    out = {}
    for f in glob.glob(str(R / d / pattern)):
        r = json.loads(Path(f).read_text())
        out[(r["arm"], r["subject"], r["seed"], float(r["eta"]))] = r
    return out


def curve_table(recs, seed=0):
    """{(arm, eta): {U: [acc per subject 1..9]}}"""
    tab = {}
    for (arm, s, sd, eta), r in recs.items():
        if sd != seed: continue
        for row in r["train"]["curve"]:
            tab.setdefault((arm, eta), {}).setdefault(row["u"], {})[s] = row
    return tab


def select(recs):
    tab = curve_table(recs); sel = {}
    for arm in ARMS:
        best = None
        for (a, eta), byU in tab.items():
            if a != arm: continue
            for U, rows in byU.items():
                if len(rows) < 9: continue
                m = float(np.mean([rows[s]["acc"] for s in range(1, 10)]))
                if best is None or m > best[0]: best = (m, eta, U)
        if best: sel[arm] = {"eta": best[1], "U": int(best[2]), "val_acc_mean_seed0": best[0]}
    return sel, tab


def wilcoxon(a, b):
    from scipy.stats import wilcoxon as W
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0): return 1.0
    return float(W(a, b, alternative="greater").pvalue)


SUBJ = list(range(1, 10))
PAIRS = (("A", "nolake", "主判据"), ("C", "A", "C 对 A"), ("B", "A", "B 对 A"), ("C", "C_nolake", "C 里湖的贡献"), ("A", "randlake", "学过的湖对随机湖"))


def row_at(r, U):
    for row in r["train"]["curve"]:
        if row["u"] == U: return row
    raise KeyError(U)


def seed_records(sel):
    """{(arm, subject, seed): T-validation receipt at the selected eta} — seed 0 from the grid, seeds 1-2 from seeds/."""
    grid, seeds = load("grid"), load("seeds"); out = {}
    for arm, c in sel.items():
        for s in SUBJ:
            for sd in (0, 1, 2):
                r = (grid if sd == 0 else seeds).get((arm, s, sd, float(c["eta"])))
                if r is not None: out[(arm, s, sd)] = r
    return out


def gate_records():
    out = {}
    for f in glob.glob(str(R / "gate" / "*.json")):
        r = json.loads(Path(f).read_text()); out[(r["arm"], r["subject"], r["seed"])] = r
    return out


def fbcsp_E():
    r = json.loads((ROOT / "holdout_receipts" / "M0__fbcsp__db12387__bf21a9e8.json").read_text())
    return {s: float(np.mean([x["acc"] for x in r["runs"] if x["subject"] == s])) for s in SUBJ}, r


def verdict(a, b):
    d = np.asarray(a) - np.asarray(b); wins, losses = int((d > 0).sum()), int((d < 0).sum())
    return {"mean_diff": float(d.mean()), "wins": wins, "losses": losses, "ties": 9 - wins - losses,
            "p_greater": wilcoxon(a, b), "p_less": wilcoxon(b, a)}


def gate_summary():
    sel = json.loads((R / "selected.json").read_text()); tv, gt = seed_records(sel), gate_records()
    missing = [(a, s, sd) for a in sel for s in SUBJ for sd in (0, 1, 2) if (a, s, sd) not in gt or (a, s, sd) not in tv]
    if missing: raise SystemExit(f"missing {len(missing)} runs, e.g. {missing[:5]}")
    arms = {}
    for arm, c in sel.items():
        U = c["U"]; per = {}
        for s in SUBJ:
            E = [gt[(arm, s, sd)]["test"] for sd in (0, 1, 2)]
            V = [row_at(tv[(arm, s, sd)], U) for sd in (0, 1, 2)]
            # train accuracy is logged every 1000 updates; take the last logged value at or before U_sel
            T = [max((row for row in tv[(arm, s, sd)]["train"]["curve"] if row["u"] <= U and row.get("train_acc") is not None),
                     key=lambda row: row["u"], default={}).get("train_acc") for sd in (0, 1, 2)]
            per[s] = {"E_acc": float(np.mean([e["acc"] for e in E])), "E_acc_seeds": [e["acc"] for e in E],
                      "E_kappa": float(np.mean([e["kappa"] for e in E])),
                      "E_settle": float(np.mean([e["settle_rate"] for e in E])),
                      "E_rejected": int(sum(e["rejected_trials"] for e in E)),
                      "val_acc": float(np.mean([v["acc"] for v in V])),
                      "train_acc": (float(np.mean(T)) if all(t is not None for t in T) else None),
                      "train_acc_u": max((row["u"] for row in tv[(arm, s, 0)]["train"]["curve"] if row["u"] <= U and row.get("train_acc") is not None), default=None),
                      **{k: float(np.mean([v[k] for v in V])) for k in V[0] if k.startswith("front_")}}
        seed_means = [float(np.mean([gt[(arm, s, sd)]["test"]["acc"] for s in SUBJ])) for sd in (0, 1, 2)]
        arms[arm] = {"eta": c["eta"], "U": U, "per_subject": per,
                     "E_acc": float(np.mean([per[s]["E_acc"] for s in SUBJ])), "E_acc_seed_sd": float(np.std(seed_means, ddof=1)),
                     "E_kappa": float(np.mean([per[s]["E_kappa"] for s in SUBJ])),
                     "E_kappa_seed_sd": float(np.std([np.mean([gt[(arm, s, sd)]["test"]["kappa"] for s in SUBJ]) for sd in (0, 1, 2)], ddof=1)),
                     "val_acc": float(np.mean([per[s]["val_acc"] for s in SUBJ])),
                     "val_kappa": float(np.mean([row_at(tv[(arm, s, sd)], U)["kappa"] for s in SUBJ for sd in (0, 1, 2)])),
                     "E_settle": float(np.mean([per[s]["E_settle"] for s in SUBJ])),
                     "E_rejected": int(sum(per[s]["E_rejected"] for s in SUBJ)),
                     "weights_sha256": {f"A{s:02d}_s{sd}": gt[(arm, s, sd)]["weights_sha256"] for s in SUBJ for sd in (0, 1, 2)},
                     "train_windows_identical": all(gt[(arm, s, sd)]["train_windows_identical"] for s in SUBJ for sd in (0, 1, 2))}
    fb, _ = fbcsp_E()
    comps = {}
    for x, y, name in PAIRS:
        if x in arms and y in arms:
            comps[f"{x}_vs_{y}"] = {"name": name, **verdict([arms[x]["per_subject"][s]["E_acc"] for s in SUBJ], [arms[y]["per_subject"][s]["E_acc"] for s in SUBJ])}
    for x in arms:
        comps[f"{x}_vs_fbcsp"] = {"name": "对 FBCSP（参照）", **verdict([arms[x]["per_subject"][s]["E_acc"] for s in SUBJ], [fb[s] for s in SUBJ])}
    return {"arms": arms, "fbcsp_E_acc": fb, "comparisons": comps}


def print_gate(g):
    arms, fb = g["arms"], g["fbcsp_E_acc"]; order = [a for a in ARMS if a in arms]
    print("| 臂 | η | U | T 验证 acc | E acc（种子 sd） | E κ | 合格率 | 拒判 |\n| --- | --- | --- | --- | --- | --- | --- | --- |")
    for a in order:
        r = arms[a]; print(f"| {a} | {r['eta']:g} | {r['U']} | {r['val_acc']:.3f} | {r['E_acc']:.3f} ({r['E_acc_seed_sd']:.3f}) | {r['E_kappa']:.3f} | {r['E_settle']:.3f} | {r['E_rejected']} |")
    print(f"| FBCSP+LDA | | | 0.612 | {np.mean(list(fb.values())):.3f} | 0.490 | 1.000 | 0 |\n")
    print("| 被试 | " + " | ".join(order) + " | FBCSP |\n| --- |" + " --- |" * (len(order) + 1))
    for s in SUBJ:
        print(f"| A{s:02d} | " + " | ".join(f"{arms[a]['per_subject'][s]['E_acc']:.3f}" for a in order) + f" | {fb[s]:.3f} |")
    print("\n| 比较 | 平均差 | 赢/平/输 | p（左>右） | p（左<右） |\n| --- | --- | --- | --- | --- |")
    for k, c in g["comparisons"].items():
        print(f"| {k}（{c['name']}） | {c['mean_diff']:+.3f} | {c['wins']}/{c['ties']}/{c['losses']} | {c['p_greater']:.3g} | {c['p_less']:.3g} |")
    print("\n| 臂 | 训练 acc（u） | T 验证 acc | E acc | W 相对变化 | cos(W, car) | cos(W, I) | 行和均值 |\n| --- | --- | --- | --- | --- | --- | --- | --- |")
    for a in order:
        P = arms[a]["per_subject"]; ta = [P[s]["train_acc"] for s in SUBJ]
        tas = f"{np.mean(ta):.3f} ({P[1]['train_acc_u']})" if all(t is not None for t in ta) else "—"
        f = lambda k: f"{np.mean([P[s][k] for s in SUBJ]):.3f}" if k in P[1] else "—"
        print(f"| {a} | {tas} | {arms[a]['val_acc']:.3f} | {arms[a]['E_acc']:.3f} | {f('front_rel_change')} | {f('front_cos_car')} | {f('front_cos_identity')} | {f('front_row_sum_mean')} |")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--select", action="store_true"); ap.add_argument("--gate", action="store_true"); a = ap.parse_args()
    if a.gate:
        g = gate_summary(); (R / "gate_summary.json").write_text(json.dumps(g, indent=1)); print_gate(g); return
    recs = load("grid")
    sel, tab = select(recs)
    print(f"{len(recs)} grid receipts")
    for (arm, eta) in sorted(tab):
        byU = tab[(arm, eta)]
        line = " ".join(f"{U}:{np.mean([r['acc'] for r in rows.values()]):.3f}({len(rows)})" for U, rows in sorted(byU.items()))
        print(f"{arm:9s} eta {eta:g}  {line}")
    print(json.dumps(sel, indent=1))
    if a.select:
        (R / "selected.json").write_text(json.dumps(sel, indent=1)); print("wrote", R / "selected.json")


if __name__ == "__main__":
    main()
