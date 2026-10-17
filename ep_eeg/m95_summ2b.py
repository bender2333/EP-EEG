"""Summarise exp2b (Phase 2 fixes) vs frozen-ridge baseline at the same width (exp1b receipts, k=32 seed0, g0=0.05)."""
import json, numpy as np
from scipy.stats import wilcoxon
from .m95_scale import OUT
D = OUT / "exp2b" / "k32_seed0_g0.05"; subs = list(range(1, 10))
base = {s: json.loads((OUT / "exp1b" / "g0.05" / f"seed0_k32_s{s}.json").read_text())["kappa_val"] for s in subs}
def W(a, b):
    a, b = np.array(a), np.array(b); return (float(wilcoxon(a, b).pvalue), float(wilcoxon(a, b, alternative="greater").pvalue)) if np.any(a - b) else (1.0, 1.0)
out = {"frozen_ridge_exp1b": base}
p = json.loads((D / "pooled" / "summary.json").read_text()); pc = json.loads((D / "pooled" / "curve.json").read_text())
ep0 = {int(s): v for s, v in p["per_subject_ep0"].items()}; fin = {int(s): v for s, v in p["per_subject_final"].items()}
out["pooled"] = {"ep0": ep0, "final": fin, "pred_final": p["pred_counts_final"], "mean_ep0": np.mean(list(ep0.values())), "mean_final": np.mean(list(fin.values())),
    "wilcoxon_final_vs_exp1b": W([fin[s] for s in subs], [base[s] for s in subs]), "wilcoxon_final_vs_ep0": W([fin[s] for s in subs], [ep0[s] for s in subs]),
    "n_single_class_final": sum(sum(c > 0 for c in v) == 1 for v in p["pred_counts_final"].values()),
    "trace": [(c["epoch"], c["kappa_mean_train_subjects"], c["loss"]) for c in pc], "wall_s": p["wall_s"],
    "best_epoch": max(pc, key=lambda c: c["kappa_mean_train_subjects"])["epoch"], "best_kappa": max(c["kappa_mean_train_subjects"] for c in pc)}
lo = {}
for s in subs:
    z = json.loads((D / f"loso_s{s}" / "summary.json").read_text()); cv = json.loads((D / f"loso_s{s}" / "curve.json").read_text())
    lo[s] = {"heldout_ep0": z["heldout_ep0"]["kappa_val"], "heldout_final": z["heldout_final"]["kappa_val"], "pred_final": z["heldout_final"]["pred_counts"],
             "lambda_final": z["heldout_final"]["lambda"], "in_fold_mean_final": float(np.mean(list(z["per_subject_final"].values()))),
             "trace_heldout": [c["kappa_heldout"] for c in cv], "trace_infold": [c["kappa_mean_train_subjects"] for c in cv], "wall_s": z["wall_s"]}
hf = [lo[s]["heldout_final"] for s in subs]; h0 = [lo[s]["heldout_ep0"] for s in subs]
out["loso"] = {"per": lo, "mean_heldout_final": float(np.mean(hf)), "mean_heldout_ep0": float(np.mean(h0)),
    "wilcoxon_final_vs_exp1b": W(hf, [base[s] for s in subs]), "wilcoxon_final_vs_ep0": W(hf, h0),
    "n_single_class_final": sum(sum(c > 0 for c in lo[s]["pred_final"]) == 1 for s in subs)}
(D / "summary_all.json").write_text(json.dumps(out, indent=1, default=float)); print(json.dumps({k: v for k, v in out.items()}, default=float)[:6000])
