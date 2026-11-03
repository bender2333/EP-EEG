"""M11 amendment 5: derive the gain-1 ep_proj receipt where ep_noproj accepted no update.

The spectral-radius projection is applied only after an accepted update. With zero accepted updates the two arms
see the same crops (same rng), refuse the same attempts and evaluate the same network (the C_res initialisation),
so the ep_proj result is the ep_noproj result by construction. Writes receipts/M11/C/runs/ep_proj/<run>.json for
every ep_noproj receipt with train.updated == 0, marked "derived_from": "ep_noproj"; configurations with any
accepted update are left for the queue to run in full.
Amendment M5-2 (gain 3): the projection acts only when rho(A) > 0.95 after an accepted update. If the ep_noproj_g3
run never exceeded 0.95 (train.rho_max <= 0.95), ep_proj_g3 follows the identical trajectory and is derived the same
way; runs that exceeded the cap are left to run in full.
"""
from __future__ import annotations
import json
from .m11_c import OUT, OPT
from .m11_common import env_info


def main():
    made = []
    for sfx in ("", "_g3"):
        src, dst = OUT / "runs" / f"ep_noproj{sfx}", OUT / "runs" / f"ep_proj{sfx}"; dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.glob("A*_s*.json")):
            r = json.loads(f.read_text())
            if (dst / f.name).exists(): continue
            if r["train"]["updated"] == 0:
                why = ("amendment M5: zero accepted updates in ep_noproj; the projection acts only after an accepted update, so "
                       "ep_proj replays the same refused attempts and evaluates the same network")
            elif sfx and r["train"]["rho_max"] <= OPT["proj"]:
                why = ("amendment M5-2: rho(A) never exceeded the 0.95 cap along the ep_noproj trajectory, so the projection never "
                       "acts and ep_proj follows the identical trajectory")
            else:
                continue
            r.update(arm=f"ep_proj{sfx}", derived_from=f"ep_noproj{sfx}", derivation=why, derivation_env=env_info())
            tmp = dst / (f.stem + ".tmp"); tmp.write_text(json.dumps(r)); tmp.replace(dst / f.name); made.append(f"{sfx or 'g1'}:{f.stem}")
    print("derived", len(made), made)


if __name__ == "__main__":
    main()
