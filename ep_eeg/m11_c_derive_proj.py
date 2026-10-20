"""M11 amendment 5: derive the gain-1 ep_proj receipt where ep_noproj accepted no update.

The spectral-radius projection is applied only after an accepted update. With zero accepted updates the two arms
see the same crops (same rng), refuse the same attempts and evaluate the same network (the C_res initialisation),
so the ep_proj result is the ep_noproj result by construction. Writes receipts/M11/C/runs/ep_proj/<run>.json for
every ep_noproj receipt with train.updated == 0, marked "derived_from": "ep_noproj"; configurations with any
accepted update are left for the queue to run in full.
"""
from __future__ import annotations
import json
from .m11_c import OUT
from .m11_common import env_info


def main():
    src, dst = OUT / "runs" / "ep_noproj", OUT / "runs" / "ep_proj"; dst.mkdir(parents=True, exist_ok=True); made = []
    for f in sorted(src.glob("A*_s*.json")):
        r = json.loads(f.read_text())
        if r["train"]["updated"] != 0 or (dst / f.name).exists(): continue
        r.update(arm="ep_proj", derived_from="ep_noproj", derivation="amendment M5: zero accepted updates in ep_noproj; the projection "
                 "acts only after an accepted update, so ep_proj replays the same refused attempts and evaluates the same network",
                 derivation_env=env_info())
        tmp = dst / (f.stem + ".tmp"); tmp.write_text(json.dumps(r)); tmp.replace(dst / f.name); made.append(f.stem)
    print("derived", len(made), made)


if __name__ == "__main__":
    main()
