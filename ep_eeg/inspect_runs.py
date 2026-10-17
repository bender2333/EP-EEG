import json, sys
for f in sys.argv[1:]:
    s = json.load(open(f + "/summary.json")); print(f, s["opts"], round(s["kappa_val"], 4))
    for r in json.load(open(f + "/runs.json")):
        e = r["extra"]; print("  A%d s%d k=%.3f settle=%.3f" % (r["subject"], r["seed"], r["kappa"], r["settle_rate"]),
              e.get("train", {}).get("failed"), (e.get("log") or [""])[-1], "acc_rate=%s" % e.get("learn_accept_rate"))
