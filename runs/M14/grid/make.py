import json
jobs=[]
for arm in ("A","C","B","nolake","C_nolake","randlake"):
    for eta in (1.0, 0.3):
        for s in range(1,10):
            jobs.append(f"{arm} {s} {eta}")
open("runs/M14/grid/jobs.txt","w").write("\n".join(jobs)+"\n"); print(len(jobs))
