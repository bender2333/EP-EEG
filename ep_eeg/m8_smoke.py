"""M8 micro smoke: subject 1, first 640 train steps / 400 val steps, 1 epoch, batch 8 (pipeline check only)."""
import time, numpy as np
from .m8_data import load_subject_m8
from .m8_train import BASE, fit, infer, score
from .sandbox import Sandbox, SandboxConfig
d = load_subject_m8(1)
cut = lambda D, n: {k: v[:n] for k, v in D.items()}
tr, va = cut(d["train"], 640), cut(d["val"], 400)
sb = Sandbox(SandboxConfig(**{**BASE, "epochs": 1, "batch": 8}), 0); t = time.time()
st = fit(sb, tr, print); act, ok = infer(sb, va)
print("n_out", sb.n_out, "dims", sb.d, "train_s", round(time.time() - t, 1), score(va, act, ok))
