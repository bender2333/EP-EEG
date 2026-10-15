"""Group-4 baselines: same-input LDA, FBCSP, EEGNet (braindecode)."""
from __future__ import annotations
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def lda(train, val, seed, track="track2", shuffle=False):
    X = train[track + "_x"]; y = train["y"]
    if shuffle:
        y = np.random.default_rng(seed).permutation(y)
    n, k, d = X.shape
    clf = LDA(solver="lsqr", shrinkage="auto").fit(X.reshape(-1, d), np.repeat(y, k))
    Xv = val[track + "_x"]
    p = clf.predict_proba(Xv.reshape(-1, d)).reshape(len(Xv), k, 4)
    return p, np.ones(p.shape[:2], bool), {"params": int(clf.coef_.size + clf.intercept_.size)}


def fbcsp(train, val, seed, n_feat=8):
    from scipy.signal import butter, sosfiltfilt
    from mne.decoding import CSP
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    bands = [(lo, lo + 4) for lo in range(4, 36, 4)]
    def bank(seg, csps=None, y=None):
        feats, fitted = [], []
        for i, (lo, hi) in enumerate(bands):
            sos = butter(4, [lo, min(hi, 124)], btype="band", fs=250, output="sos")
            x = sosfiltfilt(sos, seg.astype(np.float64), axis=-1)
            c = csps[i] if csps else CSP(n_components=4, log=True).fit(x, y)
            fitted.append(c); feats.append(c.transform(x))
        return np.hstack(feats), fitted
    Ft, csps = bank(train["seg"], y=train["y"])
    sel = SelectKBest(lambda X, y: mutual_info_classif(X, y, random_state=seed), k=n_feat).fit(Ft, train["y"])
    clf = LDA().fit(sel.transform(Ft), train["y"])
    Fv, _ = bank(val["seg"], csps)
    p = clf.predict_proba(sel.transform(Fv))[:, None, :]
    return p, np.ones(p.shape[:2], bool), {}


def eegnet(train, val, seed, epochs=300, lr=1e-3, batch=32):
    import torch
    from braindecode.models import EEGNet
    torch.set_num_threads(2); torch.manual_seed(seed); np.random.seed(seed)
    Xt = torch.tensor(train["seg"], dtype=torch.float32); yt = torch.tensor(train["y"])
    m = EEGNet(n_chans=22, n_outputs=4, n_times=Xt.shape[-1])
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        m.train()
        for idx in torch.randperm(len(Xt), generator=g).split(batch):
            opt.zero_grad(); loss = torch.nn.functional.cross_entropy(m(Xt[idx]), yt[idx]); loss.backward(); opt.step()
        sched.step()
    m.eval()
    with torch.no_grad():
        p = torch.softmax(m(torch.tensor(val["seg"], dtype=torch.float32)), 1).numpy()[:, None, :]
    return p, np.ones(p.shape[:2], bool), {"params": int(sum(q.numel() for q in m.parameters()))}
