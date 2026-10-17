"""M9c closed-form controls (no lake). All return per-sub-window scores for the frozen trial aggregation.

feature set : "track2" (66 = Activity|Mobility|Complexity x 22) or "act" (22 = Activity only)
row unit    : "rows"  = one training row per sub-window (192 x 11 = 2112 rows), scored per sub-window
              "trial" = one training row per trial (sub-window mean, 192 rows)
model       : shrinkage LDA (lsqr, auto) or standardised RidgeClassifierCV
"""
from __future__ import annotations
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import RidgeClassifierCV


def _feat(d, features):
    X = d["track2_x"]
    return X[..., :22] if features == "act" else X


def closed_form(train, val, seed, features="track2", unit="rows", model="lda"):
    Xtr, Xva, y = _feat(train, features), _feat(val, features), train["y"]
    n, k, f = Xtr.shape
    if unit == "rows":
        F, Y = Xtr.reshape(-1, f), np.repeat(y, k)
    else:
        F, Y = Xtr.mean(1), y
    if model == "lda":
        clf = LDA(solver="lsqr", shrinkage="auto").fit(F, Y)
        score = lambda Z: clf.predict_proba(Z)
    else:
        mu, sd = F.mean(0), F.std(0) + 1e-9
        clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 13)).fit((F - mu) / sd, Y)
        score = lambda Z: clf.decision_function((Z - mu) / sd)
    if unit == "rows":
        s = score(Xva.reshape(-1, f)).reshape(len(Xva), k, -1)
    else:  # linear decision on the trial mean, broadcast over sub-windows (aggregation mean is identical)
        s = np.repeat(score(Xva.mean(1))[:, None], k, 1)
    return s, np.ones(s.shape[:2], bool), {"features": features, "unit": unit, "model": model, "rows": int(len(F))}
