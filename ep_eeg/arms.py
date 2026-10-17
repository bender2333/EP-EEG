from functools import partial
from . import baselines as B
from .closed_form import closed_form as _cf

ARMS = {
    "lda_track1": partial(B.lda, track="track1"),
    "lda_track2": partial(B.lda, track="track2"),
    "shuffled_labels": partial(B.lda, track="track2", shuffle=True),
    "fbcsp": B.fbcsp,
    "eegnet": B.eegnet,
    "mlp_track2": B.mlp,
    "lda_envtraj": partial(B.lda, track="envtraj"),  # C2 control: shrinkage LDA on the same 242-d input
    # M9c closed-form decomposition controls (no lake)
    "ridge_track2_trial": partial(_cf, features="track2", unit="trial", model="ridge"),  # vs C0 lake ridge 0.366
    "lda_track2_trial": partial(_cf, features="track2", unit="trial", model="lda"),
    "lda_act_rows": partial(_cf, features="act", unit="rows", model="lda"),    # Activity only, 2112 rows
    "lda_act_trial": partial(_cf, features="act", unit="trial", model="lda"),  # Activity only, 192 rows
    "ridge_act_trial": partial(_cf, features="act", unit="trial", model="ridge"),
}


def get(arm):
    if arm in ARMS:
        return ARMS[arm]
    from . import sandbox  # EP arms registered lazily
    return sandbox.arm(arm)
