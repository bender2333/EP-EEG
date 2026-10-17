from functools import partial
from . import baselines as B

ARMS = {
    "lda_track1": partial(B.lda, track="track1"),
    "lda_track2": partial(B.lda, track="track2"),
    "lda_envtraj": partial(B.lda, track="envtraj"),  # C2 control: shrinkage LDA on the same 242-d input
    "shuffled_labels": partial(B.lda, track="track2", shuffle=True),
    "fbcsp": B.fbcsp,
    "eegnet": B.eegnet,
    "mlp_track2": B.mlp,
}


def get(arm):
    if arm in ARMS:
        return ARMS[arm]
    from . import sandbox  # EP arms registered lazily
    return sandbox.arm(arm)
