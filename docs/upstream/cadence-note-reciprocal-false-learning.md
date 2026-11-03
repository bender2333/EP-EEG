# `Learner(reciprocal=False)` gives both directions of a pair the same increment; directed asymmetry does not survive training

**cadence version**: 0.84.3 (muellerberndt/cadence `8dc92d8`); the same behaviour under 0.83.0 (`6499e56`).
**Status**: draft (EP-EEG M11). Not filed yet.

## Observation

With `Learner(..., reciprocal=False)`, forward (i→j) and feedback (j→i) efficacies are stored separately. Yet the centred ±β contrast of a synapse is the product of its two endpoint activities, so every EP step adds the **same** increment to i→j and j→i.

- **Per step**: in EP-EEG M11 step E2, over 6600 steps the largest difference between the two increments of any pair was 3e-17.
- **Small recurrent graphs** (M11 D1, `reciprocal=False`, feedback multiplied by 1 + ε·ξ):
  - cos(EP update, gradient w.r.t. the directed weights) ≈ 0.71 for every ε ∈ {0, 0.1, 0.3, 1};
  - cos(EP update, gradient w.r.t. the pair-tied weight) = 1.000 / 1.000 / 0.999 / 0.995.

  So the update estimates the gradient of the pair-tied weight.
- **Consequence for training**: an asymmetry set at initialisation survives only as an additive constant. Any multiplicative constraint then erodes it geometrically, because the EP updates keep refilling the symmetric part.
  - Example: a row-mass projection that rescales over-budget rows.
  - In M11 E2 (A01, seed 0), ‖w_fb − w_fw‖/‖w_fw‖ fell from 1.04 at initialisation to 1.6e-4 after 100 epochs (5188 projections).
- **Earlier evidence**: in EP-EEG G1, an "untied" readout and the reciprocal one gave bit-identical results.

## Why it matters

`reciprocal=False` reads like "directed, independently learned weights". In practice it is "pair-tied learning with a frozen initial offset". Users who want to study asymmetric credit assignment, or who use it as a negative control, get a control that is not wrong.

## Suggestion

Document this in `docs/learning.md` next to `reciprocal`. For example: the EP contrast is symmetric in its endpoints, so with `reciprocal=False` both directions receive identical increments; the method estimates the gradient of the symmetric part of W. If directed learning is intended, it needs a different estimator.

## Downstream evidence

EP-EEG `reports/M11.md`, sections D1 (asymmetry sweep) and E2 (diagnostic `ep_eeg/m11_e2_diag.py`, receipt `receipts/M11/E/068c27c/e2_diag.json`).
