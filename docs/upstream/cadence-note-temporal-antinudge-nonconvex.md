# `TemporalPatchNet.observe`: the −β phase is not well-posed when the readout is large (b·σ_max(C)² > 1), so every update is refused

**cadence version**: 0.84.3 (muellerberndt/cadence `8dc92d8`), `cadence.experimental.PartitionedTemporalPatchNet` (same `observe` as `TemporalPatchNet`).
**Status**: draft (EP-EEG M11 amendment 4). Not filed yet.

## Observation

EP-EEG M11 step C uses a 22-output temporal net (H = 44, 125-step crops, batch 16) whose readout C is initialised by ridge regression on the reservoir states. With β = 0.05, `observe` refuses every update with reason `phase_failed` at one of two input scales:

| Input scale | Attempts accepted | Time per attempt | What happens |
|---|---|---|---|
| gain 1 | 0 / 20 | 70–80 s | +β phase converges in 3 Newton iterations; −β phase reaches `max_iterations` = 80 with residual 13–21 |
| gain 3 | 20 / 20 | 8–13 s | both phases converge in 2 iterations |

## Mechanism (as far as we can tell)

The nudge weight per output is b = β·precision / (T·outputs) = 0.05 / (125·22) ≈ 1.8e-5. In the −β phase the reduced energy gains a term −b·‖C·tanh(h) − y‖², whose curvature in h can reach −b·σ_max(C)².

- At gain 1 the ridge readout is large: σ_max(C) = 829 on one subject, so b·σ_max(C)² = 12.5.
- Over 27 (subject, seed) configurations, b·σ_max(C)² was:
  - 5.5–12.7 for seed 0, 1.5–3.3 for seed 1 and 0.32–0.58 for seed 2 at gain 1;
  - 0.02–0.55 for all 27 at gain 3.
- When the negative term outweighs the state curvature, the energy is no longer convex. The block-Cholesky Newton step then finds no SPD descent direction, and the solve stalls at the iteration cap.

The refusal is the safe outcome, since no wrong update is applied. But:

1. **Cost.** Each refused attempt costs ~6–9× an accepted one, because the solver always runs to the cap.
2. **Silence.** Nothing tells the user that the configuration can never update. The free and +β phases look healthy, and `phase_failed` does not say which phase failed or why.

## Suggestions

- Report which detuned phase failed (plus or minus) and the solver reason (`iteration_cap`, `no_decreasing_spd_step`, …) in the `TemporalObservation`. The phases are already attached, but the top-level reason hides this.
- Optionally add a cheap pre-check, or a documented precondition, on b·σ_max(C)²; for example, warn when it exceeds 1.
- Optionally fall back to a one-sided (+β only) contrast when the −β phase is ill-posed, flagged in the observation, at the cost of the O(β) bias the centred contrast removes.

## Downstream evidence

EP-EEG `reports/M11.md` (amendment 4), diagnostic `ep_eeg/m11_c_accept_diag.py`, receipt `receipts/M11/C/accept_diag.json`.
