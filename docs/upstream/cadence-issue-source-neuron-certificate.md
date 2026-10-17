# certificate(): row mass counts edges from source neurons (no incoming edges), which locks driven networks in the linear regime

**cadence version**: 0.83.0 (commit `6499e56`)

## Observation

`cadence.certificate.certificate(graph)` sets `row_mass` to the maximum, over neurons, of the summed absolute effective weights on each neuron's incoming edges. It counts **every** incoming edge, including edges from neurons that have no incoming edges themselves (one-way input or source neurons). Downstream (EP-EEG, BCI IV-2a), the network has 66 input neurons driven by standardized features, feeding 88 recurrent "column" neurons and 4 motor neurons. Under this certificate:

- Input synapses take up to about 0.56 of the 1/L = 2 row-mass budget at initialization. Scaling them up breaks the certificate (ρ_full = 2.96 / 5.22 / 9.72 at σ_in × 4 / 8 / 16).
- As a result, the column potentials stay within about ±0.25, and a linear fit from the inputs explains the column activations with R² = 0.9998. The recurrent "lake" is a linear projection in every certified configuration we ran.

## Proposed refinement (sound, not a relaxation)

If the source set S has `ep_structure(graph, fixed_inputs=S).fixed_incoming_mass == 0`, the Jacobian is block-lower-triangular in the ordering (S, free):

1. The sources evolve as v_S ← (1−dt)·v_S + dt·c_S and converge on their own at rate 1−dt.
2. With the sources fixed, the free block evolves as v_f ← (1−dt)·v_f + dt·(W_ff·ρ(v_f) + W_Sf·ρ(v_S) + c_f). The term W_Sf·ρ(v_S) is an external field, so it adds nothing to the Lipschitz constant. The free block contracts at q_f = 1 − dt·(1 − L·ρ_free), where ρ_free counts free→free edges only.
3. By the cascade, the whole system has a unique fixed point and converges at rate max(q_f, 1 − dt).

The existing Lean proof covers the whole-graph ∞-norm version only; this cascade extension is a standard argument but lies outside it. A possible API is `certificate(graph, sources=S)`. It would assert `fixed_incoming_mass == 0` and report `row_mass_free` alongside the current full-graph numbers.

## Interaction with `LearnerConfig.scale_cap`

Once input synapses are scaled past `scale_cap` (default 8, here 3), the first `Learner.step` silently clips every plastic synapse back to ±cap. That undoes the scaling for any input synapse that is plastic. Documenting this, or giving sources their own cap, would help.

## Downstream evidence

EP-EEG `reports/M9.md` (M9c sections) and `docs/adr/0012-free-block-certificate.md`. Under the refined certificate, every operating point σ_in × {1 … 128} certifies with ρ_free = 0.889 at init, and free-phase convergence is 1.0.
