# Lock-then-refine adaptive velocity nodes

Use `python -m kwm.train --model adaptive --node_update lock` to enable
lock-then-refine updates; `--node_update lk` remains the default.
`kwm.eval --node_update {lk,lock}` overrides the checkpoint's saved rule;
omitting it preserves the saved value, with `lk` as the old-checkpoint fallback.
`--imprint_gamma` controls only the LK imprint bank and has no effect under lock.
The parameter set and checkpoint weight keys are identical between the rules.

## Method

### Measurement and update rule

Each node has velocity `nu_i = lattice_i + rho_i`. Encode the observation as
`P(conv_u(frame))` with the existing fixed eight-channel projection. Pad the
previous observation snapshot to the periodic world and transport it by
`nu_i * dt - action`. The action is this step's measured displacement; no
cumulative camera state is needed. Feature transport and convolution retain
the existing recurrence.

Estimate a per-pixel Lucas–Kanade (LK) residual. Both its damping reference and
texture threshold average over pixels **separately for each node**. Damping is
`lam_rel * (trace + mean_pixels(trace)) + eps`, with `trace = tr(G)/2`,
`lam_rel=0.01`, and `eps=0.001`. Measurement weights multiply imprint magnitude,
`trace > 0.1 * mean_pixels(trace)`, the componentwise residual-size test
`abs(d)/dt <= 0.6`, observed overlap, and unwarped agreement
`exp(-sum_channels((m-current)^2)/sigma^2)`, where
`sigma = 0.5 * mean_pixels(norm_channels(current)) + 1e-6`.

A node locks when weighted LK evidence is at least 10% of its textured imprint
mass and its pooled residual velocity norm is at most 0.3 px/frame. The lock is
reevaluated every step. The stored bit is diagnostic, without hysteresis.

Dense search evaluates every node using the 25 shifts in
`[-1, -0.5, 0, 0.5, 1]^2`. Candidate snapshots are shared across nodes. SSD is
`|m|² + |c|² - 2 m·c`, aggregated by a separable Gaussian (sigma 2, radius 6)
and divided by aggregated valid overlap. Reject overlap mass below 0.5.
Refine interior minima with a three-point parabola along each axis; reject
minima on either grid boundary and snapshot support at or below 0.99.
Tensor selection chooses LK for locked nodes and search for the others.

Apply the measurement weights to the selected field and restrict candidate
velocities to the node's reachable lattice box, `|rho| <= 0.75`. Fit two modes
along the weighted covariance's principal axis for five iterations. Accept a
split only if each mode has at least 10% of the mass and separation is at least
0.1 px/frame. Use the heavier mode (left wins exact ties), or the unsplit mean.
A second mode does not create an additional node.

Accumulate selected evidence `W`, update each offset with
`clip(n_s * evidence / max(W, 1e-6) * residual_velocity, -0.3, 0.3)`, and clamp
the result to ±0.75. The existing learned scalar `n_s` is shared by both routes.
No estimator parameters are added.

### State layout and prediction policy

LK retains its `[B,V,C+3+K,H,W]` tensor. Lock uses an explicit `LockState`:

| Field | Shape | Meaning and dtype |
| --- | --- | --- |
| `features` | `[B,V,C,H,W]` | Feature memory; feature/autocast dtype |
| `nodes` | `[B,V,4]` | `rho_x, rho_y, accumulated_evidence, lock_bit`; fp32 |
| `snapshot` | `[B,K,win,win]` | One observation imprint per sample; fp32 |

At V=25, K=8, win=32, estimator state occupies **33,168 bytes/sample**:
400 bytes of metadata and 32,768 bytes of snapshot, or 530,688 bytes at batch
16, excluding features and backward intermediates. `model.initial_state(frame)`
allocates either representation. `state_extra` counts only extra spatial
channels (11 for LK at K=8, zero for lock); `cell.extra=3` describes legacy LK
metadata. Diagnostics read returned state. Bounce and comoving boost helpers
transform node metadata/features and preserve the shared observation snapshot.

`--node_update_pred off` freezes lock metadata and snapshot during prediction;
features continue evolving. Otherwise updates consume model predictions or
teacher-forced inputs according to the existing input policy. The lock
checkpoint closure captures the update flag for backward recomputation.

### Numerics and gradients

Transport, measurement, node state and snapshots use fp32 with autocast disabled.
Search, weights, clustering and mode selection are detached. A zero-valued
gradient correction reattaches the selected, locked LK displacement derivative
without changing forward pooling. Searched displacements have zero gradient;
`n_s` learns through the actual cell update on both routes. The previous snapshot
retains its graph for LK. Fixed-shape dense routing avoids tensor-to-host scalar
conversions in forward and backward.

Training enables TF32. A review diagnostic allowing TF32 in the SSD contraction
measured **0.0239 px/frame / 19.8% misses** on seed 777, a negligible change in
that diagnostic. CPU accuracy evidence below uses fp32 with TF32 disabled;
this does not guarantee identical mode decisions on every device.

### Cost and equivariance

Snapshot transport computes one forward FFT per sample and applies node phases
before the inverse FFTs. Candidate transport also shares its forward FFT.
Candidate masks are cached by world/window size, radius and device in a bounded
cache; observations and recurrent state are never cached. Candidate chunks of
seven avoid a node × candidate × channel image-difference tensor. Dense search
still runs for locked nodes, so cost is not reduced by lock rate. Regression
checks compare node trajectories against repeated-FFT transport to an absolute
tolerance of 1e-6.

On four seed-777 sequences, all 30 node updates matched the pre-optimization
implementation exactly (maximum component difference 0).

Per-node texture and damping remove coupling to velocity-lattice edges under
a comoving integer boost. Tests bound normalized feature error by **1e-5** on
MNIST sequences for model seeds 0–2, and by **1e-6** for both LK and lock on
synthetic on-lattice translations. These are interior-node memory comparisons;
finite-lattice readouts, arbitrary off-lattice boosts and scene-only boosts
under partial observation are not exact symmetries.

The broader eight-sequence check, model seeds 0–2 and observation steps 3, 6,
and 10, measured comoving on-lattice errors from 5.45e-8 to 2.80e-7.

## Estimator evidence

No-learning evaluation uses a frozen seed-0 encoder (C=32, K=8), 30 observations,
five digits per sequence and exact camera actions. Seed 2024 uses the first 100
of a single 300-sequence draw; the other seeds each draw 100. Matched errors use
one-to-one Hungarian assignment, followed by last-ten-frame visibility filtering.
Misses are errors above 0.1 px/frame. The final rule uses per-node texture and
damping, direct actions, and heavier-mode selection on the ±1 grid.

| Seed | LK matched median | LK misses | Final lock matched median | Final lock misses |
| --- | ---: | ---: | ---: | ---: |
| 2024 | 0.078062 | 40.6% | 0.022695 | 18.4% |
| 777 | 0.091020 | 44.8% | 0.024010 | 19.8% |
| 31337 | 0.079873 | 41.0% | 0.023973 | 16.8% |
| Pooled | 0.081799 | 42.1% | 0.023746 | 18.3% |

See [estimator evaluation](GATE_estimator.md) for reproduction, raw results and
reduced regression-test baselines. Estimator accuracy alone does not establish
improved forecasts after training.

## GPU timing

Prior measurements on a **shared RTX A6000**, batch 16, accumulation 2, 60
training steps:

| Update rule | Seconds/step | Peak allocated memory |
| --- | ---: | ---: |
| LK | 4.52 | 15.2 GB |
| Lock | 5.34 | 13.2 GB |

These supplied integration measurements were not rerun during CPU validation.
Shared-device contention limits timing precision; 60 steps measure cost rather
than converged forecasting quality.

## Validation and limitations

The default suite skips the two dataset regressions; `KWM_SLOW=1 pytest -q`
includes them and `pytest -m slow` selects only them. Use one OpenBLAS/OMP thread
for reproducible CPU timing. The LK reference is frozen from **main@302d8f8**;
its forward values and gradients must remain bit-identical. The default
`lk_displacement` damping and default LK update arithmetic are unchanged.

CPU validation with PyTorch 2.9.0+cu128 and one OpenBLAS/OMP thread:

| Suite | Result | Runtime |
| --- | --- | ---: |
| Default | 95 passed, 2 slow cases skipped | 53.02 s |
| `KWM_SLOW=1` | 97 passed | 128.75 s |

Both suites report one dependency deprecation warning. Independent comparison
against main@302d8f8 confirms bit-identical LK outputs and gradients in five
cell configurations and twelve model configurations, including equivariance
helpers. The two reduced accuracy cases exactly match their final-rule references.

- Search radius, support rejection and offset clipping limit recoverable motion.
- The heavier-mode rule can discard a smaller object sharing a velocity cell.
- Discrete gates and mode choices can amplify small floating-point differences.
- Prediction-driven updates can incorporate errors from generated frames;
  updating and freezing nodes both need trained-model evaluation.
- The pre-existing LK combination `--grad_ckpt --node_update_pred off` has a
  backward recomputation flag bug. It is outside this change; lock retains its
  explicit checkpoint flag handling.

## End-to-end training comparison

**Pending.** Compare LK and the final lock rule with matched initialization,
training data, optimization budgets and evaluation sets. Include prediction
updates both on and off. Report 20/150-frame forecast MSE, node error,
equivariance, training time and peak memory. The estimator table and 60-step
cost measurement do not fill this comparison.
