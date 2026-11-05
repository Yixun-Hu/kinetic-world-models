# Lock-then-refine adaptive velocity nodes

Lock-then-refine is an opt-in velocity update for the adaptive model. It uses
Lucas–Kanade (LK) where the current node velocity has coherent, small residuals,
and a local displacement search elsewhere. Both routes pool evidence into the
heavier of up to two motion modes. Feature recurrence and model parameters are
unchanged.

## Method

### State and measurement

Node velocity is `nu = lattice + rho`. Lock uses an explicit `LockState`,
initialized to zero:

| Field | Shape | Contents |
| --- | --- | --- |
| `features` | `[B,V,C,H,W]` | Feature memory in the feature/autocast dtype |
| `nodes` | `[B,V,4]` | fp32 `rho_x, rho_y, accumulated_evidence, lock_bit` |
| `snapshot` | `[B,K,win,win]` | One fp32 observation imprint per sample, shared across nodes |

At the default `V=25`, `K=8`, `win=32`, metadata and snapshot require
400 + 32,768 = **33,168 bytes/sample**, excluding features and backward
intermediates. LK keeps its existing `[B,V,C+3+K,H,W]` tensor state.

The current imprint is `P(conv_u(frame))`, using the existing fixed projection
to eight channels. Zero-pad the previous snapshot to the periodic world, then
spectrally transport it by `nu * dt - action` and crop to the observed window.
Call this transported imprint `m` and the current imprint `c`. The action is
the measured displacement for this step. After an update, replace the shared
snapshot with `c`; there is no decaying per-node imprint bank on this route.

LK uses central differences of `m` and a 5×5 Gaussian window with sigma 1 to
solve the damped local normal equations against `m - c`. With `trace = tr(G)/2`,
damping is `0.01 * (trace + mean_pixels(trace)) + 0.001`, where the mean is
computed separately for each node. Clamp each displacement component to ±1 pixel.
For either displacement field `d`, the measurement weight multiplies:

- imprint magnitude `norm_channels(m)`;
- texture gate `trace > 0.1 * mean_pixels(trace)`;
- velocity-size gate `max(abs(d)) / dt <= 0.6`;
- observed overlap support, computed by bilinear sampling of the window mask
  on the world torus;
- unwarped agreement `exp(-sum_channels((m-c)^2) / sigma^2)`, with
  `sigma = 0.5 * mean_pixels(norm_channels(c)) + 1e-6`.

### Lock bit and search

First pool LK displacement as `sum(weight * d) / (sum(weight) + 1e-6)`.
A node locks if its LK evidence divided by its textured, supported imprint mass
(denominator clamped to `1e-6`) is at least **0.1**, and its pooled residual
velocity norm is at most **0.3 px/frame**. The bit is evaluated afresh each step;
the stored bit is diagnostic and introduces no hysteresis.

Search compares `m(x)` with `c(x+d)` on the **25** shifts in
`[-1, -0.5, 0, 0.5, 1]^2`. It aggregates SSD, `|m|² + |c|² - 2 m·c`
(clamped nonnegative), with a separable Gaussian of sigma **2**, radius **6**,
and replicate padding. Divide by aggregated valid overlap, clamped to `1e-6`;
overlap mass below **0.5** receives sentinel cost `1e6`. Refine an interior
minimum with a three-point parabola on each axis when curvature exceeds `1e-8`,
clipping the refinement to half a grid interval. Reject boundary minima, cost
at or above `1e5`, and transported snapshot support at or below **0.99**.
Search is evaluated for every node; tensor selection uses LK at locked nodes
and search at unlocked nodes. Search validity gates only the search route.

### Two-mode pooling and evidence-weighted gain

Recompute measurement weights on the selected field. Discard candidate velocities
outside the node's reachable lattice box, `abs(nu + d/dt - lattice) <= 0.75`
componentwise. Compute the weighted covariance's principal axis and initialize
two projected centers at minus/plus one weighted standard deviation. Run **five**
weighted two-center iterations; distance ties go left. Accept a split only when
total mass exceeds `1e-6`, each cluster has at least **10%** of the mass, and
their mean velocities differ by at least **0.1 px/frame**. Use the heavier
cluster (left wins mass ties); otherwise use the unsplit mean. A secondary mode
does not create another node. Weighted means use denominators clamped to `1e-6`.

Let `delta` be the selected mean displacement (velocity multiplied by `dt`),
`evidence` its mass, and `W` the previous accumulated evidence. The cell computes:

```text
W_new    = W + evidence
gain     = evidence / max(W_new, 1e-6)
residual = (nu + delta / dt) - nu
step     = clip(n_s * gain * residual, -0.3, 0.3)
rho_new  = clip(nu - lattice + step, -0.75, 0.75)
```

These fp32 expressions preserve the rounding of the trained implementation.
The existing learned scalar `n_s`, initialized to **1**, scales both routes;
it is unconstrained and adds no parameters. Features still stream with the
previous node velocities and use the existing convolution/ReLU recurrence.

### Gradient routing

Transport, estimator arithmetic, node metadata and snapshots use fp32 with
autocast disabled. Gates, search, clustering and selected pixel weights are
detached. The forward displacement equals the detached mode estimator exactly:
`delta + (differentiable - differentiable.detach())`, where `differentiable` is
the selected weighted LK field divided by the selected evidence, and is zero
at unlocked nodes. Thus only locked nodes carry the LK displacement derivative;
searched displacements carry none. The previous snapshot retains its graph for
LK, and `n_s` receives gradients through the cell update on both routes.

## Usage

```bash
python -m kwm.train --model adaptive --node_update lock --real_velocity --real_camera --grad_ckpt --out runs/adaptive_lock
python -m kwm.eval --run runs/adaptive_lock
```

All defaults remain unchanged: `--node_update lk`, feedback prediction input,
and prediction-phase node updates on. Lock is supported only for adaptive
models. `kwm.eval --node_update {lk,lock}` overrides a checkpoint's saved rule;
omitting it preserves the saved value, with `lk` as the fallback for older
checkpoints. Both rules have identical parameter sets and checkpoint weight keys.
`--imprint_gamma` affects only LK. `--node_update_pred off` freezes lock metadata
and snapshot during prediction while features continue evolving.

## Evaluation protocol

Both rules trained on Kinetic MNIST World with real velocities and camera motion
for **20,000 steps**, batch **16**, with gradient checkpointing and an
**identical optimization budget**: the same seeds **0–4**, initialization, data
stream and schedule within each pair. Training used **30** observation frames,
**20** prediction frames, learning rate **0.001**, and teacher forcing **0.125**.

For each run, score the mean over checkpoints **16k, 17k, 18k, 19k, 20k** on
**1,000** held-out sequences from seed **20261025** (MNIST test digits). This
fixed checkpoint average avoids checkpoint selection. MSE@20 and MSE@150
average viewport error over the first **20** and **150** predicted frames.
The headline protocol uses native feedback input and prediction updates on.
The recorded campaign used a held-out-seed evaluation driver; this branch's
stock evaluation CLI uses its fixed validation seed and cannot select that
held-out seed.

Tests are two-sided paired t-tests on the five seed-level MSE differences
(degrees of freedom **4**). Percentage changes are the **mean of the five
per-seed relative changes**, not the ratio of the displayed mean MSEs.
Each test reports the number of improving seeds: with **n=5**, even **5/5**
improvements give a minimum two-sided exact sign-test p-value of **0.0625**.
Several comparisons were examined; the p-values below are unadjusted.

## Results

**Native feedback input, node updates on — the effect of this change:**

| Horizon | LK mean MSE | Lock mean MSE | Mean paired change | Paired t-test and seeds improved |
| --- | ---: | ---: | ---: | --- |
| 20 frames | 0.0284 | 0.0220 | −22.2% | p = 2e-4; **5/5 seeds** |
| 150 frames | 0.1298 | 0.0918 | −26.1% | p = 0.065; **5/5 seeds**, not significant |

At 150 frames, per-seed changes range from **−0.6% to −50.6%**. Part of this
spread reflects LK feedback instability: seed 3's late-checkpoint MSEs include
spikes to **0.253** and **0.243**.

**Supplementary — zero prediction input:** mean paired changes are **−22.1% at 20 frames (p = 1e-4; 5/5 seeds)** and **−16.2% at 150 frames (p = 0.040; 5/5 seeds)**; this requires the prediction-input evaluation option of a later change (PR3), and under zero input the node update is inactive during prediction for both rules (up to LK roundoff).

Under Holm correction over the four comparisons (two horizons × two inputs),
only the 20-frame results remain significant at **0.05**; the supplementary
150-frame result does not. The combined effect of changing both the node rule
and prediction input is not attributed to this change.

## Cost

On the shared A6000 at batch **16** with gradient checkpointing, lock takes
**1.19–1.29×** LK's time per step across seeds. Median logged time pooled over
seeds **2–4** is **4.47 vs 5.74 s/step** (LK vs lock); device contention makes
these descriptive timings. Peak allocated memory in the training logs is
**6,433–6,434 vs 5,239–5,241 MiB** (about **6.7 vs 5.5 GB**, or **6.3 vs
5.1 GiB**). The previously quoted **6.4 vs 5.2 GB** divides the logged values
by 1,000 even though the logger's `peak_mem_mb` column measures MiB.
Search runs for locked nodes too, so a higher lock rate does not reduce cost.

## Limitations

- Lock nodes drift more on the model's own predictions. In the review's seed-1
  checkpoint probe, mean endpoint node-velocity change over **150** prediction
  steps was **0.18 vs 0.04 px/frame** (lock vs LK). These are changes in velocity,
  not position drift divided by elapsed time. With prediction updates off,
  lock's 150-frame mean paired gain is **−29.9%** versus **−26.1%** with updates
  on (off: **p = 0.054; 5/5 seeds**, still not significant); its mean MSE is
  **0.0867 vs 0.0918**. The default remains on.
- Search radius, support rejection, the **0.6 px/frame** component gate and
  offset clipping limit recoverable motion. The heavier-mode rule can discard
  a smaller object sharing a velocity cell.
- Training uses bf16 autocast and TF32 on CUDA; the recorded evaluation uses
  fp32. Discrete lock, mode and edge decisions can amplify small numerical
  differences. This precision difference applies to both arms.
- The estimator was selected in a separate study, not part of this change,
  using MNIST test digits; the held-out sequences use the same digit pool with
  different sequence seeds. These results cover real velocities and camera
  motion, not integer-velocity data, walls or odometry noise.
- Seed-0 checkpoints carry no provenance record. Seeds **1–4** were trained on
  the later PR3 tree, whose lock computation the review verified bit-for-bit
  against this implementation. Seed **3** of both arms resumed from its last
  checkpoint; pairing was preserved.

**AI assistance:** Code written with gpt-6-astra and Claude, reviewed
adversarially by Claude Opus; all numbers come from the recorded evaluations,
training logs and review probes.
