# Second-order lock nodes and K-accel

`--order 2` adds a detached acceleration estimate to the snapshot lock estimator.
It requires `--model adaptive --node_update lock`. The default remains order 1;
LK is unchanged. No forecasting training is included in this PR. The specified
2% null calibration target was **not met** by any of the requested inflation
values; the default of 4 is a conservative fallback, not a successful calibration.

## State and estimator

Both lock orders use `LockState(features, nodes, snapshot)` with 17 fp32 node
columns. LK retains its original spatial state. The first four lock columns
and their arithmetic retain the PR1 meaning. The additional columns remain
zero for order 1.

| Columns | Meaning |
| --- | --- |
| 0–1 | PR1 residual velocity rho |
| 2 | PR1 cumulative evidence W |
| 3 | Current lock bit |
| 4 | Fit weight S_w |
| 5 | Weighted mean measurement time t_bar |
| 6 | Centred time moment C_tt = sum w(t−t_bar)² |
| 7–8 | Weighted mean measured velocity z_bar |
| 9–10 | Centred cross moment C_tz = sum w(t−t_bar)(z−z_bar) |
| 11 | Sum of squared weights |
| 12 | Centred velocity scatter C_zz = sum w||z−z_bar||² |
| 13 | Selected split mode: 0 unsplit, 1 left, 2 right |
| 14 | Next observation time, in time units |
| 15–16 | Additive instantaneous velocity correction |

The nine fit columns are a centred, weighted Welford representation of the
seven proposed raw sums plus squared weights and total velocity scatter. This
avoids cancellation in fp32 and needs no stored history. A single scatter
scalar suffices because the gate uses the trace of the 2D slope covariance.
Mode, clock and two correction columns make recurrence replay explicit; frozen
slopes and gates are recomputed from frozen sums instead of stored again.

The PR1 measurement uses its original velocity nu_bar = lattice + rho. Its
selected residual delta gives z = nu_bar + delta/dt; its selected evidence gives
w. The PR1 update still uses its original gain, learned n_s, step clip and rho
clip. The second-order sums consume detached z and w; neither n_s nor the gate
scales the measured acceleration. Feature transport does not feed back into the
snapshot encoder, so order-1 and order-2 estimator measurements on the same
input frames are identical.

At frame time t the new displacement measurement spans [t−dt,t]. Its timestamp
is therefore t−dt/2. This is a deliberate half-step clarification of the proposed
integer-index convention: shifting every fit timestamp by dt/2 leaves the slope
unchanged, but the correction must use the physical measurement midpoint to
predict v(t) and the mean velocity over [t,t+dt] consistently.

With C_tt > 0, the slope is alpha_hat = C_tz/C_tt. The effective sample count is
n_eff = S_w²/sum(w²). C_tt < 1e-6 or n_eff < 3 closes the gate and returns a zero
slope. The empirical variance model is:

```
RSS = max(C_zz − dot(C_tz, alpha_hat), 0)
s² = RSS / (S_w − 2 S_w/n_eff)
Var(alpha_hat) = inflation × s²/C_tt       # trace over x and y
 g = clip(1 − kappa Var(alpha_hat)/||alpha_hat||², 0, 1)
```

Positive denominator guards avoid NaNs at empty/degenerate nodes. A zero slope
closes the gate. `--accel_kappa inf` bypasses the fit and keeps the extra state
zero, giving full state identity as well as closing the gate without inf×0. All fit
statistics, variance, slope, gate and mode decisions are detached and fp32.
The empirical residual model does not assume independent constant displacement
noise: consecutive one-step measurements share frames, motivating the inflation
sweep. That scalar correction does not remove association and convergence bias.

## Motion, resets and prediction

During observation, instantaneous velocity is
nu(t) = nu_bar + g alpha_hat(t−t_bar). Each step transports features through
nu(t) dt + 0.5 g alpha_hat dt². The effective instantaneous offset is clamped to
the node's overlapping ±0.75 cell. The PR1 rho clamp is retained independently.
`clamp_rate_obs` and `clamp_rate_pred` in model aux report the fraction of
node-steps whose additive correction requests an instantaneous velocity outside
that cell. The within-step acceleration term remains in the displacement.

When the selected identity changes between consecutive two-mode splits, or a
locked node becomes unlocked, its fit sums are cleared **before** the current
measurement is added. A disappearing/reappearing split alone is not a mode
switch. Split identities use the canonical principal-axis left/right ordering.
The PR1 mean/evidence are not reset. Bounce transfers the original prefix,
reflects rho, and discards the discontinuous acceleration fit/correction. The
shared snapshot remains in the observation frame. On-lattice comoving boosts
shift z_bar and the lattice velocity by u; centred moments, slope and correction
remain invariant. Off-lattice node mergers retain the prior approximate merge
policy and do not establish exact acceleration-fit equivariance.

Prediction freezes the final observation fit and gate:

- `--node_update_pred off`: no new mean or snapshot measurements; the correction
  advances by g alpha_hat dt each step.
- `--node_update_pred on` (default): the PR1 running mean continues consuming the
  model's generated input frames; the frozen correction also advances by
  g alpha_hat dt. Prediction measurements never refit acceleration or trigger
  fit resets.

Away from clamps, fixed acceleration composes to nu tau + 0.5 alpha tau².
Node trajectories, moment velocity diagnostics and `node_velocity_error` use the
outgoing interval mean v(t)+0.5 g alpha_hat dt, matching the environment's vel[t].
The instantaneous v(29) and the mean over [29,30] differ by a/2.

When the gate is zero, explicit selection preserves the original arithmetic on
velocity, including the lattice addition. The bit-identity tests compare both
orders with κ=inf, including prediction on/off and checkpointed gradients.
Checkpoint closures capture observation/prediction and node-update decisions;
recording diagnostics runs without autograd side effects.

Two controls are available on the lock path. `--nu_forget F` multiplies the PR1
running evidence by F before an update; F=1 bypasses the multiplication.
`--pred_shrink learned` multiplies velocity by sigmoid(theta) before each
prediction transport, then applies the usual measurement update. One learned
scalar is initialized with logit(0.999); `none` adds no parameter or operation.
With order 2, the carried correction is shrunk too. These controls require
separate trained comparisons to distinguish acceleration estimation from a
prior toward slower motion.

## Environment

For each digit, propose v0 uniformly in [-vmax,vmax]², and propose zero
acceleration with probability accel_zero_p. Otherwise draw a uniform direction
and a log-uniform magnitude in [accel_min,accel]. Jointly reject and redraw
**both** v0 and a until their velocities at times 0 and accel_horizon lie within
the componentwise speed bound. Linear velocity makes those endpoint checks
sufficient throughout the interval. Acceleration mode uses continuous v0 even
if `--real_velocity` is omitted. Camera sampling still follows `--real_camera`.

Positions follow p0+v0 t+0.5 a t², wrapped on the world torus. Only acceleration
mode evaluates this closed form; it uses float64 before wrapping and stores
fp32 positions to keep long-horizon coordinate error below 1e-5 pixels. The
non-acceleration incremental loop and RNG consumption are unchanged. The renderer
is shared by the generator and ceiling script: Fourier placement, MAX digit
composition, real camera shift, crop and [0,1] clamp.

`digit_states.acc` is [B,n_digits,2], including all zeros when acceleration is
disabled. `vel[t]` is v0+a(t+dt/2), the mean over the **outgoing** interval.
The horizon is in time units, so dt=0.5 uses the same velocity/acceleration law.
Subsampling adjusts interval-mean velocities to the longer interval. Acceleration
with walls, a speed ring or random scene boosts is rejected because those modes
would violate this simple bounded joint law. Explicit diagnostic boosts are
applied after sampling and may exceed the original speed bound.

Default tags are unchanged. `_acc<A>` is appended only when A is nonzero;
nondefault zero probability, minimum and horizon are appended only in that mode.
Old self-described, acceleration-disabled caches retain their filenames and
load after their original hash and normalized configuration are checked. Existing
cached tensors and evaluation reductions are not rewritten. Source changes keep
the PR5a warning behavior. Checkpoint flags and evaluation JSON record the new
settings; eval can override order, kappa and forgetting without retraining.

## Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--accel` | 0 | Maximum magnitude, px/time-unit²; 0 disables |
| `--accel_zero_p` | 0.3 | Proposal zero probability |
| `--accel_min` | 0.002 | Minimum nonzero magnitude |
| `--accel_horizon` | 180 | Bounded-velocity horizon in time units |
| `--order` | 1 | 2 enables acceleration on adaptive lock nodes |
| `--accel_kappa` | 9 | Significance strength; inf closes the gate |
| `--accel_var_inflation` | 4 | Conservative fallback; null target not met |
| `--nu_forget` | 1 | Evidence retention, 0 < F ≤ 1 |
| `--pred_shrink` | none | none or learned sigmoid velocity shrink |

## Limitations

The joint speed bound induces a strong deceleration correlation; a trained model
may exploit that shortcut. Small accelerations can be undetectable over 30
partially observed frames. Occlusion, changing associations and frequent unlocks
can leave fewer than three effective measurements. Principal-axis mode labels
can be unstable for nearly degenerate splits. Large acceleration eventually
reaches the overlapping cell clamp, limiting long-horizon motion. A frozen
prediction fit cannot recover from an incorrect observation-phase slope. The
constant-velocity `oracle` model is undefined on K-accel and train/eval reject
that combination; the rendering ceiling is a separate diagnostic.

The gate's false-positive target remains unmet on the requested MNIST null set.
A synthetic MA(1) measurement-noise test passes 2%, but it does not validate
calibration under real image association errors. The controlled single-digit
scene keeps the digit visible with camera velocity v0+U[-.25,.25]²; it tests
recoverability under visibility, not the full random-camera distribution.
No trained end-to-end improvement is established by these estimator tests.

## Realised law and rendering ceiling

`scripts/accel_law.py`, seed 0, 400,000 accepted digits: acceptance
0.725489, realised zero fraction 0.414147.
The proposal zero probability is 0.3; rejection raises the realised fraction.

| Nonzero magnitude percentile | Magnitude (px/time-unit²) |
| --- | --- |
| 10 | 0.00234653 |
| 50 | 0.00467181 |
| 90 | 0.01185401 |
| 99 | 0.01812620 |

| Magnitude bin | corr(a_k, v0_k), pooled x/y |
| --- | --- |
| [0.002, 0.004) | -0.170013 |
| [0.004, 0.008) | -0.354601 |
| [0.008, 0.02) | -0.699363 |

| Horizon tau | Median 0.5 × magnitude × tau², all digits | Nonzero digits |
| --- | --- | --- |
| 20 | 0.5060 px | 0.9344 px |
| 50 | 3.1625 px | 5.8398 px |
| 150 | 28.4625 px | 52.5578 px |

The ceiling uses 300 sequences × 5 MNIST test digits, seed 1, real camera,
and the production MAX renderer. Positions at frame 29 are known perfectly;
MSE@H averages frames 30 through 29+H. Values include sequence-level standard
errors; this is an independent Torch draw from the planning script's NumPy draw.

| Order-1 extrapolation velocity / baseline | MSE@20 ± SE | MSE@50 ± SE |
| --- | --- | --- |
| Mid-observation v(14.5) | 0.030259 ± 0.001093 | 0.072647 ± 0.001897 |
| Instantaneous v(29) | 0.007597 ± 0.000365 | 0.047950 ± 0.001359 |
| Forward difference p(30)−p(29) | 0.006875 ± 0.000335 | 0.046867 ± 0.001334 |
| Observed backward difference p(29)−p(28) | 0.008337 ± 0.000395 | 0.049015 ± 0.001384 |
| Repeat last observed frame | 0.226564 ± 0.002819 | 0.230396 ± 0.002660 |
| Zero frame (planning reference “still”) | 0.156261 ± 0.002053 | 0.156718 ± 0.001734 |

The planning reference's “last finite difference” used the first future position,
and its “still frame” was actually zero. Both reference definitions and the
observable backward difference/true last-frame baseline are reported explicitly.
The independent ceiling draw gives lower point estimates than the planning
reference. This 1,500-digit ceiling draw has zero fraction 0.442667, compared
with 0.414147 in the 400,000-digit law check, which can contribute to its lower
MSE. The table reports sampling uncertainty; the law statistics agree closely.
Raw results are in [accel_law.json](../results/order2_gate/accel_law.json).

## Gate harness results

100 sequences, seed 777, batches of two, frozen C=32/K=8 encoder, seed 0.
Hungarian matching assigns all digits before the last-ten-frame visibility filter.
The complete calibration and detection table is in [GATE_order2.md](GATE_order2.md);
raw results are in [results/order2_gate](../results/order2_gate/).

At κ=9, null node-step open rates for inflation 1/2/4 are 3.628% / 2.963% / 2.161%.
None is below 2%; inflation 4 is the most conservative allowed fallback.
Its frame-29 null rate is 2.960%. No extra gate guard was added to force a pass.

| Estimator | Matched median velocity error at v(29.5) | Mean |
| --- | --- | --- |
| order1 | 0.060391 | 0.112824 |
| order2 | 0.058931 | 0.112322 |
| order1_F0.9 | 0.046026 | 0.101751 |
| order1_F0.95 | 0.052192 | 0.107232 |

| Acceleration magnitude bin | Assigned visible digits | Gate open at frame 29 |
| --- | --- | --- |
| [0, 1e-07) | 224 | 4.464% |
| [0.002, 0.004) | 97 | 10.309% |
| [0.004, 0.008) | 100 | 1.000% |
| [0.008, 0.02) | 79 | 12.658% |

The controlled one-digit scene with a=(.004,0) has velocity error
0.00307922, acceleration error 0.00071728,
and gate 0.972892. This yields the fast regression tolerances of
0.0034 px/frame for velocity and 0.00079 px/frame² for acceleration (10% margin).

The completed production-data run accounts for 653.59 seconds
of single-thread CPU wall time, including the K-real calibration.
The order-2 matched median improves slightly over
order 1 here; both forgetting controls perform better. Detection is sparse and
nonmonotonic across acceleration bins. These results do not establish robust
acceleration detection or improved trained predictions.

## Validation

CPU only, Python 3.12 / PyTorch 2.9.0, one OMP/OpenBLAS thread:

| Command / scope | Result | Runtime |
| --- | --- | --- |
| `python -m pytest -q` | 187 passed, 3 slow cases skipped | 67.71 s |
| `KWM_SLOW=1 python -m pytest -q` | 190 passed | 170.06 s |
| Fast `tests/test_order2.py` | 30 passed, 1 slow case skipped | 6.93 s |

Both full suites emit the existing dependency deprecation warning. Frozen
references come from integration revision `0c1198a048b4c0d7db310552efa2112bc4dfd9db`.
Tests cover disabled-environment tensors, both PR1 cells and their input/parameter
gradients, full state/output/gradient identity with κ=inf, empirical weighted
variance, synthetic MA(1) null noise, controlled image acceleration recovery,
transport composition, both prediction policies, checkpoint replay, boosts,
bounce resets, fractional time, clamp rates, evaluation overrides and old cache
compatibility. The slow acceleration regression checks the first 12 sequences
against the published 100-sequence harness output. The panorama helper also
passes order-2 prediction on/off parity after a preceding fractional-time setting.
