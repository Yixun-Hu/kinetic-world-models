# Second-order lock nodes and K-accel

`--order 2` is an experimental opt-in for `--model adaptive --node_update lock`.
Order 1 remains the default; both LK and lock retain bit-identical outputs,
states and gradients against the frozen pre-change cell and model references.
The CPU result is **NO-GO**: selected order 2 has displacement proxy mean
5.96454 px, versus 5.94696 px for best forgetting (F=0.9), about 0.30% worse.
At κ=9 the null rate is 0.396%, passing the 2% target; the default κ=4 has
2.074% null and fails that target. The required displacement improvement is
at least 10%. All sweep and calibration results
are in [GATE_order2.md](GATE_order2.md). No trained forecasting improvement is established.

## Estimator and gate

Both lock orders use `LockState(features, nodes, snapshot)`, with 17 fp32 node
columns. The first four columns retain the original running offset, evidence,
and lock bit. Order 1 leaves all additional columns zero.

| Columns | Meaning |
| --- | --- |
| 0–3 | Running offset rho (2), evidence W, lock bit |
| 4–6 | Fit weight S_w, weighted time t_bar, centred time moment C_tt |
| 7–10 | Weighted velocity z_bar (2), centred cross moment C_tz (2) |
| 11–12 | Sum of squared weights S_w2, centred velocity scatter C_zz (trace) |
| 13 | Diagnostic split label: 0 unsplit, 1 left, 2 right |
| 14 | Next frame timestamp, advanced during prediction too |
| 15–16 | Correction relative to the current running mean |

The detached measured velocity is z = lattice + rho + delta/dt. Its timestamp
is the midpoint of the measured interval. The learned PR1 scale n_s affects
only the running-mean update, not the fitted acceleration. Weighted Welford
updates avoid subtracting large raw moments and require no stored history.

```
n_eff = S_w² / S_w2
alpha_hat = C_tz / C_tt
RSS = max(C_zz − dot(C_tz, alpha_hat), 0)
s² = RSS / (S_w (1 − 2/n_eff))
Var(alpha_hat) = s² / C_tt × S_w2/S_w       # trace over x and y
g = clip(1 − kappa Var(alpha_hat)/||alpha_hat||², 0, 1)
```

This uses the review's allowed Kish approximation to the weighted sandwich,
exact for equal weights. It is invariant to multiplying every evidence weight
by a constant. With heterogeneous weights it approximates the time leverage;
it is not a claim of exact heteroscedastic coverage under arbitrary association
errors. The empirical null tests therefore remain necessary. Gates require
n_eff ≥ 4 and weighted time variance C_tt/S_w ≥ 1e-6. Zero slopes close the gate.
`--accel_kappa inf` bypasses fitting and preserves full order-1 state identity.

κ acts on the **2-D covariance trace**. With equal component variances, κ=9
requires an acceleration norm above sqrt(18) ≈ 4.2 component standard errors;
it does not mean three standard errors. The default is κ=4, with κ=9 available.
Under the ideal isotropic Gaussian, known-variance null, the open probability
is exp(−κ): approximately 1.8% and 0.012%, respectively. Estimated variance,
small effective samples and association noise need not follow that ideal law.

Variance inflation was tried at 1, 2 and 4 before review. It was the wrong fix:
the original variance scaled inversely with pixel-evidence magnitude. Inflation
could not repair that units error and has been removed from the CLI and sweep.
The earlier MA(1) attribution was also incorrect. For displacement noise
z_t = v + e_t − e_(t−1), the slope numerator telescopes; the white-noise plug-in
**overestimates**, rather than underestimates, long-window slope variance
(roughly by N/3). Persistent association errors can instead produce positive
correlation. Remaining calibration limitations include those errors and small
effective sample sizes.

## Resets, motion and prediction

Before adding a measurement, clear the fit on lock loss or a measured velocity
jump. A jump is ||z_t − z_hat_t|| > 6s, where z_hat_t is the old fit evaluated at
the new measurement timestamp, and s is the residual trace scale above, floored
at 1e-4 px/time-unit for roundoff. The test applies only after at least three
effective measurements and only when the new measurement has positive weight.
Principal-axis label changes do not reset the fit. The original running mean
and its evidence are not reset.

Instantaneous velocity uses the fit's own intercept:

```
nu(t) = (1−g) nu_bar + g (z_bar + alpha_hat (t−t_bar))
```

An explicit zero-correction selection preserves the order-1 arithmetic at g=0.
The instantaneous offset is clamped to the overlapping ±0.75 node cell; outgoing
transport then uses nu(t) + 0.5 g alpha_hat dt. Diagnostics and velocity matching
use this outgoing interval mean, matching the environment's `vel[t]` convention.
Clamp rates count evidence-bearing node-steps, include the half-step acceleration
term, and report order-1 offset saturation on the same denominator.

Prediction freezes the observation fit and gate:

- **`--node_update_pred off` is the primary experimental setting.** The running
  mean and snapshot freeze; the fit is evaluated at each new time.
- `on` preserves the existing default for order-1 compatibility. The running
  mean consumes generated frames, but each correction is recomputed as
  g × (frozen fit extrapolation − current running mean). It never adds an
  independently advancing correction on top of that updated mean.

Neither prediction policy refits acceleration or triggers fit resets.
`--nu_forget F` discounts the running evidence before each update; F=1 bypasses
that multiplication. Order 2 + F fits the resulting measurement stream, without
forgetting the fit itself. `--pred_shrink learned` scales the running velocity,
correction and entire frozen fitted velocity line, including its acceleration;
its sigmoid factor starts at 0.999. `none` adds no parameter or operation.

Bounce reflects the original prefix and clears the fit/correction. On-lattice
boosts shift z_bar while retaining centred moments and acceleration. Off-lattice
merges remain approximate; split labels use a mass-weighted categorical vote,
not an arithmetic average.

## Environment and flags

Acceleration mode proposes v0 uniformly in [-vmax,vmax]², zero acceleration
with probability `accel_zero_p`, and otherwise a uniform direction with
log-uniform magnitude in [`accel_min`, `accel`]. Jointly redraw v0 and a until
velocities at time 0 and `accel_horizon` satisfy the componentwise speed bound.
Those endpoint checks suffice for the full linear-velocity interval.

Positions use p0 + v0 t + 0.5 a t² in float64 before torus wrapping; stored tensors
are fp32. Rendering uses the existing Fourier placement, MAX composition,
camera shift, crop and clamp. `vel[t]` is v0+a(t+dt/2), the outgoing interval
mean; subsampling adjusts it to the longer interval. Units are pixels per time
unit and pixels per time-unit², also at dt=0.5.

Acceleration forces continuous initial velocities and therefore uses a `vel1`
tag even without `--real_velocity`. `_acc<A>` and nondefault acceleration-law
fields appear only when acceleration is enabled. Disabled-environment tensors,
RNG consumption and tags remain unchanged. Walls, speed rings and random scene
boosts are incompatible with this bounded acceleration law. The constant-velocity
oracle is also rejected on K-accel; the rendering ceiling is a separate diagnostic.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--accel` | 0 | Maximum magnitude, px/time-unit²; zero disables |
| `--accel_zero_p` | 0.3 | Proposal probability of zero acceleration |
| `--accel_min` | 0.002 | Minimum nonzero proposed magnitude |
| `--accel_horizon` | 180 | Speed-bound horizon in time units |
| `--order` | 1 | 2 enables the experimental acceleration fit |
| `--accel_kappa` | 4 | 2-D trace threshold; 9 available; inf closes the gate |
| `--nu_forget` | 1 | Running-evidence retention, 0 < F ≤ 1 |
| `--pred_shrink` | none | none or learned sigmoid motion shrink |

## Realised law and rendering ceiling

Seed 0, 400,000 accepted digits: acceptance 0.725489; realised zero fraction
0.414147 (proposal probability 0.3). Nonzero magnitude p10/median/p90/p99 is
0.002347 / 0.004672 / 0.011854 / 0.018126 px/time-unit². Componentwise correlations
between a and v0 are −0.170 / −0.355 / −0.699 in magnitude bins [0.002,0.004),
[0.004,0.008), [0.008,0.02). The speed bound thus induces a deceleration prior.

The ceiling uses 300 five-digit sequences, seed 1, MNIST test images, real camera
and perfect positions at frame 29. MSE@H averages future frames 1 through H.

| Extrapolation / baseline | MSE@20 ± SE | MSE@50 ± SE |
| --- | --- | --- |
| Mid-observation v(14.5) | 0.030259 ± 0.001093 | 0.072647 ± 0.001897 |
| Instantaneous v(29) | 0.007597 ± 0.000365 | 0.047950 ± 0.001359 |
| Forward difference p(30)−p(29) | 0.006875 ± 0.000335 | 0.046867 ± 0.001334 |
| Observed backward difference p(29)−p(28) | 0.008337 ± 0.000395 | 0.049015 ± 0.001384 |
| Repeat last observed frame | 0.226564 ± 0.002819 | 0.230396 ± 0.002660 |
| Zero frame | 0.156261 ± 0.002053 | 0.156718 ± 0.001734 |

Raw law and ceiling results: [accel_law.json](../results/order2_gate/accel_law.json).

## CPU gate and limits

The gate uses 100 sequences per five-digit dataset, seed 777, a frozen C32/K8
encoder, F ∈ {0.6,0.7,0.8,0.9,0.95} and κ ∈ {4,9}. All arms share a baseline
Hungarian assignment among evidence-bearing nodes. The displacement proxy is
the mean Euclidean displacement error at τ=20 and 50, averaged equally over
horizons, using each final node's velocity and gated acceleration. It does not
include future clamps or image rendering. The null tables report both the
all-evidence and n_eff-eligible denominators and the n_eff distribution.

The separate paired single-digit ensemble uses 50 seeds, a fast C2/K2 encoder
(world32/window24), and a camera tracking v0 plus uniform jitter. It selects the
highest-evidence node without using truth. Its zero-acceleration twin has 0/50
false positives at either κ; a=(0.004,0) detection is 50/50 at κ=4 and 47/50 at
κ=9. This easier scene is not the five-digit random-camera distribution, and
50 null seeds cannot establish a population false-positive rate below 2%.

The go/no-go criterion is **K-real null node-step rate < 2% and standalone order 2
at least 10% better than the best F on displacement proxy mean**. It is not met.
The κ=9 null rate is 0.396% over evidence-bearing node-steps and 1.149% among
n_eff-eligible node-steps. With F=0.9 these rates are 0.566% and 1.667%.
Default κ=4 gives 2.074% (eligible 6.027%), or 2.524% (eligible 7.438%) with F=0.9.

| Arm | Median velocity error | Mean displacement @20 | @50 | Proxy mean |
| --- | --- | --- | --- | --- |
| Order 1 | 0.060391 | 2.74983 | 9.19514 | 5.97248 |
| Best F = 0.9 | 0.050203 | 2.75338 | 9.14054 | 5.94696 |
| Order 2, κ=4 (default) | 0.058527 | 2.78613 | 9.48371 | 6.13492 |
| Order 2, κ=9 | 0.060391 | 2.74616 | 9.18292 | 5.96454 |
| Order 2 + F=0.9, κ=9 | 0.050215 | 2.80049 | 9.39653 | 6.09851 |

At κ=9, detection in the three nonzero acceleration bins is 0%, 2%, 0%
(97, 100, 79 assigned digits); at κ=4 it is 0%, 4%, 5.06%. Final K-accel n_eff
has p10/median/p90 of 1.15/4.31/9.73; 53.36% of nodes reach the floor of four.
Order 2 + best F has median n_eff 4.24. The full null and detection tables,
including both κ values, are in [GATE_order2.md](GATE_order2.md).
Any future GPU comparison should first evaluate these estimator overrides on
order-1-trained checkpoints with prediction updates off. An order-2 training
campaign is not justified by estimator diagnostics alone.
