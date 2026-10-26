# Prediction-input pilot gate

**NO-GO for frozen prediction.** Every pure-frozen, pre-lift, de-ghosted and
mix-once control fails the joint MSE@20/@150 gate. The sole exception is
**zero-input for 20 steps, then freeze, on lock-10k**. That exception fails on
LK-13k and on the reviewer’s lock-9k/LK-11k checks: **1/4 checkpoint/rule
combinations**, not a robust prediction policy. Frozen overrides are retained
as documented evaluation-only negative controls; training accepts flowm|zero.

## Protocol and checkpoint identity

All newly reproduced rows use the **same first 16 sequences** of the existing
32-sequence selection cache, 30 observations and 150 predictions, noisy camera
actions, fp32 CPU inference, no autocast, and **four torch threads**. Predictions
are clamped to [0,1]. MSE@H averages **all first H frames**, sequences and
viewport pixels. It is not the error at frame H. The selection seed is 12345;
no cache was regenerated. Cache SHA-256:

`1d3f58ce06e2c743f9d26135e47fa8d4aa66da98f08baa27f8764597bcf266f4`.

The archives are pinned read-only inputs, independent of advancing last.pt files:

| Checkpoint | Step | File SHA-256 |
| --- | ---: | --- |
| pilot_lock/ckpt_step10000.pt | 10000 | `a78fb0bbeeae9f2d198c4610e8f8ec5ee3b893e95ac33dabf0aca7e6aaa0a0ab` |
| pilot_lk/ckpt_step13000.pt | 13000 | `8404e718d1c9c7074e0a3e683189e9492e32f6c04c733d694df092799efa7a94` |

Prediction node updates are enabled for flowm and zero. Frozen transport
ignores that setting. The observation pass is shared across policies. K=0
freezes the actual last pre-lift intermediate; K>0 **drops the last K
observations**, then replays their actions with zero inputs and no node
measurements. K=1 is effectively K=0 up to the dropped final frame and
activation placement. K remains an explicit sweep knob, default 0; the default
sweep retains K=0/4 and omits K=1/2. See [exact definitions](PR3_pred_input.md).

## Reproduced gate and reviewer variants

A candidate passes only if **both** MSE@20 and MSE@150 are no greater than
`min(flowm, zero)` on the same checkpoint and subset. There is no tolerance
or significance adjustment. MSE@50 is descriptive. Each cell below is
**MSE@20 / MSE@50 / MSE@150**.

| Policy / experiment | lock-10k | LK-13k | Gate: lock / LK |
| --- | --- | --- | --- |
| flowm | 0.029367 / 0.044946 / 0.083409 | 0.035295 / 0.058887 / 0.111627 | reference / reference |
| zero | 0.026780 / 0.042603 / 0.084857 | 0.031235 / 0.054237 / 0.095042 | reference / reference |
| frozen | 0.060444 / 0.085372 / 0.124119 | 0.084617 / 0.120428 / 0.174394 | FAIL / FAIL |
| frozen_prelift K=0 (default) | 0.048426 / 0.073138 / 0.117348 | 0.052015 / 0.085262 / 0.142220 | FAIL / FAIL |
| frozen_prelift K=4 | 0.043644 / 0.065184 / 0.104567 | 0.052710 / 0.080658 / 0.134549 | FAIL / FAIL |
| Per-pixel winner-take-all, post-lift | 0.051059 / 0.058438 / 0.080043 | 0.070175 / 0.081852 / 0.107982 | FAIL / FAIL |
| Per-pixel winner-take-all, pre-lift + ReLU | 0.071438 / 0.074249 / 0.091121 | 0.099645 / 0.101588 / 0.116748 | FAIL / FAIL |
| Oracle true-velocity node in digit boxes; WTA elsewhere | 0.056730 / 0.063243 / 0.082611 | 0.073464 / 0.084930 / 0.109949 | FAIL / FAIL |
| Locked nodes only (flag > 0.5) | 0.075083 / 0.089555 / 0.111441 | — | FAIL / — |
| Transport, then one mix + ReLU | 0.044932 / 0.071198 / 0.116984 | 0.047989 / 0.082491 / 0.141440 | FAIL / FAIL |
| Zero-input for 20 steps, then frozen | 0.026780 / 0.036749 / 0.065977 | 0.031235 / 0.048716 / 0.097513 | **PASS** / FAIL |

Rows after the pre-lift controls are **script-internal experiments**, reproduced
by `scripts/pred_input_gate.py --variants`; they are not kwm.eval modes.
Winner-take-all retains the largest squared-energy node independently at every
pixel. Oracle masking keeps the nearest true-velocity node in visible digit
boxes at the final observation and uses WTA elsewhere; later digits win box
overlaps, matching the reviewer. Locked-only retains a mean **4.75 nodes per
sample** on lock (all nodes are retained if none is locked). The hybrid reuses
the first 20 zero-input recurrent states/predictions and then transports its
step-20 anchor without further recurrence. Its MSE@20 equals zero by construction.

## Short-horizon mismatch versus long-horizon ghosts

At **τ=1**, one learned mix plus ReLU **after** frozen transport matches the
zero-input prediction. The script checks the unclamped prediction tensors,
not just their scalar MSE:

| Checkpoint | Frozen MSE at τ=1 | Zero MSE at τ=1 | Mix-once MSE at τ=1 | Max prediction difference: mix vs zero |
| --- | ---: | ---: | ---: | ---: |
| lock-10k | 0.021147 | 0.013967 | 0.013967 | 0 |
| LK-13k | 0.025532 | 0.013331 | 0.013331 | 2.38e-07 |

The initial gap is the **missing mix step / decoder-state mismatch**, not
ghost separation. Mix-once still fails at @20 and @150. Over longer horizons,
analytic transport separates content duplicated across velocity nodes.
De-ghosting recovers much of the tail: post-lift WTA and oracle masking bring
lock MSE@150 within about 4% of the better baseline (slightly better), while
LK retains a residual gap. **De-ghosting fixes much of the tail, but not
MSE@20**; even the oracle mask fails the joint gate. For these recurrently
trained decoders, recurrence is still needed at the trained 20-step horizon.

## Hybrid robustness and H3

The following are **independent reviewer results at d093e2e, 32 sequences**,
not newly generated first-16 rows. They are retained separately to avoid
conflating checkpoints or subsets:

| Checkpoint | Zero @20 / @50 / @150 | Zero20-then-freeze @20 / @50 / @150 | Flowm @150 | Hybrid gate |
| --- | --- | --- | ---: | --- |
| lock-9k | 0.0218 / 0.0343 / 0.0689 | 0.0218 / 0.0344 / 0.0720 | 0.0749 | FAIL |
| LK-11k | 0.0284 / 0.0475 / 0.0833 | 0.0284 / 0.0484 / 0.0946 | 0.0924 | FAIL |

The hybrid wins only at lock-10k, where zero’s long-horizon error has degraded.
It does not justify a viable frozen prediction mode. Its lack of robustness
closes the proposed hybrid alongside the pure-frozen and de-ghosting controls.

Across the four pilot checkpoint/rule combinations, **zero beats flowm at
MSE@20 in 4/4, but at MSE@150 in only 3/4** (lock-10k is the exception).
H3 therefore uses **MSE@20 as primary**, with MSE@50/@150 secondary under Holm
and MSE@150 stability reported. The held-out, five-seed, checkpoint-averaged
design and the conditional three-seed trained-zero arm are recorded in the
[H3 plan amendment](PR3_pred_input.md#h3-plan-and-amendment).

## Node energy diagnostics

`node_concentration` is the nearest-true-velocity node’s squared feature energy
divided by all-node energy in each visible digit box. It **does not track MSE**:
flowm scores below frozen despite much lower prediction error. Background,
overlaps and older trails dilute it; its flat frozen profile does not show
copies separating. It is consistent with, but not diagnostic of, ghosting.

`effective_node_count` computes `exp(-sum_i p_i log p_i)` from the per-node
squared feature-energy distribution **at each viewport pixel**, then averages
over pixels/sequences (and all prediction frames for `mean_prediction`).
Zero-energy pixels count as zero. JSON contains τ=0, 1, 10, 20 and the full
prediction mean. τ=0 always means the ordinary final observation state:

| Checkpoint | Per-pixel effective nodes at τ=0 | Window-pooled effective nodes | Largest pooled node share |
| --- | ---: | ---: | ---: |
| lock-10k | 19.667 | 23.036 / 25 | 0.0863 |
| LK-13k | 22.071 | 24.510 / 25 | 0.0618 |

The reviewer’s 23–24.5 / 25 count pools energy over the window **before**
entropy; it is a separate statistic, not the requested per-pixel mean. Both
show broadly distributed energy. The original controls have:

| Checkpoint | Policy | Concentration τ=1 | τ=10 | τ=20 | Mean per-pixel effective nodes, τ=1…150 |
| --- | --- | ---: | ---: | ---: | ---: |
| lock-10k | flowm | 0.1211 | 0.1472 | 0.1445 | 19.429 |
| lock-10k | zero | 0.1578 | 0.1920 | 0.2016 | 15.303 |
| lock-10k | frozen | 0.1264 | 0.1522 | 0.1439 | 15.930 |
| lock-10k | frozen_prelift K=0 (default) | 0.1592 | 0.1676 | 0.1602 | 16.098 |
| lock-10k | frozen_prelift K=4 | 0.1841 | 0.1861 | 0.1752 | 16.334 |
| LK-13k | flowm | 0.0808 | 0.0830 | 0.0804 | 22.101 |
| LK-13k | zero | 0.0953 | 0.0995 | 0.0924 | 18.770 |
| LK-13k | frozen | 0.0846 | 0.1035 | 0.0961 | 19.008 |
| LK-13k | frozen_prelift K=0 (default) | 0.0964 | 0.1047 | 0.0987 | 19.836 |
| LK-13k | frozen_prelift K=4 | 0.1016 | 0.1030 | 0.1013 | 20.076 |

Default kwm.eval computes both diagnostics on every main rollout, adding cost
and JSON fields. Labels never enter the normal dynamics. Only the explicitly
labelled script oracle uses truth to construct its mask.

## Runtime, validation and limitations

**47.07 CPU-core minutes; 12.97 wall minutes** for the final
four-thread `--variants` sweep. All four flowm/zero baseline rows were reused
after matching pinned checkpoint/cache hashes, the first-16 subset, numerical
source dependencies, software and thread count. Their original provenance is
retained. Zero’s first 20 steps are rerun only to recover the hybrid anchor.
One observation pass per checkpoint, cached anchor FFTs, compact carried LK
metadata, shared frozen/mix-once transports, and omission of K=1/2 remove
redundant work. Timings come from process totals in `runtime.json`; summing row
times would double-count the frozen/mix-once group.

The default flowm/zero/frozen/K=0/K=4 controls account for **at most 25.74
CPU-core minutes** of this run: subtract only the exclusively timed optional
variants, conservatively leaving shared mix-once work, hybrid-prefix recovery
and setup charged to the default controls. This is a measured component bound,
not an additional standalone benchmark. The default gate is below the requested
30–40 core-minute budget; the complete optional-variants sweep exceeds it.

The one-time refresh and initial reproduction took **74.10 CPU-core minutes /
20.27 wall minutes**, before the compact-LK optimization. This additional
cost is reported in `runtime_refresh.json`, not hidden in the recurring-gate
figure. The previous JSONs used last.pt serialization hashes and lacked the
new entropy diagnostic, so the four baselines were recomputed once. Combined
gate execution for this revision was **121.17 CPU-core minutes /
33.24 wall minutes**. Thus the recurring sweep is trimmed;
bootstrapping the refreshed baseline artifacts costs extra.

One earlier cache-validation attempt stopped on a reduction-order assertion
after approximately **43.5 seconds wall time** (from log timestamps; CPU time
was not captured). That failed attempt is additional to the completed-run
totals above. A standalone 20-frame reduction rounded differently from the
same prefix in a 150-frame tensor. The script now preserves the full-horizon
layout and checks exact scalar, per-sequence and framewise prefix equality;
regression tests cover one and four threads. Gate comparisons use no tolerance.

Validation used the specified flowm Python, `CUDA_VISIBLE_DEVICES=""`, and at
most four torch threads. `python -m pytest -q`: **237 passed, 2 existing slow
tests skipped in 52.49 s**. `tests/test_pred_input.py`: **75 passed in 6.42 s**
(under 30 s). Both runs reported only the existing dependency deprecation
warning. Eight golden cases load checkpoints saved by the actual pre-PR3 code
and match its eval outputs with `torch.equal`; optimized gate transports,
carried velocities and decoding also have exact-equality regression tests.

These are selection-cache diagnostics, not held-out multi-seed evidence.
The decoders were trained on recurrent prediction states with teacher forcing
0.125; evaluation-only frozen states create a training/distribution mismatch.
The NO-GO does not establish the outcome of separately training a frozen model,
which is outside the plan. Ballistic periodic transport also omits future
collisions, velocity changes and new observations.

No dependency, pilot checkpoint or external cache was modified. The referenced
PLAN_v4_KWM.md and its amendment were absent from this worktree, so their H3
correction is recorded in PR3_pred_input.md. K=1/2 result files were removed
from the default sweep. [Reproduction command](REPRODUCE.md#pr3-prediction-input-pilot-gate);
[raw results](../results/pred_input_gate) include per-sequence prefix errors,
framewise MSE, diagnostics, identities, provenance, gate decisions and timing.
