# Prediction-input pilot gate

**NO-GO for pure frozen prediction.** Every pure-frozen, pre-lift, de-ghosted and
mix-once control fails the joint MSE@20/@150 gate. The N=20 hybrid exception is
**zero-input for 20 steps, then freeze, on lock-10k**. That cutoff fails on
LK-13k and on the reviewer’s lock-9k/LK-11k checks: **1/4 checkpoint/rule
combinations**, not a robust prediction policy. `kwm.eval --dec_input
zero_then_frozen --freeze_after N` now exposes the hybrid for standard
evaluation (default N=20). It needs checkpoint-averaged held-out evaluation.
Pure frozen overrides remain evaluation-only negative controls; training
accepts flowm|zero only. The regenerated sweep below also evaluates N=10/30.
N=10 passes on lock-10k only; **N=30 passes on both reproduced checkpoints**.
These cutoff comparisons use the selection cache and do not establish
checkpoint-averaged held-out performance. The default remains N=20.

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

Prediction node updates are enabled for flowm and zero, including the hybrid's
first N steps. None occur after freezing. The hybrid calls the full public
`kwm.eval.rollout` path, including the observation pass. Other controls share
an observation pass. K=0
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
| zero_then_frozen N=10 | 0.026049 / 0.040830 / 0.079821 | 0.030740 / 0.056447 / 0.114830 | **PASS** / FAIL |
| zero_then_frozen N=20 | 0.026780 / 0.036749 / 0.065977 | 0.031235 / 0.048716 / 0.097513 | **PASS** / FAIL |
| zero_then_frozen N=30 | 0.026780 / 0.039092 / 0.062963 | 0.031235 / 0.049556 / 0.090153 | **PASS** / **PASS** |

De-ghosting and mix-once rows remain **script-internal experiments**. The
`zero_then_frozen` rows use `kwm.eval.configure_prediction` and
`kwm.eval.rollout`; there is no separate gate implementation of the hybrid.
`scripts/pred_input_gate.py --variants --freeze_after 10 20 30` reproduces all
these rows. The gate sweep defaults to those three cutoffs and accepts subsets.
Winner-take-all retains the largest squared-energy node independently at every
pixel. Oracle masking keeps the nearest true-velocity node in visible digit
boxes at the final observation and uses WTA elsewhere; later digits win box
overlaps, matching the reviewer. Locked-only retains a mean **4.75 nodes per
sample** on lock (all nodes are retained if none is locked). The hybrid freezes
the Nth zero-input step's post-activation feature memory: the `out` of
`_Base.feature_update` in [cells.py](../kwm/cells.py#L62), carried by
`KineticWM._forward` in [model.py](../kwm/model.py). Its tail transports that
anchor with the carried node velocities and the cumulative camera actions
**since step N**, without more recurrence or activation. N>=20 has the same
MSE@20 as zero by construction. [Exact time convention](PR3_pred_input.md#exact-anchor-and-time-convention).

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

At N=20 the hybrid wins only at lock-10k, where zero’s long-horizon error has
degraded. Availability as an evaluation mode enables the checkpoint-averaged
held-out check; it does not establish robustness or justify a frozen training
arm. The N=10/30 sweep above is additional selection-set evidence, not an
independent held-out test or a reason to retune the registered H3 endpoint.

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
show broadly distributed energy. The controls and hybrid cutoffs have:

| Checkpoint | Policy | Concentration τ=1 | τ=10 | τ=20 | Mean per-pixel effective nodes, τ=1…150 |
| --- | --- | ---: | ---: | ---: | ---: |
| lock-10k | flowm | 0.1211 | 0.1472 | 0.1445 | 19.429 |
| lock-10k | zero | 0.1578 | 0.1920 | 0.2016 | 15.303 |
| lock-10k | frozen | 0.1264 | 0.1522 | 0.1439 | 15.930 |
| lock-10k | frozen_prelift K=0 (default) | 0.1592 | 0.1676 | 0.1602 | 16.098 |
| lock-10k | frozen_prelift K=4 | 0.1841 | 0.1861 | 0.1752 | 16.334 |
| lock-10k | zero_then_frozen N=10 | 0.1578 | 0.1920 | 0.1874 | 17.333 |
| lock-10k | zero_then_frozen N=20 | 0.1578 | 0.1920 | 0.2016 | 17.440 |
| lock-10k | zero_then_frozen N=30 | 0.1578 | 0.1920 | 0.2016 | 17.307 |
| LK-13k | flowm | 0.0808 | 0.0830 | 0.0804 | 22.101 |
| LK-13k | zero | 0.0953 | 0.0995 | 0.0924 | 18.770 |
| LK-13k | frozen | 0.0846 | 0.1035 | 0.0961 | 19.008 |
| LK-13k | frozen_prelift K=0 (default) | 0.0964 | 0.1047 | 0.0987 | 19.836 |
| LK-13k | frozen_prelift K=4 | 0.1016 | 0.1030 | 0.1013 | 20.076 |
| LK-13k | zero_then_frozen N=10 | 0.0953 | 0.0995 | 0.0948 | 21.053 |
| LK-13k | zero_then_frozen N=20 | 0.0953 | 0.0995 | 0.0924 | 21.182 |
| LK-13k | zero_then_frozen N=30 | 0.0953 | 0.0995 | 0.0924 | 21.148 |

Default kwm.eval computes both diagnostics on every main rollout, adding cost
and JSON fields. Labels never enter the normal dynamics. Only the explicitly
labelled script oracle uses truth to construct its mask.

## Runtime, validation and limitations

The regenerated PR3 follow-up sweep took **105.39 CPU-core minutes /
28.63 wall minutes**, with four torch threads and no CUDA.
All 25 rows were regenerated on the same pinned checkpoints and first 16
selection-cache sequences. No baseline rows were reused: the model source
changed, so the cache identity checks required recomputation. The hybrid rows
run full `kwm.eval.rollout` calls, including observations, recurrent prefixes,
analytic tails and all standard diagnostics. Other controls still share an
observation pass and cached anchor FFTs; frozen/mix-once share a timing group.
Use process totals in `runtime.json`, not the sum of row timings.

The full and focused test suites ran during part of this sweep; wall time includes that
contention, while the recorded process CPU time excludes the test process.
`runtime_refresh.json` is retained as a historical artifact from the preceding
PR3 revision, not part of this run.

Validation used the specified flowm Python, `CUDA_VISIBLE_DEVICES=""`, and at
most four torch threads. `python -m pytest -q`: **277 passed, 2 skipped, 1 warning in 54.38s**.
`tests/test_pred_input.py`: **111 passed, 1 warning in 9.72s** (under 30 s).
The only warning is the existing dependency deprecation warning. Tests cover
bit-identical endpoints (N=0 versus frozen, N>=T_pred versus zero), both node
update settings, the exact post-activation anchor and fractional-time camera
sum, gate/legacy/eval agreement within 1e-6, suffixes, JSON metadata, inference
throughput and rejection of training support. Eight pre-PR3 checkpoint golden
cases still match with `torch.equal`. N=20/30 gate prefixes also match zero's
scalar, per-sequence and framewise MSE@20 exactly. GPU execution was not run;
the mode uses the standard device-aware evaluator and transport operations.

These are selection-cache diagnostics, not held-out multi-seed evidence.
The decoders were trained on recurrent prediction states with teacher forcing
0.125. The hybrid needs checkpoint-averaged held-out evaluation; the pilot
does not establish the outcome of separately training a frozen model, which
is outside the plan. Ballistic periodic transport omits future collisions,
velocity changes and new observations after the freeze.

No dependency, pilot checkpoint or external cache was modified. The referenced
PLAN_v4_KWM.md and its amendment are absent from this worktree; the H3 correction
is recorded in PR3_pred_input.md. [Reproduction command](REPRODUCE.md#pr3-prediction-input-pilot-gate);
[raw results](../results/pred_input_gate) include per-sequence prefix errors,
framewise MSE, diagnostics, identities, provenance, gate decisions and timing.
