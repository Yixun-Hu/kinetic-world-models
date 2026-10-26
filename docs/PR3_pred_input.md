# Prediction-phase input policies

`kwm.eval --dec_input {flowm,zero,frozen,frozen_prelift}` overrides only the
prediction policy. Omitting it uses the checkpoint's saved policy, with `flowm`
as the fallback for old checkpoints. The normal observation pass, parameter
keys, default recurrence arithmetic, decoder and existing metric reductions
are unchanged. Checkpoints contain only `flowm` or `zero`; frozen modes are
evaluation-only negative controls that failed the pilot gate. Default forward
values are tested against golden outputs from the actual pre-PR3 code at
`0c1198a`, including checkpoints saved by that code. Training gradients also
have exact-equality regression coverage.

| Policy | Prediction dynamics |
| --- | --- |
| `flowm` | Feed the last observation, then each previous prediction, through the recurrent cell. |
| `zero` | Feed zeros through the same recurrence; retain the configured prediction node-update policy. |
| `frozen` | Freeze the final post-lift, post-activation feature memory and its velocity metadata; decode independent analytic transports. This is the negative control. |
| `frozen_prelift`, K=0 | Freeze the last observation's transported-and-mixed recurrent content, before adding its lift and before activation. Activate after each analytic transport. |
| `frozen_prelift`, K>0 | Take the state after observation T−K; replay the last K observation actions with zero inputs and node measurements disabled, then freeze. Activate after each analytic transport. |

`--frozen_k K` defaults to **0** and is only an explicit `frozen_prelift` sweep
knob. Require `0 <= K < T_obs`. The ordinary observation pass still processes
all frames: the K-step replay is a separate branch, and observation-state and
node-estimator diagnostics retain the ordinary pass. K>0 deliberately excludes
the last K observation lifts and measurements from the prediction anchor.
It therefore confounds more recurrence with less evidence. K=1 is effectively
K=0 up to the dropped final observation: `lift(0)=0`, so it uses the same
mixed content, with earlier velocity metadata and ReLU before transport as
well as after. Fractional shifts need not commute with ReLU, so this is not a
claim of bit equality. The default gate omits the redundant K=1 and K=2 rows;
K=4 remains an explicit evidence-dropping control.

## Exact anchor and time convention

Use one-based observations `1,...,T`, with `h_t` the complete state returned
by the cell. Let `S_t` denote that cell's transport (including any observation
collision), `M` its spatial convolution, `L` its observation lift, and `A`
its activation. Feature recurrence is

```
p_t = M(S_t(h_(t-1)).features)
h_t.features = A(L(obs_t) + p_t)
```

In [cells.py](../kwm/cells.py), lines **62–67** expose the exact intermediate:
line **65**, `prelift = self.mix(hs)`, is the K=0 frozen tensor. Line **64**
computes the lift; line **66** adds it and applies the activation. This is an
actual forward intermediate, with its autograd graph, not a subtraction from
the post-ReLU state. Call sites are FloWM line **84**, spectral line **100**,
LK line **232**, and lock line **267**. In LK, `feat` is the transported
feature-channel slice; in lock, it is the spectral transport of
`h_prev.features` at line **249**. `frozen` anchors `h_T.features` instead.

K=0 carries the final measured velocity metadata from `h_T`. K>0 carries
metadata from `h_(T-K)` through the zero-input branch (LK's constant spatial
metadata planes can accumulate fp32 averaging roundoff). Its feature anchor is
the result of K applications of `A(M(S_t(features)))`, with the real tail
camera actions, no fresh lift and no node measurements. This branch still
uses the model's recurrence, including its observation collision operator.

Prediction step tau is elapsed time `tau * dt` after the final observation.
Both frozen modes perform exactly one spectral shift **from the same anchor**
at each horizon, with displacement

```
(lattice_i + carried_rho_i) * tau * dt - sum(actions_out[:, :tau])
```

Fixed spectral and FloWM nodes use rho=0; oracle nodes use their supplied true
velocities. Lock uses `LockState.nodes[..., :2]`; LK uses its carried spatial
offset channels. There is no additional order-2/acceleration state in this
branch. No velocity rounding, repeated transport of the previous prediction,
feature mixing, new lift, node measurement, or collision occurs in prediction.
`frozen` decodes the transported features directly. `frozen_prelift` applies
the cell activation (currently ReLU) to each transported feature tensor before
decoding, including when K>0 already activated its anchor.

Both frozen modes ignore `--node_update_pred`. Evaluation JSON records
`node_update_pred: false` and `node_update_pred_ignored: true`. Metadata and
snapshots remain carried unchanged during prediction. Relative readouts still
use the current step's action divided by dt.

## Diagnostics and output names

The main evaluation supplies the cache's `cent`, `half`, `cam`, `vel`, and
`visible` labels as read-only diagnostic inputs. For each predicted frame and
each visible digit, find the node nearest its true world velocity in Euclidean
distance. Sum **squared feature-channel energy** over that digit's periodic
bounding box intersected with the viewport. Divide the selected node's energy
by all nodes' energy over the same pixels. Metadata/imprint channels are
excluded. Labels never modify the predicted state or decoder.

`aux['node_concentration']` has shape `[batch, predicted_time, digit]`. Scores
are in [0,1]; zero-energy visible regions score zero; invisible or empty
regions are NaN and excluded from aggregation. Exact velocity ties choose the
first node. JSON `node_concentration` maps `"1"`, `"10"`, `"20"` to the mean
across eligible digit instances at those exact predicted frames, or null if
none are eligible. Overlapping boxes can share energy, and blank background
within boxes is included. This measures local node specialization rather than
object assignment or forecast accuracy; a perfect single-node oracle scores 1.
Concentration does **not** track MSE: flowm scores below frozen on concentration
while predicting substantially better. Background in boxes, overlaps and old
trails dilute it. It is consistent with, but not diagnostic of, ghosting.

`effective_node_count` uses per-pixel squared feature energies `e_i`, normalizes
`p_i=e_i/sum_j(e_j)`, and computes `exp(-sum_i p_i log(p_i))` **before** spatial
averaging over the viewport. A zero-energy pixel has zero effective nodes;
single-node and uniform-energy pixels have 1 and V, respectively. Auxiliary
values have shape `[batch, 1+predicted_time]`. JSON includes the ordinary final
observation at `"0"`, predicted frames `"1"`, `"10"`, `"20"`, and
`"mean_prediction"` averaged over all pixels, sequences and predicted frames.
The gate also reports the reviewer's distinct **window-pooled** entropy count
(energy is summed spatially before entropy); this is the source of the 23–24.5
effective nodes out of 25, not the per-pixel statistic.

Default evaluation computes both diagnostics on every main rollout. This adds
diagnostic cost and JSON fields, but neither statistic changes predictions.

JSON also records `dec_input` and the effective `frozen_k` (null for other
modes). Append `_dec<mode>` to every evaluation output basename **only when the
mode differs from the saved checkpoint mode**. When a pre-lift K differs from
the default K=0, also append `_fk<K>` to prevent sweeps
from overwriting each other. Existing seed/tag suffixes precede these suffixes.
For example, a saved `flowm` checkpoint evaluated with `--dec_input zero`
writes `eval_deczero.json`; `--dec_input frozen_prelift` writes
`eval_decfrozen_prelift.json`; adding `--frozen_k 4` writes
`eval_decfrozen_prelift_fk4.json`. No override preserves existing basenames.
Legacy `frozen_k` training arguments are ignored: K>0 requires an explicit
evaluation override.

The requested new JSON fields and updated source provenance necessarily change
JSON bytes. Existing numerical metrics, prediction arrays, default figure
contents and filenames retain their default behavior. The training CLI has no
frozen mode or `frozen_k` argument, so newly saved checkpoint arguments contain
neither a frozen policy nor its sweep knob.

## Training and validation

`kwm.train --dec_input` accepts **flowm|zero only**, as before PR3. Both frozen
modes require evaluation overrides; training-mode frozen forwards are rejected.
Evaluation throughput measures frozen inference and records a null training-step
time with `training_skipped`; the normal flowm/zero benchmark is unchanged.
Analytic transport is differentiable, but no frozen training support or training
claim ships with this negative evaluation result.

The fast tests also cover original forward/backward bit equality, saved-mode
and override equivalence, identity-mix first-step transport, removal of the
last lift, carried real velocities, unchanged observation states, prediction
node freezing, oracle concentration, per-pixel entropy, filename rules and
comoving boosts. The stored golden contains eight tiny checkpoint/output pairs
(four model/estimator combinations, each with saved flowm and zero); it does
not rely on a reimplementation sharing the new cell plumbing. Its regeneration
script extracts the base code into `.pytest_cache/` and uses synthetic inputs.

CPU validation with `CUDA_VISIBLE_DEVICES=""`, the `flowm` Python environment,
and at most four torch threads: see [the gate report](GATE_pred_input.md) for
the final focused/full-suite counts and runtimes. The CLI test
also checks byte-identical JSON/PNG outputs for omitted versus explicit
checkpoint-matching mode overrides.

A scene-and-camera boost leaves **ego/viewport predictions unchanged**; it
shifts their world-coordinate interpretation by boost × elapsed scene time.
Tests reuse the existing boost/interior-node helpers and mask matching node
subsets before decoding. Finite-lattice all-node readouts and absolute-velocity
moment readouts need not be exactly equivariant. Estimator equivariance remains
covered separately by PR1 tests.

## Pilot gate and limitations

**NO-GO for frozen prediction.** Pure frozen, pre-lift and de-ghosted controls
fail the joint MSE@20/@150 gate on both lock-10k and LK-13k, using the same
first 16 selection-cache sequences. Zero-input recurrence for 20 steps followed
by freezing is the sole exception on lock-10k; it fails on LK-13k and on the
reviewer's lock-9k/LK-11k checks (32 sequences). This is 1/4 checkpoint/rule
combinations, not a robust prediction policy. The extra variants exist only
inside `scripts/pred_input_gate.py --variants`.

See [GATE_pred_input.md](GATE_pred_input.md) for the measured gate table,
per-variant pass/fail decisions, exact protocol and runtime. The gate compares
each frozen variant to the better of `flowm` and `zero` at both prefix MSE@20
and prefix MSE@150; MSE@50 is reported as additional evidence.

At **tau=1**, transport followed by one learned mix and ReLU matches zero-input
recurrence (MSE 0.013967 lock / 0.013331 LK). The initial frozen gap is the missing
mix step and decoder/state mismatch, not ghost separation. At long horizons,
transport separates content duplicated across velocity nodes. Hard de-ghosting
recovers the lock MSE@150 tail to within roughly 0–4% of the better baseline,
but does not fix MSE@20. Oracle node masking also fails. Recurrence remains
necessary at the trained horizon for these recurrently trained decoders.

These are selection-cache diagnostics, not held-out or multi-training-seed
evidence. The decoders were trained on recurrent states (teacher forcing 0.125),
so these evaluation overrides encounter states absent from training. The result
does not establish how a separately trained frozen model would perform; such
training is outside the plan. Analytic transport is ballistic on a
periodic world: future wall collisions, changing object velocities and new
observations are not modeled. FloWM observation/replay transport retains its
integer action rounding, while its frozen prediction uses real spectral shifts;
fractional observation steps remain unsupported for FloWM. No new dependencies
or checkpoint parameter keys are introduced.

## H3 plan and amendment

This worktree does not contain the review's referenced `PLAN_v4_KWM.md` or its
amendment. The applicable PR3 correction is recorded here rather than editing
an external plan.

**H3 primary: evaluation-time zero < flowm at prefix MSE@20 on K-real with the
lock model**, checkpoint-averaged over 16k–20k, on the held-out test seed, over
five PR1 training seeds, using a paired t-test. MSE@50 and MSE@150 are secondary
with Holm correction; report MSE@150 stability. Across the four pilot
checkpoint/rule combinations, zero beats flowm at @20 in 4/4 and at @150 in
3/4; lock-10k is the @150 exception. These pilot comparisons motivate the
endpoint; they are not the held-out H3 test.

Three trained `--dec_input zero` seeds form a secondary arm, and are the first
cut if the budget tightens. Pre-register: **if trained-zero is not better than
evaluation-time-zero, report evaluation-time zero only.** Freeze-and-render,
de-ghosting and zero-then-freeze are closed as negative results; no frozen
training arm is proposed.
