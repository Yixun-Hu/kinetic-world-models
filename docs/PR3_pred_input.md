# Prediction-phase input policies

`kwm.eval --dec_input {flowm,zero,frozen,frozen_prelift}` overrides only the
prediction policy. Omitting it uses the checkpoint's saved policy, with `flowm`
as the fallback for old checkpoints. The normal observation pass, parameter
keys, default recurrence arithmetic, decoder and existing metric reductions
are unchanged. Default forward values and training gradients are tested with
exact equality against the previous recurrence/rollout.

| Policy | Prediction dynamics |
| --- | --- |
| `flowm` | Feed the last observation, then each previous prediction, through the recurrent cell. |
| `zero` | Feed zeros through the same recurrence; retain the configured prediction node-update policy. |
| `frozen` | Freeze the final post-lift, post-activation feature memory and its velocity metadata; decode independent analytic transports. This is the negative control. |
| `frozen_prelift`, K=0 | Freeze the last observation's transported-and-mixed recurrent content, before adding its lift and before activation. Activate after each analytic transport. |
| `frozen_prelift`, K>0 | Take the state after observation T−K; replay the last K observation actions with zero inputs and node measurements disabled, then freeze. Activate after each analytic transport. |

`--frozen_k K` defaults to 1 for new models and old checkpoints; evaluation
preserves a saved K unless explicitly overridden. Require `0 <= K < T_obs`.
K only affects `frozen_prelift`. The ordinary observation pass still processes
all frames: the K-step replay is a separate branch, and observation-state and
node-estimator diagnostics retain the ordinary pass. K>0 deliberately excludes
the last K observation lifts and measurements from the prediction anchor.

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

JSON also records `dec_input` and the effective `frozen_k` (null for other
modes). Append `_dec<mode>` to every evaluation output basename **only when the
mode differs from the saved checkpoint mode**. When a pre-lift K differs from
the saved K (old-checkpoint default 1), also append `_fk<K>` to prevent sweeps
from overwriting each other. Existing seed/tag suffixes precede these suffixes.
For example, a saved `flowm` checkpoint evaluated with `--dec_input zero`
writes `eval_deczero.json`; `--dec_input frozen_prelift --frozen_k 0` writes
`eval_decfrozen_prelift_fk0.json`. No override preserves existing basenames.

The requested new JSON fields and updated source provenance necessarily change
JSON bytes. Existing numerical metrics, prediction arrays, default figure
contents and filenames retain their default behavior. New training arguments
also appear in newly saved checkpoint argument dictionaries.

## Training and validation

`kwm.train --dec_input frozen_prelift [--frozen_k K]` is supported. Feature
anchors, convolution, analytic transport, activation and decoder remain in the
gradient graph. Tests verify finite nonzero gradients to observation features,
encoder/mixing weights and camera displacements, and matching checkpointed vs
ordinary backward gradients, for all four model variants and K=0/2. Frozen
training ignores teacher forcing because it does not consume prediction-phase
frames. The naive `frozen` control is intentionally absent from the training
CLI. The existing `flowm|zero` defaults are unchanged.

The fast tests also cover original forward/backward bit equality, saved-mode
and override equivalence, identity-mix first-step transport, removal of the
last lift, carried real velocities, unchanged observation states, prediction
node freezing, oracle concentration, filename rules and comoving boosts.

CPU validation with `CUDA_VISIBLE_DEVICES=""`, the `flowm` Python environment,
and one OpenBLAS/OMP thread: **73 new tests passed in 7.14 s**;
the full suite had **230 passed, 2 existing slow tests skipped in 70.41 s**.
Both runs reported the existing dependency deprecation warning. The CLI test
also checks byte-identical JSON/PNG outputs for omitted versus explicit
checkpoint-matching mode overrides.

A scene-and-camera boost leaves **ego/viewport predictions unchanged**; it
shifts their world-coordinate interpretation by boost × elapsed scene time.
Tests reuse the existing boost/interior-node helpers and mask matching node
subsets before decoding. Finite-lattice all-node readouts and absolute-velocity
moment readouts need not be exactly equivariant. Estimator equivariance remains
covered separately by PR1 tests.

## Pilot gate and limitations

**No frozen variant passes on either pilot (0/10 comparisons).** The first-16
selection-cache sweep used the requested `last.pt` files at lock step 10000
and LK step 13000, which had advanced beyond the reviewer's checkpoints.
All variants lose to the required baseline at both gated horizons. The modes
remain opt-in experiments; the default policy is unchanged.

See [GATE_pred_input.md](GATE_pred_input.md) for the measured gate table,
per-variant pass/fail decisions, exact protocol and runtime. The gate compares
each frozen variant to the better of `flowm` and `zero` at both prefix MSE@20
and prefix MSE@150; MSE@50 is reported as additional evidence.

These are selection-cache pilot checkpoint diagnostics, not held-out or
multi-training-seed evidence. The naive post-lift anchor duplicates the last
frame across all velocity nodes; analytic transport separates those copies
and produces ghosts. Removing one lift or adding a short zero-input tail does
not guarantee velocity specialization. Analytic transport is ballistic on a
periodic world: future wall collisions, changing object velocities and new
observations are not modeled. FloWM observation/replay transport retains its
integer action rounding, while its frozen prediction uses real spectral shifts;
fractional observation steps remain unsupported for FloWM. No new dependencies
or checkpoint parameter keys are introduced.
