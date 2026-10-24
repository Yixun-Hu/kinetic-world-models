# PR1: lock-then-refine adaptive nodes

`--node_update lock` opts into the delivered gate's **D2 heavier, ±1.0 px**
estimator. `--node_update lk` remains the default. Existing model parameters,
initialization, LK arithmetic, checkpoint weight keys, and CLI defaults are
preserved. No commit was made.

## Implementation map

| Location | Change |
| --- | --- |
| `kwm/train.py:134` | CLI choice `lk,lock`, default `lk`. |
| `kwm/model.py:61`, `kwm/model.py:313` | Constructor and `build_model` option, including old-checkpoint fallback. |
| `kwm/model.py:104`, `kwm/model.py:152`, `kwm/model.py:255` | Explicit state initialization, node readout, decoding and saved-state handling. |
| `kwm/model.py:220` | Capture the prediction-update flag in each checkpoint closure. |
| `kwm/cells.py:136`, `kwm/cells.py:230` | State allocation and functional lock recurrence. |
| `kwm/lock_nodes.py:13` | `LockState` layout. |
| `kwm/lock_nodes.py:117` | Dense local SSD search and subpixel refinement. |
| `kwm/lock_nodes.py:65`, `kwm/lock_nodes.py:185` | Heavier-mode pooling, lock routing and displacement gradients. |
| `kwm/cells.py:155`, `kwm/equiv.py:76` | Bounce and boost support for compact node state and shared snapshot. |
| `tests/test_lock_nodes.py:29` | Acceptance tests, including a frozen pre-change cell in `tests/reference/adaptive_cell_pre_lock.py`. |

The frozen reference is copied from the pre-change cell at
`9aa69dbe5a45c612232a3bae3cbb67003fe417e5`; imports are qualified for standalone
test loading. Production code does not import the gate script.

## Estimator and lock rule

At each enabled update, form the current imprint `P(conv_u(frame))`. Pad the
previous observed-window snapshot to the world, then spectrally transport it
by `nu * dt - camera_delta`, separately for each node. The persistent snapshot
is shared; these transported views are transient. The feature recurrence
continues to use its existing spectral transport and convolution.

Compute the existing LK displacement and trace. Apply the existing imprint
mass, texture threshold (0.1 of the per-sample mean trace), residual-size gate
(0.6 px/frame), and unwarped coherence weight, multiplied by observed overlap.
The lock is **reevaluated each step**:

- `agreement = pooled_LK_evidence / max(textured_imprint_mass, 1e-6)`;
- lock iff `agreement >= 0.1` and `norm(pooled_LK_residual / dt) <= 0.3`.

The stored lock bit records this decision. It is not sticky and there is no
hysteresis. A node can unlock on the next step.

Search is dense for every node. The 25 candidates are `[-1, -.5, 0, .5, 1]^2`.
Shift the current snapshot spectrally, sharing candidates over nodes. Compute
SSD by `|m|² + |c|² - 2 m·c`, contracting imprint channels without allocating
node × candidate × channel differences. Candidate chunks contain at most
seven shifts. Aggregate with a separable Gaussian (sigma 2, radius 6), divide
by aggregated observed overlap, and reject overlap below 0.5. Choose a
per-pixel minimum and refine both axes with the three-point parabola. Minima
on either candidate-grid boundary are unusable; local snapshot support must
also exceed 0.99.

Select LK at locked nodes and search elsewhere with `torch.where`. Apply the
same coherence/texture/size gates to the selected field, restrict residual
velocities to the node's reachable lattice box (offset bound 0.75), and fit
two modes along the weighted covariance's principal axis for five iterations.
A split needs at least 10% mass in each mode and separation at least
0.1 px/frame. Choose the **heavier** mode, with the left mode winning ties;
otherwise use the unsplit mean. Secondary modes do not create new nodes.

Accumulate selected evidence `W`, apply gain `evidence / max(W, 1e-6)`, multiply
the residual velocity by the existing learned scalar `n_s`, clip each update
component to ±0.3, and bound each lattice offset to ±0.75. No estimator
parameters were added.

## State, precision and gradients

The LK state remains the original `[B,V,C+3+K,H,W]` tensor, with `state_extra=11`
at K=8. Lock uses a named tuple, initialized with `cell.initial_state(frame)`:

| Field | Shape | Precision / meaning |
| --- | --- | --- |
| `features` | `[B,V,C,H,W]` | Existing feature dtype, including bf16 under autocast. |
| `nodes` | `[B,V,4]` | fp32: `rho_x, rho_y, accumulated_evidence, lock_bit`. |
| `snapshot` | `[B,K,win,win]` | One fp32 observation imprint per sample. |
| `camera` | `[B,2]` | fp32 cumulative action at the snapshot. |

For lock, `cell.extra=4` and `model.state_extra=4` describe compact node
metadata, not spatial feature channels. The snapshot and camera are separate
state tensors. At K=8, win=32, V=25, the estimator state is **33,176 bytes per
sample** (32,768 snapshot + 400 nodes + 8 camera), excluding features and
backward intermediates. A batch of 16 uses 530,816 bytes for this state.
Neither snapshot nor recurrent node data is stored on the Python module.
`last_nodes` remains unused in lock mode; trajectory diagnostics read returned
state. Bounce and comoving boost helpers transform node content/metadata and
preserve the shared observation snapshot and camera reference.

Transport/estimation and state updates have autocast explicitly disabled.
Imprints, offsets, evidence, lock bits and camera state stay fp32. Search,
weights, clustering and mode decisions run under `torch.no_grad()`. After
selection, a weighted sum of the **locked LK field** reattaches just its
existing displacement derivative, with a zero-valued gradient correction so
that forward pooling is unchanged. Searched displacements have zero gradient;
`n_s` trains on both routes. The previous snapshot retains its graph for LK.
The implementation uses fixed-shape dense routing, with no tensor-to-Python
scalar conversions or data-dependent indexing in the estimator step.

With prediction updates off, offsets, evidence, lock, snapshot and camera are
frozen; features continue evolving. With updates on, the existing model input
policy applies (self-predictions, or teacher-forced targets during training).
The checkpoint closure captures this choice before forward restores the
legacy flag, so backward recomputation uses the same phase.

## CPU validation

Environment: `/home/yixunhu/miniconda3/envs/flowm/bin/python`, PyTorch
2.9.0+cu128, `CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.
The pre-change suite passed **69/69** in **26.93 s**.

Final full-suite result: **84 passed, 1 existing dependency deprecation warning,
455.60 s (7 min 35 s)**: all 69 original tests plus 15 new cases. Full command:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/yixunhu/miniconda3/envs/flowm/bin/python -m pytest -q -s
```

Raw output and numeric results are saved as `acceptance.log` and
`acceptance.json` in the timing output root below.

| Check | Result |
| --- | --- |
| Default / explicit LK vs frozen pre-change cell | Bitwise equal outputs, node trajectories, and parameter gradients on a fixed batch, including frozen prediction steps. |
| Delivered D2 H ±1.0 parity, 20 sequences | **0 maximum absolute difference**, both across all 30 steps and at the final step, all 25 nodes and both components. |
| Seed-777 accuracy, 100 sequences | **0.0241064988 px/frame matched median, 20.0% misses** (500 visible digits). |
| Checkpoint forward/backward | Updates on/off both pass; loss tolerance 1e-5, gradients also compared. |
| Functional state replay | Interleaving another sequence does not affect replay; input state is unchanged. |
| CPU bf16 | Runs with fp32 nodes, snapshot and camera; features bf16; backward works. |
| Gradient routing | Nonzero LK displacement gradient when locked; zero when searched; nonzero scale gradient on both routes. |
| Node permutation | Features and node state agree after permutation within fp32 tolerance; shared snapshot identical. |
| CPU profiler | No `aten::item` or `aten::_local_scalar_dense` in the lock step. |
| Heavier mode / boundary | Chooses the larger mode; grid-edge minima are rejected. |
| Bounce / boost state helpers | Reflect node content and offsets; preserve the shared snapshot. |

Parity uses seed 2024, the **first 20 of a single 300-sequence generation**,
not a fresh draw of 20. Accuracy uses one 100-sequence draw with seed 777.
Both use `EnvConfig(real_velocity=True, real_camera=True)`, split `test`,
30 observed frames, cell initialization seed 0, C=32, K=8, `n_s=1`, and
processing batches of two. The metrics are the gate's Hungarian matching,
last-ten-frame visibility filter, lower median, and miss threshold >0.1.

Seed-777 closest median is also 0.0241064988; median first hit is step 3,
never-hit fraction 11.8%, and 17/60 same-cell pairs are resolved (28.3333%)
across 53 sequences containing such pairs. Relative deviation from the rounded
gate references is **0.0270%** for median error and effectively **0%** for
miss rate, both within the requested 10%.

The 20-sequence seed-2024 parity subset has matched/closest median
0.0193365104, miss rate 17.0%, median first hit step 4, never-hit fraction 11.0%,
and 1/9 same-cell pairs resolved (11.1111%, in eight sequences). These subset
metrics are not compared against the gate's full 100-sequence seed-2024 row.

The parity implementation preserves the gate's fp32 arithmetic explicitly:
`camera_new = camera_old + action`, followed by subtraction to obtain the
snapshot displacement, and its velocity-to-residual update ordering. Using
`action` directly first produced **0.0016284585 px/frame** maximum difference:
small roundoff can change a discrete two-mode decision. Carrying the two
camera coordinates removed that discrepancy completely. This is parity with
the **delivered norm/dot gate**; it does not resolve or relabel the older
pre-revision gate's separately documented 1e-5 numerical acceptance issue.

Synthetic Galilean tests reuse `tests/test_galilean.py`'s blobs, dimensions,
velocities and boosts, now with observation-driven updates:

| Boost | LK feature error | Lock feature error |
| --- | ---: | ---: |
| `(1,0)` | 7.9942438e-8 | 5.2747605e-8 |
| `(-1,1)` | 8.7047511e-8 | 7.5468876e-8 |

For off-lattice `(0.27,-0.41)`, the locked-node velocity-shift maximum component
error is **0.0016585290 px/frame**, below 0.05.

A broader diagnostic is not a universal improvement: the existing random
four-sequence finite-lattice evaluation (C=4, seed 777, t=6) measured
**0.00432979 lock vs 0.00207601 LK** normalized feature error. Global LK damping
and texture means depend on which velocities remain inside the finite lattice;
lock and mode decisions can amplify those changes. Exact gate parity retains
these rules. The synthetic acceptance checks pass, but a blanket claim that
lock improves boost error on arbitrary scenes is not supported.

## Gate evidence and remaining comparison

These are previously measured no-learning gate results from
[`GATE_estimator.md`](GATE_estimator.md), not end-to-end training results:

| Estimator | Pooled matched median, px/frame | Miss rate |
| --- | ---: | ---: |
| Existing LK | 0.0818 | 42.1% |
| D2 heavier ±1.0 (implemented) | 0.0238 | 18.4% |
| D2 heavier ±1.5 (accuracy comparison) | 0.0231 | 18.3% |

The gate's seed-777 ±1.0 reference is 0.0241 / 20.0%. The historical batch-16
CPU estimator-only median was 1.1165 s/step for ±1.0 versus 1.6773 s/step for
±1.5; these exclude the feature recurrence and are not GPU training costs.

## GPU timing

Both requested GPU runs were attempted and failed before training. GPU probes report no exposed `/dev/nvidia*` devices,
`torch.cuda.is_available() == False`, device count 0, and
`RuntimeError: No CUDA GPUs are available` with `CUDA_VISIBLE_DEVICES=1`.
No GPU timing or peak-memory number can be inferred from the CPU tests.

| Mode | Completed training steps | GPU seconds/step | Peak allocated memory | Exit code |
| --- | ---: | --- | --- | ---: |
| lk | 0 | unavailable | unavailable | 1 |
| lock | 0 | unavailable | unavailable | 1 |

Each requested output directory contains `attempt.log` (complete error and
arguments) and `attempt.json` (status `blocked_no_cuda_device`, null timing and
memory). The saved `attempt_timing.py` driver runs the two modes sequentially,
restricts visibility to GPU 1, explicitly selects CUDA, and would cap PyTorch's
allocator at 14,000,000,000 bytes if the device were available. No GPU was used;
the requested <15 GB peak is therefore **unvalidated**, not a measured claim.

Requested output root:
`/tmp/claude-1016/-home-yixunhu-codespace-flowm/831f1992-2b3c-47ed-ba4f-3bd336e0abb9/scratchpad/pr1_timing/{lk,lock}`.

Training arguments (the explicit device prevents silent CPU fallback; use the
saved `attempt_timing.py` driver to include its allocator cap):

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1
PY=/home/yixunhu/miniconda3/envs/flowm/bin/python
OUT=/tmp/claude-1016/-home-yixunhu-codespace-flowm/831f1992-2b3c-47ed-ba4f-3bd336e0abb9/scratchpad/pr1_timing
for mode in lk lock; do
  "$PY" -m kwm.train --model adaptive --real_velocity --real_camera \
    --bs 16 --accum 2 --steps 60 --n_val 32 --val_n_quick 32 \
    --val150_n 0 --val_every 1000 --log_every 20 \
    --node_update "$mode" --device cuda --out "$OUT/$mode"
done
```

GPU time/memory validation and an end-to-end training comparison remain.
The latter should compare LK and lock forecast quality, with prediction-phase
updates both on and off, matched seeds and budgets. The 60-step timing run,
even when available, is not sufficient evidence of a forecasting improvement.
