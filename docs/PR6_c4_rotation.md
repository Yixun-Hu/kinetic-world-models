# PR6: C4 rotation equivariance

Enable with `python -m kwm.train --rot c4`. The checkpoint records `rot`, and
`build_model` and evaluation restore it. Missing `rot` means `none`; the default
channel count remains 32. Both feature channels C and imprint channels K must
be divisible by four. K remains eight. No dependencies were added.

## Action and coordinates

`kwm/rot.py` defines the action used by the model, tests and evaluation. Maps
have shape `[..., H, W]`: x is the column and y is the row. Vectors use
`R(dx,dy)=(-dy,dx)`, so a rightward displacement becomes a downward displacement
on a displayed image. This is a positive mathematical rotation when the y
axis is drawn upward; with array rows drawn downward it is **`torch.rot90(x,-1)`**.
The task's simultaneous requirements of this vector matrix and
`torch.rot90(x,+1)` cannot both hold. The explicit vector matrix and transport
identity determine the sign here.

For a 2×2 window (pivot `(0.5,0.5)`), the worked example is:

```text
input       R input       R² input      R³ input
a b         c a           d c           b d
c d         d b           b a           a c
```

For the actual 32×32 window the pivot is `(15.5,15.5)`, including on the 50×50
world map. Forward positions transform as `(x,y) -> (31-y,x) mod world`.
The inverse gather at destination `(x,y)` reads `(y,31-x) mod world`.
`TorusRotation` precomputes all four integer index permutations as a
nonpersistent buffer; the helper keeps a bounded device/geometry cache.
Window pixels stay inside the observed window and the complement permutes.
This makes writing observations into the top-left corner commute with rotation.
Rotating about the world centre instead would move the observation window.
The permutation works for any **square** grid size, even or odd; a rectangular
torus cannot have this same quarter-turn action without changing its shape.

`rot_vec` and `rot_nodes` act on the final xy axis. `perm_rot[k,i]` satisfies
`lattice[perm_rot[k,i]] == R**k lattice[i]`, with closure and bijection assertions.
To gather transformed state, use `perm_rot[-k]`. Features have C=4F channels in
`f*4+r` order, and rotation sends orientation r to `(r+k)%4`. `rot_state`
transforms space, feature/imprint orientations, node labels, and rho; it carries
evidence/lock bits and rotates the shared LockState snapshot. The LK tensor's
rho and evidence channels remain metadata, outside its regular fields.
`rot_episode` transforms scalar observations, camera actions, ego-frame digit
centres about the pivot modulo the world size (50 by default), and digit velocities. World-frame dataset centres
must first have the camera position subtracted.

## Layers and readout

The p4 weight tying follows the regular-representation construction of
Cohen and Welling (2016). Kernels expand from base parameters on every forward
using `torch.rot90(...,-r)` and orientation rolls, then ordinary `F.conv2d`
with circular padding. There is no interpolation or learned orientation copy.

| Layer | Base parameters | Expanded weight |
| --- | --- | --- |
| Lift | `[Fout,in,k,k]` | output r uses `R**r base` |
| Regular | `[Fout,Fin,4,k,k]` | output r/input r′ uses `R**r base[r′-r]` |
| Scalar output | `[out,Fin,k,k]` | input r′ uses `R**r′ base` |

`conv_u` lifts scalars to regular fields; `conv_h` mixes regular fields;
the decoder is regular convolution, ReLU, scalar-output convolution.
The fixed imprint matrix is `P0 ⊗ I4`, where P0 is the existing seeded
orthogonal construction on `[K/4,C/4]`. Projection therefore commutes with
orientation shifts. P has orthonormal rows when K≤C (columns otherwise).

The moments readout retains regular mass and channelwise node maximum.
The mean vector becomes `[vx,vy,-vx,-vy]`, its projections on
`e_r=R**r(1,0)`. These four channels transform as one regular field.
Spread is invariant and has one shared output weight per field. A regular
1×1 convolution and scalar lift implement the projection. Both branches use
the total fan-in for initialization. Relative readout subtracts the rotated
camera displacement before these moments.

The norm and maximum over channels/nodes are invariant to their permutations;
node softmax, weighted mass, vector mean and scalar spread transform accordingly.
ReLU and componentwise symmetric bounds commute with the signed permutation.
All ordinary parameter and state arithmetic, including FFTs, remains fp32.

## Transport and numerical details

Integer rolls commute with the action bit-for-bit. Fractional streaming needs
one opt-in numerical change. Legacy rfft2 phase transport gives the Nyquist
row and column different sign conventions; on a random 50×50 map a quarter
turn produces about **0.10 normalized error** (a half turn is about 3e-7).
This is not a convolution-weight issue.

C4 uses the product of the two one-dimensional Fourier phases, replacing the
Nyquist phase of **each** even axis with `cos(pi*displacement_axis)`. This is
real tensor-product trigonometric interpolation and treats x/y symmetrically.
It is used in feature/LK streaming, lock snapshot transport, dense search
candidates and frozen-anchor prediction. The default phase implementation is
untouched. Random fractional-shift comparisons fall to about **3e-7**;
integer-shift and odd-size tests also pass. Fractional shifts at Nyquist are
not an invertible representation of continuous translations: successive
cosine attenuations need not equal one cumulative shift. This limitation
already exists for real sampled even-grid interpolation.

A second C4-only change raises the invariant denominator floor in
`velocity_weights` from `1e-6` to `1e-2`. Otherwise near-empty blob tails amplify
fp32 FFT noise into a velocity signal and recurrent feedback can exceed the
requested output tolerance. This affects weak-memory readout in C4 and is a
deliberate deviation from simply substituting tied convolutions. It introduces
no direction dependence; `none` keeps its exact original formula. Regression
cases cover four additional random initializations with weak memory.

One additional C4-only numerical change computes search SSD directly as
`sum_channels((m-current_shifted)**2)`. The legacy norm/dot expansion loses
significant digits on nearly matching patches. On held-out seed-777 sequence
22, this changed a later mode decision and produced a 0.064 px/frame node
offset discrepancy. Direct differences remove that cancellation; candidate
chunks still bound the temporary allocation (about 23 MB for batch four,
25 nodes, seven candidates, eight imprint channels and 32×32 windows).
The standalone estimator harness uses the same C4 arithmetic. `none` retains
its exact norm/dot path. Thresholds, damping and update rules are unchanged.

The central-difference stencil, isotropic 5×5 Gaussian and replicate Gaussian
padding rotate with the observation square. The LK tensor transforms by
`G -> R G Rᵀ`, its right side by R, damping and texture trace are scalar, and
the solved displacement rotates by R. The ±1 search grid at half-pixel steps
is square; separable parabolic refinement exchanges axes. Overlap masks,
boundary exclusion, coherence norms, reachable boxes, covariance splitting,
heavier-mode selection and evidence pooling commute in exact arithmetic.
The principal axis may change sign, exchanging the two modes; the heavier
mode remains the same. Exact equal-mass ties, equal search minima, repeated
covariance eigenvalues and diagnostic argmax ties can select differently.
Random nondegenerate inputs avoid active exact ties. Floating-point changes
near hard thresholds can still change decisions; this is not a uniform error
bound over arbitrary trained parameters, horizons or degenerate scenes.

The C4 checkpoint closure captures the prediction update flag for both LK and
lock, including frozen nodes. The legacy LK behavior remains unchanged under
`none`, including its pre-existing checkpoint/frozen-node limitation.

## Equivariance evidence

`tests/test_rot.py` uses six observations of three moving blobs, random camera
actions, four predictions, random model seed 9, C=8, world=50, window=32,
25 nodes, and relative readout. It tests both adaptive estimators and flowm,
zero and zero-then-frozen inputs (freeze after two predictions). Memory error
is the maximum over features, metadata, snapshot/imprint, offsets and node
velocities, so large accumulated evidence cannot hide a feature error.
Gradients use relative L2 of the concatenated parameter gradients, plus
per-tensor `rtol=1e-5, atol=1e-7` comparisons for near-zero derivatives.
The table takes the worst applicable case across estimators/input policies.

| Mode | k | Memory | Output | Gradient |
| --- | ---: | ---: | ---: | ---: |
| none | 1 | 1.683 | 1.261 | 0.1456 |
| none | 2 | 1.087 | 1.319 | 0.2240 |
| none | 3 | 1.683 | 1.276 | 0.2024 |
| c4 | 1 | 2.67e-7 | 2.01e-7 | 6.71e-7 |
| c4 | 2 | 2.81e-7 | 2.04e-7 | 5.79e-7 |
| c4 | 3 | 2.66e-7 | 1.87e-7 | 8.09e-7 |

Layer tests require normalized error below 1e-6. Integer-map permutations,
inverse actions and window embedding are exact. The unchanged boost acceptance
thresholds are also applied to C4 models in `test_galilean.py` and
`test_lock_nodes.py`. The new C4 relative-readout boost case uses beta=20 to
suppress finite softmax weight on empty edge nodes (the original none case
retains beta=10); its 1e-3 acceptance bound is unchanged. Three pre-change source snapshots in `tests/reference`
check bit-identical initialization, checkpoint keys, forward states, predictions
and gradients for six `none` configurations, including LK, lock, bf16 autocast
and activation checkpointing. Existing prediction golden tests still apply.

`kwm.eval --run RUN --rot_check` reports mean per-sequence normalized L2 as
`rot_equiv_error` using unclamped fp32 predictions, k=1 and the first 32
sequences of a held-out validation cache. The seed is `--test_seed`, or 777
when the main evaluation uses the selection seed. It uses the saved observation
length and all 150 evaluation prediction steps, and honors prediction overrides.
The optional `--only_equiv` runs this diagnostic without the main forecast
metrics and merges its provenance. Output filenames are unchanged.

A separate real-MNIST check (32 seed-777 held-out sequences, C=8, model seed 9,
six observations and four predictions, lock, relative readout, updates on)
measures **6.88e-7** mean rotation error. Before the direct-difference SSD fix
it was 7.03e-4, almost entirely from sequence 22. That sequence and its batch
neighbors are now a fast regression test. Raw short-horizon evidence is in
`results/pr6_c4/heldout_rotation_short.json`. The diagnostic supports the full
150-step evaluation horizon; the 32-sequence numerical measurement here uses
four prediction steps, matching the acceptance horizon.

An additional 150-step stress test on sequence 22 measures **9.02e-4**, above
the requested tolerance. This untrained autoregressive model grows beyond
1e8 around prediction step 95 and reaches roughly 2e13 by step 150; estimator
roundoff and discrete decisions then amplify the discrepancy. Up to roughly
90 predicted steps its output error stays around 2–4e-7. Thus the short-horizon
acceptance criteria pass, but a uniform 1e-5 fp32 guarantee for unstable long
rollouts has **not** been achieved. The complete 32-sequence, 150-step stress
run was stopped; the diagnostic itself still evaluates all 150 steps and
reports the actual discrepancy, including failures. See
`results/pr6_c4/heldout_rotation_long.json`.

## Parameters and CPU cost

Counts include the adaptive scale and moment temperature (both scalars).
LK and lock have the same trainable counts. F is C/4 in C4 mode.

| C | F | none parameters | c4 parameters |
| ---: | ---: | ---: | ---: |
| 32 | 8 | 21,154 | 5,306 |
| 64 | 16 | 83,266 | 20,850 |

At fixed C, tied kernels have approximately one-quarter the free parameters
and the same expanded convolution dimensions. C4 C=64 has approximately the
original C=32 parameter count and four times its dominant convolution work;
this is an operation-count comparison, not a measured fourfold wall time.

CPU timings: Intel Xeon Gold 5220R, PyTorch 2.9.0+cu128, CUDA hidden, one
PyTorch/OMP/OpenBLAS thread, fp32, batch four, 50×50 world, 32×32 window,
six observations plus four predictions, flowm prediction input, node updates
on. Each sample is a complete forward and squared-error backward, with gradients
cleared and no optimizer step. Five measurements follow two warm-up steps.
`scripts/c4_cost.py` writes individual samples and median/mean seconds.

| Estimator | Rotation | Parameters | Median s/step | Mean s/step |
| --- | --- | ---: | ---: | ---: |
| lk | none | 21,154 | 11.300 | 11.365 |
| lk | c4 | 5,306 | 13.157 | 13.056 |
| lock | none | 21,154 | 12.865 | 13.009 |
| lock | c4 | 5,306 | 12.489 | 12.429 |

C4 measured 16.4% higher median cost for LK and 2.9% lower for lock in these runs.
The C4 lock row was remeasured after the direct-SSD fix. These are short
shared-host CPU measurements; the lock difference is smaller
than the spread of individual samples. Raw samples are in
`results/pr6_c4/cost.json`.

## No-learning estimator

The existing standalone estimator gate imports cleanly and now accepts `--rot`.
Seed-0 frozen C=32/K=8 encoders use the first 12 sequences of one seed-2024
300-sequence draw, 30 observations, exact camera actions and no learning.
Hungarian matching and last-ten-frame visibility filtering match
[GATE_estimator.md](GATE_estimator.md); 60 digits are visible.

| Encoder / transport | Matched median (px/frame) | Misses >0.1 |
| --- | ---: | ---: |
| none / legacy | 0.01565608 | 15.0% (9/60) |
| c4 / symmetric Nyquist / direct SSD | 0.02071618 | 18.33% (11/60) |
| c4 / symmetric Nyquist / norm-dot SSD, ablation | 0.02071618 | 18.33% (11/60) |
| c4 / legacy / norm-dot SSD, ablation | 0.02304201 | 16.67% (10/60) |

The C4 estimator **regresses** by 0.00506010 px/frame (**32.3%**) in matched
median and 3.33 percentage points (two digits) in miss rate. With the same C4
encoder and legacy transport/norm-dot SSD, median error is already worse; symmetric Nyquist
improves that median by 0.00232583 but adds one miss. Thus the changed random
encoder/projection accounts for the median regression in this draw. There
are eight independent spatial filters instead of 32 and two independent
imprint fields instead of eight unrelated projections. These change the random
metric used for registration. This is evidence about this seed/slice, not a
causal claim that tying necessarily harms trained or average estimator quality.
No threshold or encoder seed was tuned to erase the regression. The direct-SSD
stability fix leaves the matched median and miss rate unchanged on this slice.

Raw results and per-digit errors are in `results/pr6_c4/estimator_*.json`.
Reproduce each main row with:

```bash
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/estimator_gate.py --n 12 --dataset-size 300 --seed 2024 \
  --batch-size 2 --estimators D_lock_LK_2mode_grid1_heavier_pernode \
  --skip-benchmark --rot none --out results/pr6_c4/estimator_none.json
python scripts/estimator_gate.py --n 12 --dataset-size 300 --seed 2024 \
  --batch-size 2 --estimators D_lock_LK_2mode_grid1_heavier_pernode \
  --skip-benchmark --rot c4 --out results/pr6_c4/estimator_c4.json
python scripts/c4_cost.py
python -m pytest -q
```

Validation used `/home/yixunhu/miniconda3/envs/flowm/bin/python` with
`CUDA_VISIBLE_DEVICES=""`, one OMP/OpenBLAS thread and worktree-local pytest
temporary files. `KWM_SLOW=1 python -m pytest -q` passed **314 tests in
304.42 s**, including both dataset regressions. The final standalone
`tests/test_rot.py` passed **29 tests in 65.90 s**, below the 90 s budget.
Early development test runs used pytest's default `/tmp` scratch directory;
the reported final runs use worktree-local temporary directories.
Both runs emitted the existing `pkg_resources` dependency deprecation warning.
`git diff --check` is clean. The three reference source files match HEAD
byte-for-byte. Per-case equivariance measurements are saved in
`results/pr6_c4/equivariance.json`.

## Scope and limitations

C4 is a model symmetry. The dataset contains upright MNIST digits, so rotating
an episode does not preserve the data distribution. Rotation equivariance
alone says nothing about forecast quality or estimator accuracy after training.
Only quarter turns are supported; continuous rotations and camera yaw are
future work. Existing boosts still have their finite-velocity-lattice and
off-lattice limitations. GPU/TF32/bf16 rotation error was not certified; the
acceptance numbers are CPU fp32.

Collision operators are rejected with `--rot c4`: learned velocity mixing and
learned walls are not tied, and the experimental ordered wall-bounce processing
has not been certified under this action. Oracle models accept vector-rotated
oracle nodes in unchanged object slots; fixed/adaptive lattice models permute
their slots. New C4 checkpoints cannot load ordinary untied weights as C4
weights. Defaults preserve tensors and existing metric filenames; checkpoint
args necessarily gain `rot`, and source-provenance hashes necessarily change.
