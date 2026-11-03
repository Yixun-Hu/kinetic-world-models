# PR6: C4 rotation equivariance

Enable with `python -m kwm.train --rot c4`. The checkpoint records `rot`, and
`build_model` and evaluation restore it. Missing `rot` means `none`; the default
channel count remains 32. Both feature channels C and imprint channels K must
be divisible by four in `c4` mode. K remains eight. No dependencies were added.
The algebra is C4-equivariant; fp32 lock rollouts are **equivariant up to the
lock estimator's rounding sensitivity**, measured with the one-ulp control below.

## Action and coordinates

`kwm/rot.py` defines the action used by the model, tests and evaluation.
In the transport's (x = column, y = row) convention, R(dx,dy) = (−dy, dx),
equal to `torch.rot90(x, -k, dims=(-2,-1))` on square observation maps for Rᵏ.
This appears clockwise when rows increase downward; “counter-clockwise”
assumes a y-up display.

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
symmetric Nyquist factors. Legacy rfft2 phase transport gives the Nyquist
row and column different sign conventions, breaking quarter-turn symmetry
for fractional displacements on even grids independently of convolution weights.

C4 uses the product of the two one-dimensional Fourier phases, replacing the
Nyquist phase of **each** even axis with `cos(pi*displacement_axis)`. This is
real tensor-product trigonometric interpolation and treats x/y symmetrically.
It is used in feature/LK streaming, lock snapshot transport, dense search
candidates and frozen-anchor prediction. The default phase implementation is
untouched. Fractional-shift, integer-shift and odd-size transport tests pass.
Fractional shifts at Nyquist are not an invertible representation of continuous translations: successive
cosine attenuations need not equal one cumulative shift. This limitation
already exists for real sampled even-grid interpolation.

A second numerical change raises the invariant denominator floor in
`velocity_weights` from `1e-6` to `1e-2`. Otherwise near-empty blob tails amplify
fp32 FFT noise into a velocity signal. This affects weak-memory readout and is a
deliberate deviation from simply substituting tied convolutions. It introduces
no direction dependence; `none` keeps its exact original formula. Regression
cases cover four additional random initializations with weak memory.

The third numerical change computes search SSD directly as
`sum_channels((m-current_shifted)**2)`. The legacy norm/dot expansion loses
significant digits on nearly matching patches and can change later mode
decisions. Direct differences remove that cancellation; candidate
chunks still bound the temporary allocation (about 23 MB for batch four,
25 nodes, seven candidates, eight imprint channels and 32×32 windows).
The standalone estimator harness uses the same arithmetic. `none` retains
its exact norm/dot path. Thresholds, damping and update rules are unchanged.

The internal control `--rot none_c4num` uses plain untied convolution/readout
layers and the original dense imprint P, together with all three C4 numerical
changes: symmetric Nyquist factors, the `1e-2` velocity-weight floor, and
direct-difference SSD. It has the same parameter shapes and initialization as
`none`; the saved `rot` argument restores the numerical choice on load. The
internal cell attribute `c4_numerics=False` disables the three changes together
for the parity test. The tests also compare its readout floor, fractional
transport, and lock update with C4 on shared constructed features/imprints.
Use this arm to distinguish tying from numerics in the GPU campaign. The
absolute `1e-2` floor is scale-dependent, not a learned or universal tolerance.

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

The checkpoint closure captures the prediction update flag for both LK and
lock in **every rotation mode**, including frozen nodes. This fixes the known
upstream issue #3: `adaptive` + `lk` + `grad_ckpt` + `node_update_pred off`
previously recomputed prediction steps with updates enabled during backward.
This deliberately changes `none` LK gradients only in that experimental
configuration. The pre-rotation parity test excludes exactly its backward
comparison while still checking initialization, forward predictions and states;
a separate test requires checkpointed and ordinary gradients to be bit-identical
in that configuration for all three rotation modes. Other `none` initialization,
forward, state and gradient comparisons remain bit-identical.

## Equivariance evidence

`tests/test_rot.py` covers all three quarter turns, both adaptive estimators,
flowm/zero/zero-then-frozen prediction inputs, layer actions, memory fields,
node relabelling and gradients. The small nondegenerate blob cases retain their
`1e-5` regression bound; this is not a guarantee for arbitrary lock rollouts.
Layer tests use `1e-6`. Integer-map permutations, inverse actions and window
embedding are exact. The boost helper uses the model's numerical transport,
including symmetric Nyquist factors for `c4` and `none_c4num`.
The relative-readout boost test uses beta=20 for **both** none and C4:
at beta=10 finite softmax weight on empty edge nodes makes the `1e-3` bound
marginal in either mode (none seed 1 also exceeds it).

The three unchanged pre-rotation source snapshots in `tests/reference` check
initialization, checkpoint keys, forward states, predictions and gradients
for `none`, including LK, lock, bf16 autocast and activation checkpointing,
with prediction updates both on and off. Only the backward configuration
identified above is excluded. Existing prediction golden tests still apply.
The unused model `perm_rot` buffer has been removed; `rot_state` computes the
permutations it actually uses.

### Reading `--rot_check`

`kwm.eval --run RUN --rot_check` checks at most 32 held-out sequences in fp32,
using unclamped predictions and **k=1,2,3** (`--n_val` can lower the 32-sequence
budget, and `--bs` can lower the default diagnostic batch size of four).
For each sequence and horizon h,
the error is `||expected[:h] - actual[:h]||₂ / max(||expected[:h]||₂, 1e-12)`.
It reports the mean and maximum over sequences at horizons **1, 4, 20 and the
full prediction horizon** (150 for normal evaluation). Horizons longer than
the supplied rollout are omitted and duplicate horizons appear once.

The JSON schema is:

```text
rot_equiv_error: {k: {horizon: {mean, max}}}
ulp_baseline:    {horizon: {mean, max}}
```

JSON object keys are strings. The baseline uses the same unrotated rollout,
then advances every observed pixel by one float32 ulp toward +infinity with
`torch.nextafter(obs.float(), +inf)`. Zeros advance to the smallest positive
subnormal; camera actions and oracle nodes are unchanged. The metric and
horizons are identical to the rotation check. Autocast, CUDA matmul TF32 and
cuDNN TF32 are disabled inside the check; both backend flags are restored,
including if evaluation raises. The output provenance records these settings.

The held-out seed is `--test_seed`, or 777 when the main evaluation uses the
selection seed. Saved observation length and prediction overrides are honored.
`--only_equiv` merges both diagnostic objects and their provenance into an
existing result. Output filenames are unchanged.

Lock amplifies a one-ulp input perturbation into roughly `1e-5` node-velocity
changes after one measurement, with later threshold/mode decisions reaching
`1e-3`–`1e-1`; this also happens with `--rot none`. Rotation reorders fp32
reductions, so its residual cannot be interpreted below this sensitivity.
The floor is empirical and scene-dependent, not a universal bound or a proof
from one chosen perturbation. In particular, a trained C4 lock model at
150 steps can show rotation error around `1e-3` **with an ulp baseline around
`1e-3`** and still implement exact C4 equivariance in exact arithmetic.

### Repeated measurements with the ulp control

All measurements below use CPU fp32, no TF32, and report normalized prediction
error as **mean / maximum over sequences**. Each rotation error is paired with
its baseline at the same horizon. Raw objects include all three k values and
all available diagnostic horizons.

Untrained C4 lock, C=8, model seed 9, dataset seed 777: 32 sequences,
six observations plus four predictions, batch size four. Raw data: [JSON](../results/pr6_c4/heldout_rotation_short.json).

| Horizon | k=1: mean / max | k=2: mean / max | k=3: mean / max | One ulp: mean / max |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8.877e-07 / 6.489e-06 | 8.191e-07 / 4.380e-06 | 1.058e-06 / 7.942e-06 | 9.162e-07 / 8.999e-06 |
| 4 | 6.878e-07 / 4.565e-06 | 6.547e-07 / 3.323e-06 | 3.075e-05 / 9.616e-04 | 3.067e-05 / 9.618e-04 |

The same untrained model, sequence 22 of the 32-sequence seed-777 draw:
six observations plus 150 predictions, batch size one. There is one sequence,
so each mean equals its maximum. Raw data: [JSON](../results/pr6_c4/heldout_rotation_long.json).

| Horizon | k=1: mean / max | k=2: mean / max | k=3: mean / max | One ulp: mean / max |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.273e-06 / 2.273e-06 | 3.923e-06 / 3.923e-06 | 6.304e-06 / 6.304e-06 | 8.995e-06 / 8.995e-06 |
| 4 | 1.224e-06 / 1.224e-06 | 2.036e-06 / 2.036e-06 | 9.616e-04 / 9.616e-04 | 9.618e-04 / 9.618e-04 |
| 20 | 3.526e-07 / 3.526e-07 | 2.567e-07 / 2.567e-07 | 4.169e-03 / 4.169e-03 | 4.264e-03 / 4.264e-03 |
| 150 | 9.020e-04 / 9.020e-04 | 1.947e-02 / 1.947e-02 | 4.203e-02 / 4.203e-02 | 2.394e-02 / 2.394e-02 |

The former 150-step `9.02e-4` figure is superseded by the paired long-horizon
measurement above. It reflects rounding amplified by lock on an **untrained**
model. Its extreme output growth occurs for one C=8 initialization (seed 9),
not C4 seeds 0–3 or C=32 in the review, and is unchanged by reverting the
Nyquist or floor choices. The trained `pilot_lock` model in the review does not
show that growth. This is a seed-specific initialization artefact, not evidence of a
C4-induced long-rollout instability.

Trained `pilot_lock/last.pt` (read-only), step 20,000, C=32, `rot=none`,
30 observations plus 150 predictions, eight held-out seed-777 sequences,
batch size four. The full CLI invocation took **1164.16 seconds** on CPU.
Raw output: [JSON](../results/pr6_c4/eval_pilot_lock_last_rot_check.json). Checkpoint SHA-256:
`a3bff48fef60981739f68a5651fafad581fb72e3a15007ea4587d9102de406da` (unchanged after evaluation).

| Horizon | k=1: mean / max | k=2: mean / max | k=3: mean / max | One ulp: mean / max |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.826e-01 / 2.468e-01 | 1.407e-01 / 2.060e-01 | 1.986e-01 / 2.709e-01 | 5.920e-07 / 8.923e-07 |
| 4 | 1.866e-01 / 2.529e-01 | 1.495e-01 / 2.031e-01 | 2.010e-01 / 2.626e-01 | 6.336e-07 / 9.860e-07 |
| 20 | 2.653e-01 / 3.717e-01 | 2.203e-01 / 3.147e-01 | 2.691e-01 / 3.737e-01 | 1.073e-06 / 1.802e-06 |
| 150 | 5.261e-01 / 5.928e-01 | 4.863e-01 / 5.496e-01 | 4.805e-01 / 6.074e-01 | 3.056e-03 / 1.891e-02 |

Reproduce from this worktree:

```bash
mkdir -p .pytest_cache/tmp .pytest_cache/xdg .pytest_cache/mpl
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/.pytest_cache/tmp"
export XDG_CACHE_HOME="$PWD/.pytest_cache/xdg" MPLCONFIGDIR="$PWD/.pytest_cache/mpl"
/home/yixunhu/miniconda3/envs/flowm/bin/python -m kwm.eval \
  --run /home/yixunhu/codespace/kinetic-world-models/runs/pilot_lock \
  --ckpt last --device cpu --n_val 8 --bs 4 --rot_check --only_equiv \
  --skip_ft --skip_tp --out results/pr6_c4/ --tag pilot_lock_last_rot_check
```

The pilot checkpoint is an untied
`rot=none` model, so its rotation residual is a negative control, not a C4
acceptance measurement.

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

The standalone estimator gate accepts `--rot` and `--encoder-seed`.
Frozen C=32/K=8 encoders use the first 12 sequences of the **same seed-2024
300-sequence draw**, 30 observations, exact camera actions and no learning.
Hungarian matching and last-ten-frame visibility filtering match
[GATE_estimator.md](GATE_estimator.md); 60 digits are visible. Each table entry
is **matched median (px/frame) / miss rate above 0.1 px/frame**.

| Encoder seed | none C32 K8: median / miss | c4 C32 K8: median / miss |
| --- | ---: | ---: |
| 0 | 0.01566 / 15.0% | 0.02072 / 18.3% |
| 1 | 0.01695 / 16.7% | 0.01735 / 18.3% |
| 2 | 0.02086 / 20.0% | 0.02372 / 23.3% |
| 3 | 0.02151 / 18.3% | 0.01668 / 15.0% |
| Mean over encoder seeds | 0.01874 / 17.50% | 0.01962 / 18.75% |

**No regression is demonstrated at this sample size.** The distributions over
encoder seeds overlap: none spans about 0.0157–0.0215 and C4 0.0167–0.0237.
The mean difference is about 0.0009 px/frame (5%) and 1.25 percentage points
in miss rate, within the encoder-seed spread (sample standard deviations
0.0029 for none and 0.0033 for C4, in px/frame). A single seed-0 comparison
had selected the best none draw and cannot establish a tying penalty. This 12-sequence slice is **not a
model-selection tool** for the GPU campaign.

The review's C=64 and K=32 controls do not change that conclusion. The gate
only sees the linear map `P · conv_u(frame)`; C4 has K/4 independent spatial
imprint filters with four rotations each, versus K independent filters in
none. Increasing C alone leaves that effective imprint diversity unchanged;
increasing K did not resolve a difference on this slice either. The evidence
supports random-metric draw-to-draw variance, not an inherent tying regression.

Raw per-digit results are in `results/pr6_c4/estimator_<mode>_seed<s>.json`.
All eight JSON files are complete and share dataset SHA-256
`09fafa77ba078b7562bfd2c0851987cd657ba0c4df611ed1949a144ab0b49024`.
Recorded total runtimes are 35.4–41.5 seconds per run, below the
ten CPU-minute budget. Reproduce:

```bash
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for mode in none c4; do
  for seed in 0 1 2 3; do
    python scripts/estimator_gate.py --n 12 --dataset-size 300 --seed 2024 \
      --encoder-seed "$seed" --batch-size 2 \
      --estimators D_lock_LK_2mode_grid1_heavier_pernode --skip-benchmark \
      --rot "$mode" --out "results/pr6_c4/estimator_${mode}_seed${seed}.json"
  done
done
python -m pytest -q
KWM_SLOW=1 python -m pytest -q
```

Fresh continuation validation used
`/home/yixunhu/miniconda3/envs/flowm/bin/python`, with CUDA hidden and
OMP/OpenBLAS/MKL limited to one thread. Temporary files, pytest basetemps,
logs and outputs stayed in this worktree.

| Command | Result | Pytest runtime | Process wall time |
| --- | --- | ---: | ---: |
| `python -m pytest -q` | 332 passed, 2 slow tests skipped | 206.85 s | 219.24 s |
| `KWM_SLOW=1 python -m pytest -q` | 334 passed | 322.32 s | 333.79 s |

Both runs include all three cases in `tests/test_rotation_check.py`, the
untied-control parity checks, and the checkpoint/ordinary gradient comparisons.
The two warnings are dependency deprecations (`pkg_resources` and the requested
PyTorch TF32 flag API). Logs are `.pytest_cache/pr6-continued-default.log` and
`.pytest_cache/pr6-continued-slow.log`; each invocation used a distinct
worktree-local `--basetemp`.

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
weights. Apart from the explicitly corrected LK checkpoint backward configuration,
defaults preserve tensors and existing metric filenames; checkpoint
args necessarily gain `rot`, and source-provenance hashes necessarily change.
