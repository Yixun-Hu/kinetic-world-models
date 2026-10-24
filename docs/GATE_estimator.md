# No-learning estimator gate

**Recommend the heavier-mode D2 rule with the ±1.0-pixel grid as the candidate
for integration**, subject to the numerical acceptance issue below. On three
100-sequence datasets, pooled matched error is **0.0238 px/frame** and
miss rate **18.4%**, versus A_LK's **0.0818 / 42.1%**.
The ±1.5 heavier variant is the accuracy winner (**0.0231 / 18.3%**).
Reducing the grid from 49 to 25 candidates costs two misses among 1,500 digits,
while making estimator inference **1.50× faster** on the measured CPU.
Retain ±1.5 as the accuracy comparison in the integration PR.

The implementation now evaluates dense local search for every node, selects LK
or search with `torch.where`, and computes SSD as `|m|² + |c|² - 2 m·c` by a
channel contraction. It never allocates the node × candidate × channel
image-difference tensor. Search construction and evaluation run under
`torch.no_grad()` with autocast disabled; displacements are fp32. Lock diagnostics
stay as tensors throughout the step loop. The boundary fix applies to every
B/C/D search, including the smaller grid.

**Numerical acceptance remains open.** Against the saved pre-revision D2 on the
first 20 sequences of the original seed-2024 dataset, all 30 steps and all 25
nodes were checked. Maximum absolute component difference is **6.9975853e-5**;
final-step maximum is **3.0755997e-5**. This does **not** meet the requested
`<1e-5` tolerance. Dense routing with the old SSD arithmetic was exactly equal
on the first four sequences; the difference is in the norm/dot arithmetic.
Neither fp64 accumulation nor centering consistently fixed it, so those
experimental changes were discarded. The delivered `--parity-reference`
check records the maximum and exits unsuccessfully when the tolerance fails.
Accuracy results below use the delivered fp32 norm/dot implementation; they
are not a claim of numerical parity or readiness to merge a production PR.

## Protocol and metrics

All rows use the same frozen, randomly initialized
`AdaptiveNodeCell(C=32, num_v=25)`, eight imprint channels, initialization seed
0, fixed projection and `s=1`. Estimators see only images and exact camera
actions. Labels are used only for metrics. There is no learning, CUDA use,
autocast or TF32 in the accuracy runs.

There are three datasets of 100 sequences and 30 observed frames, each with
five digits: seed **2024 uses indices 0–99 of the original single 300-sequence
generation call**; seeds **777** and **31337** each use one 100-sequence call.
All use CPU `make_batch(..., EnvConfig(real_velocity=True, real_camera=True),
split='test')`. Generating a new batch of 100 at seed 2024 would give different
sequences. Processing batches contain two sequences and do not affect the
input dataset. All 1,500 digits pass the visibility filter. Same-cell pair
counts are **45 / 60 / 68**, in **39 / 53 / 49** sequences, respectively.

Errors are Euclidean px/frame. `closest` is the nearest-node median. `matched`
is the median after Hungarian assignment on all five digits, followed by
filtering to digits visible at least once in the last ten frames. `miss` is
the matched fraction above 0.1. `lock` is the median first visible step with a
node within 0.1, zero-based and censored at 30; `never` is that censoring
fraction. A first hit need not persist. All medians use the lower middle value,
as `torch.median` does.

`pair` is the fraction of same-lattice-cell digit pairs resolved within 0.1 by
two distinct Hungarian-assigned nodes. `2mode` is the stricter diagnostic in
which that original cell's actual node and its secondary mode must match both
digits. Secondary modes never become extra nodes in the main metrics. A dash
means not applicable. All percentages in the tables are percentages, not
fractions.

`B2` means `B_local_2mode`; `D2` means `D_lock_LK_2mode`. `C124`/`C1248` use
lags `(1,2,4)`/`(1,2,4,8)`. `plain` means one pooled mode. `nearest` is the
original rule; `heavier` chooses the mode with larger pooled mass (left wins
an exact mass tie). The other mode is secondary. All searches use a 0.5-pixel
spacing. The default radius is ±1.5 pixels; `grid1` means ±1.0 pixels.

## Three-seed accuracy

Seed 2024, original first 100:

variant / rule | closest | matched | miss | lock | never | pair | 2mode
--- | --- | --- | --- | --- | --- | --- | ---
A_LK | 0.0774 | 0.0781 | 40.6% | 5 | 21.2% | 17.8% | —
B2 / nearest | 0.0479 | 0.0479 | 27.6% | 6 | 18.8% | 26.7% | 8.9%
B2 / heavier | 0.0419 | 0.0425 | 23.4% | 5 | 14.6% | 31.1% | 8.9%
C124 / plain | 0.0476 | 0.0485 | 30.2% | 5 | 19.4% | 17.8% | —
C124 / nearest | 0.0374 | 0.0374 | 28.0% | 5 | 22.4% | 37.8% | 4.4%
C124 / heavier | 0.0251 | 0.0251 | 20.0% | 4 | 14.2% | 31.1% | 2.2%
C1248 / plain | 0.0417 | 0.0424 | 30.4% | 5 | 23.8% | 22.2% | —
C1248 / nearest | 0.0359 | 0.0359 | 30.2% | 5 | 26.2% | 33.3% | 2.2%
C1248 / heavier | 0.0241 | 0.0241 | 23.4% | 4 | 18.6% | 28.9% | 2.2%
D2 / nearest | 0.0263 | 0.0263 | 22.8% | 6 | 18.8% | 33.3% | 15.6%
D2 / heavier | 0.0215 | 0.0215 | 18.0% | 4 | 11.6% | 31.1% | 6.7%
D2 grid1 / nearest | 0.0266 | 0.0266 | 25.6% | 6 | 21.6% | 28.9% | 15.6%
D2 grid1 / heavier | 0.0228 | 0.0228 | 18.8% | 4 | 11.6% | 33.3% | 6.7%

Seed 777, 100 sequences:

variant / rule | closest | matched | miss | lock | never | pair | 2mode
--- | --- | --- | --- | --- | --- | --- | ---
A_LK | 0.0891 | 0.0910 | 44.8% | 5 | 25.0% | 18.3% | —
B2 / nearest | 0.0532 | 0.0540 | 30.8% | 7 | 23.2% | 28.3% | 11.7%
B2 / heavier | 0.0492 | 0.0504 | 27.8% | 5 | 17.4% | 28.3% | 5.0%
C124 / plain | 0.0559 | 0.0559 | 30.6% | 4 | 21.2% | 28.3% | —
C124 / nearest | 0.0385 | 0.0390 | 25.8% | 5 | 20.8% | 28.3% | 10.0%
C124 / heavier | 0.0316 | 0.0321 | 21.6% | 4 | 15.6% | 30.0% | 6.7%
C1248 / plain | 0.0448 | 0.0448 | 31.2% | 4 | 25.2% | 30.0% | —
C1248 / nearest | 0.0350 | 0.0355 | 29.8% | 5 | 25.0% | 33.3% | 8.3%
C1248 / heavier | 0.0256 | 0.0258 | 21.2% | 4 | 17.6% | 38.3% | 6.7%
D2 / nearest | 0.0353 | 0.0358 | 25.4% | 6 | 21.8% | 33.3% | 10.0%
D2 / heavier | 0.0245 | 0.0245 | 19.4% | 4 | 11.4% | 28.3% | 5.0%
D2 grid1 / nearest | 0.0385 | 0.0386 | 28.6% | 7 | 23.2% | 30.0% | 10.0%
D2 grid1 / heavier | 0.0241 | 0.0241 | 20.0% | 3 | 11.8% | 28.3% | 5.0%

Seed 31337, third fresh set of 100 sequences:

variant / rule | closest | matched | miss | lock | never | pair | 2mode
--- | --- | --- | --- | --- | --- | --- | ---
A_LK | 0.0769 | 0.0799 | 41.0% | 5 | 20.6% | 17.6% | —
B2 / nearest | 0.0485 | 0.0488 | 27.8% | 7 | 22.0% | 38.2% | 8.8%
B2 / heavier | 0.0427 | 0.0434 | 25.8% | 4 | 14.6% | 26.5% | 4.4%
C124 / plain | 0.0525 | 0.0540 | 32.8% | 4 | 19.8% | 16.2% | —
C124 / nearest | 0.0344 | 0.0346 | 27.4% | 5 | 23.0% | 26.5% | 10.3%
C124 / heavier | 0.0293 | 0.0293 | 23.2% | 3 | 15.2% | 27.9% | 7.4%
C1248 / plain | 0.0460 | 0.0473 | 31.4% | 4 | 23.0% | 20.6% | —
C1248 / nearest | 0.0316 | 0.0319 | 31.6% | 5 | 26.6% | 20.6% | 8.8%
C1248 / heavier | 0.0233 | 0.0241 | 23.8% | 3 | 18.0% | 26.5% | 5.9%
D2 / nearest | 0.0308 | 0.0317 | 26.6% | 6 | 20.8% | 29.4% | 13.2%
D2 / heavier | 0.0231 | 0.0233 | 17.4% | 3 | 8.4% | 39.7% | 4.4%
D2 grid1 / nearest | 0.0297 | 0.0301 | 27.2% | 7 | 21.0% | 33.8% | 14.7%
D2 grid1 / heavier | 0.0236 | 0.0242 | 16.4% | 4 | 7.8% | 44.1% | 4.4%

Heavier beats nearest in both matched median and miss rate for B and D on
every seed. For D on the third seed alone, ±1.5 heavier gives **0.0233 / 17.4%**,
nearest gives **0.0317 / 26.6%**, and A_LK gives **0.0799 / 41.0%**.

Pooled D2 results show the grid tradeoff:

rule / grid | closest | matched | miss | lock | never | pair | 2mode
--- | --- | --- | --- | --- | --- | --- | ---
nearest / ±1.5 | 0.0308 | 0.0311 | 24.9% | 6 | 20.5% | 31.8% | 12.7%
heavier / ±1.5 | 0.0229 | 0.0231 | 18.3% | 4 | 10.5% | 33.5% | 5.2%
nearest / ±1.0 | 0.0311 | 0.0314 | 27.1% | 7 | 21.9% | 31.2% | 13.3%
heavier / ±1.0 | 0.0235 | 0.0238 | 18.4% | 4 | 10.4% | 35.8% | 5.2%

The smaller grid is competitive with the heavier rule; its nearest-rule miss
rate is worse. Heavier improves primary-node recovery but reduces the stricter
secondary-mode diagnostic. The speed/accuracy tradeoff motivates ±1.0; ±1.5
remains the measured accuracy reference. No equivalence margin was tested.

## Corrected multi-lag conclusion

An edge minimum has no two-sided parabolic refinement. Previously, a clamped
1.5-pixel displacement could pass the 0.6-pixel/frame size gate at lag 4 or 8
and become a spurious velocity measurement. `usable` now requires both
candidate coordinates to be strictly inside the grid, in addition to valid
cost and overlap. This rule also applies to B and D, where the lag-1 size gate
already excluded ±1.5-pixel minima.

The old numerical comparison is withdrawn because it included boundary-clamped
pseudo-measurements. The corrected results still show a pooled miss-rate
increase from lag 8 for both two-mode rules: **27.1% → 30.5%** for nearest and
**21.6% → 22.8%** for heavier. However, median error improves for every rule,
plain C's miss rate slightly falls, and heavier C's misses fall on seed 777
(21.6% → 21.2%). Thus the blanket claim that lag 8 raises misses is retracted;
the qualified two-mode pooled tradeoff remains supported. D2 heavier retains
the best pooled matched median and miss rate.

All six C variants were rerun on all three datasets. Pooled over 300 sequences:

rule | C124 matched / miss | C1248 matched / miss
--- | --- | ---
plain | 0.0537 / 31.2% | 0.0447 / 31.0%
nearest | 0.0378 / 27.1% | 0.0345 / 30.5%
heavier | 0.0290 / 21.6% | 0.0244 / 22.8%

## Paired three-seed intervals

comparison against A_LK | median reduction (95% CI), px/frame | miss reduction (95% CI), percentage points
--- | --- | ---
D2 heavier / ±1.5 (best accuracy) | 0.0587 (0.0528–0.0660) | 23.87 (21.27–26.53)
D2 heavier / ±1.0 (recommended grid) | 0.0580 (0.0522–0.0652) | 23.73 (21.07–26.47)

The paired bootstrap resamples whole sequences from the pooled 300 sequences,
with 2,000 replicates and RNG seed 2024. Estimator and A_LK errors use the same
indices and visibility masks. Intervals are descriptive 95% percentile
intervals, unadjusted for choosing among variants. Dataset medians were not
averaged.

## CPU cost

CPU operation, batch 16 | seconds/step
--- | ---
Production AdaptiveNodeCell inference | 1.6550
Snapshot-LK estimator only | 0.4096
D2 nearest / ±1.5, estimator only | 1.6832
D2 heavier / ±1.5, estimator only | 1.6773
D2 nearest / ±1.0, estimator only | 1.1341
D2 heavier / ±1.0, estimator only | 1.1165

Timing uses batch 16, one PyTorch/OpenBLAS/OMP thread, nine warmup steps and
11 measured steps (frames 9–19); each entry is their median wall-clock time.
The production row measures the complete existing `AdaptiveNodeCell` inference
step, including encoding, feature transport/mixing and imprint/state updates.
Estimator rows start with preencoded imprints and include snapshot handling,
transport, displacement estimation, mode pooling and node updates, but exclude
encoding and the feature recurrence. They are not whole-model or training
step times, and adding one to the production row would double-count existing
estimator work. Timing runs are sequential after the accuracy jobs finish.
Hardware: Intel Xeon Gold 5220R, 2.20 GHz; PyTorch 2.9.0+cu128. No GPU timings
were attempted because the GPU was reserved.

## Prediction-phase proxy

estimator | inputs at measurement | closest | matched | miss
--- | --- | --- | --- | ---
A_LK | 30 true frames | 0.0797 | 0.0797 | 40.0%
A_LK | 50 true frames (control) | 0.0735 | 0.0735 | 40.0%
A_LK | 30 true + 20 predicted | 0.0947 | 0.0947 | 48.0%
D2 heavier / ±1.5 | 30 true frames | 0.0265 | 0.0265 | 17.5%
D2 heavier / ±1.5 | 50 true frames (control) | 0.0221 | 0.0221 | 14.5%
D2 heavier / ±1.5 | 30 true + 20 predicted | 0.0369 | 0.0369 | 21.0%
D2 heavier / ±1.0 | 30 true frames | 0.0260 | 0.0260 | 18.0%
D2 heavier / ±1.0 | 50 true frames (control) | 0.0228 | 0.0229 | 15.0%
D2 heavier / ±1.0 | 30 true + 20 predicted | 0.0377 | 0.0377 | 21.0%

The proxy uses 40 sequences from generator seed 778 with 30 observed and 20
future frames. The generator model is loaded on CPU with
`kwm.model.build_model(checkpoint['args'])`, then its saved weights, from
`adaptive_bs16_ckpt/best.pt` in the supplied 200-step benchmark directory.
`node_update_pred=True` is explicit. All gate variants retain the frozen
seed-0 imprint and update nodes on all 50 inputs. They receive the same 20
self-predictions from that checkpoint; D does not change the generator's
predictions. A 50-true-frame control uses the same sequences and actions.
The visibility filter for the 50-step result uses frames 40–49.

The shared prediction MSE is **0.119518**, versus true-frame
mean squared intensity **0.154766**. Checkpoint SHA-256:
`b1156def12dd0a3f1664f0aaeb38b5cbf6cc3abfda2ef6629bc054fa007fef03`.

This measures sensitivity to predicted inputs, not the forecast quality of a
D-integrated model. End-to-end retraining and comparisons with prediction-phase
node updates on/off remain unmeasured.

## Estimator settings

- A replays the production LK overwrite/0.9-decay imprint from per-sample
  snapshots and past velocities, including sequential spectral shifts and
  Nyquist effects. The snapshot-LK cost control uses only the preceding
  snapshot. Replay cost is not the production cell cost.
- B transports a snapshot by `nu - camera_action`, searches shared spectral
  shifts of the current imprint, and aggregates SSD with a separable Gaussian
  (sigma 2, radius 6), normalized by observed overlap. Masks use bilinear
  periodic sampling. Interior minima receive axis-wise parabolic refinement.
- C transports lag-L snapshots by `nu*L - camera_delta`. The search radius is
  in pixels, not pixels/frame. Divide the residual by L, then fit velocity by
  displacement WLS with weights `evidence*L²` and two Cauchy reweightings
  (scale 0.15). Snapshots are per sample, never persistent per-node images.
- D computes snapshot LK first. A node is locked when evidence/textured mass
  is at least 0.1 and pooled LK residual norm is at most 0.3. Dense search is
  evaluated for all nodes, then `torch.where` selects LK at locked nodes and
  search at unlocked nodes. Locking is reevaluated every step.
- Two-mode variants restrict candidate velocities to the node's reachable
  lattice box, cluster along the displacement covariance's principal axis
  for five iterations, and accept a split only with at least 10% mass in each
  mode and separation at least 0.1 px/frame. D clusters the selected field,
  including LK at locked nodes. C robustly fits selected primary and secondary
  modes across lags. Secondary evidence is accumulated only diagnostically.
- All use the original mass, texture, residual-velocity size gate 0.6 and
  unwarped agreement weighting, evidence running mean, componentwise step
  clip 0.3 and lattice-offset bound 0.75. No learned estimator parameters were
  added. The old window-correlation controls remain available but were not
  part of this revision's rerun matrix.

## PR requirements

- Keep search under `torch.no_grad()`.
- Carry velocities, accumulated evidence and snapshots in the recurrent state
  in fp32, not mutable Python attributes. This standalone inference harness
  still owns its state; it is not a checkpoint-safe production cell interface.
- Maintain an fp32 island under bf16 autocast for transport/estimation,
  velocities, evidence and snapshot updates.
- Avoid host synchronizations inside steps: no `.item()`, `.tolist()`, Python
  conversions of device scalars, or data-dependent `nonzero` routing.
- Only locked nodes carry a displacement gradient through LK; the searched
  displacement is detached. The learned scale `s` must still train on both
  paths. Keep weighting/selection decisions outside the displacement gradient.

Resolve the numerical-parity failure above before claiming the requested
implementation acceptance. Measure GPU step time when the reserved GPU is
available and validate an end-to-end trained integration, including
prediction-phase updates on/off; this CPU estimator gate does not supply
those results.

## Reproduction and artifacts

Use `/home/yixunhu/miniconda3/envs/flowm/bin/python` with
`CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. The script
hides CUDA by default and never overwrites an explicit visibility setting.
Only `scripts/estimator_gate.py` and `docs/GATE_estimator.md` were changed;
`kwm/`, `tests/` and the pre-existing `results/gate_estimator.json` were left
untouched. No commit was created.

Accuracy runs (one output per dataset):

```bash
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=/home/yixunhu/miniconda3/envs/flowm/bin/python
VARIANTS=A_LK,B_local_2mode,B_local_2mode_heavier,C,D_lock_LK_2mode,D_lock_LK_2mode_heavier,D_lock_LK_2mode_grid1,D_lock_LK_2mode_grid1_heavier
"$PY" scripts/estimator_gate.py --n 100 --dataset-size 300 --seed 2024 --batch-size 2 --estimators "$VARIANTS" --skip-benchmark --out /tmp/gate2024.json
"$PY" scripts/estimator_gate.py --n 100 --seed 777 --batch-size 2 --estimators "$VARIANTS" --skip-benchmark --out /tmp/gate777.json
"$PY" scripts/estimator_gate.py --n 100 --seed 31337 --batch-size 2 --estimators "$VARIANTS" --skip-benchmark --out /tmp/gate31337.json
"$PY" scripts/estimator_gate.py --pool-results /tmp/gate2024.json /tmp/gate777.json /tmp/gate31337.json --out /tmp/gate_pooled.json
"$PY" scripts/estimator_gate.py --cost-only --n 16 --bench-steps 11 --out /tmp/gate_costs.json
```

The measured accuracy run used separate 50-sequence shards (`--n 50`,
`--start-index 0` or `50`, with unchanged `--dataset-size`) in independent CPU
processes. `--pool-results` rejects overlapping shards, mismatched dataset
hashes, different specs and unpaired sequence indices.

Parity and prediction commands (the parity command currently exits nonzero):

```bash
git show 44cb8ed71718f99c8aa2bef980063d62f5fed45d:scripts/estimator_gate.py > /tmp/gate_previous.py
"$PY" scripts/estimator_gate.py --n 20 --dataset-size 300 --batch-size 2 --parity-reference /tmp/gate_previous.py --out /tmp/gate_parity.json
CKPT=/tmp/claude-1016/-home-yixunhu-codespace-flowm/831f1992-2b3c-47ed-ba4f-3bd336e0abb9/scratchpad/kwm_bench/runs/adaptive_bs16_ckpt
"$PY" scripts/estimator_gate.py --prediction-checkpoint "$CKPT" --prediction-cache /tmp/gate_predictions.pt --prediction-n 40 --prediction-seed 778 --batch-size 2 --estimators A_LK,D_lock_LK_2mode_heavier,D_lock_LK_2mode_grid1_heavier --out /tmp/gate_prediction.json
"$PY" scripts/estimator_gate.py --n 2 --estimators D_lock_LK_2mode_heavier,D_lock_LK_2mode_grid1_heavier --self-check --skip-benchmark --out /tmp/gate_check.json
```

The revision's raw shards, complete per-sequence errors, dataset hashes,
pooled paired bootstrap, timings, parity check and prediction proxy outputs
are stored locally in `/tmp/estimator_gate_revision/`. The consolidated
`results.json` and `costs.json` are the sources for these tables. The original
300-sequence results file is historical; its C rows contain the old boundary
bug and are superseded by the corrected tables above.

The expanded `--self-check` passes production-LK trajectory parity, translation
and camera signs, lag units, robust-fit units, both mode rules, locked LK-mode
routing, rejection of boundary minima on both grids, and no-grad/fp32 search
under bf16 autocast. The separate 20-sequence old-D2 comparison fails the
stricter requested tolerance, as reported above.
