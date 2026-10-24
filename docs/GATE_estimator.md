# No-learning estimator evaluation

The integrated lock-then-refine rule uses per-node texture thresholds and LK
damping, direct camera actions, a ±1-pixel search grid, and the heavier of two
supported velocity modes. It is available in the standalone harness as
`D_lock_LK_2mode_grid1_heavier_pernode`; `A_LK` is the unchanged LK baseline.
See [the method](PR1_lock_nodes.md) for the recurrence, state and gradients.
The harness retains older variants for ablation and historical reproduction;
it is an optional offline analysis tool, not a production or test dependency.

## Protocol

Every row uses a frozen, randomly initialized `AdaptiveNodeCell(C=32, num_v=25)`,
eight imprint channels, initialization seed 0, fixed projection and `s=1`.
Estimators see only observations and exact camera actions. Labels enter only
metrics. Accuracy runs use CPU fp32, no learning, autocast or TF32.

Each dataset contains 100 sequences, 30 observed frames and five digits per
sequence. Seed **2024 uses indices 0–99 of a single 300-sequence generation**;
seeds **777** and **31337** each use one 100-sequence generation. Generating
only 100 at seed 2024 changes the data. All use
`EnvConfig(real_velocity=True, real_camera=True)`, split `test`, and processing
batches of two. All 1,500 digits pass the visibility filter. Same-cell pair
counts are 45 / 60 / 68 in 39 / 53 / 49 sequences, respectively.

Errors are Euclidean px/frame. Hungarian assignment matches all five digits to
distinct nodes before filtering digits visible at least once in the last ten
frames. `matched` is the lower median (`torch.median`); `miss` is the matched
fraction above 0.1. `closest` uses the nearest node without uniqueness. `first
hit` is the first visible step with a node within 0.1, zero-based and censored
at 30; `never` is the censoring fraction. A first hit need not persist.
`pair` measures same-lattice-cell digit pairs recovered within 0.1 by distinct
assigned nodes. Secondary modes never become extra nodes in these metrics.

## Final-rule results

| Seed | LK matched median | LK misses | Final lock matched median | Final lock misses |
| --- | ---: | ---: | ---: | ---: |
| 2024 | 0.078062 | 40.6% | 0.022695 | 18.4% |
| 777 | 0.091020 | 44.8% | 0.024010 | 19.8% |
| 31337 | 0.079873 | 41.0% | 0.023973 | 16.8% |
| Pooled | 0.081799 | 42.1% | 0.023746 | 18.3% |

| Seed | Rule | Closest median | First hit | Never | Pairs resolved |
| --- | --- | ---: | ---: | ---: | ---: |
| 2024 | LK | 0.077446 | 5 | 21.2% | 17.8% |
| 2024 | Final lock | 0.022695 | 4 | 11.4% | 33.3% |
| 777 | LK | 0.089146 | 5 | 25.0% | 18.3% |
| 777 | Final lock | 0.024010 | 3 | 11.8% | 28.3% |
| 31337 | LK | 0.076931 | 5 | 20.6% | 17.6% |
| 31337 | Final lock | 0.023586 | 4 | 7.8% | 42.6% |

Raw results, including per-sequence errors, assignments, final nodes, dataset
hashes, source hashes and paired bootstrap intervals:

- [Seed 2024](../results/lock_nodes_gate/seed2024.json)
- [Seed 777](../results/lock_nodes_gate/seed777.json)
- [Seed 31337](../results/lock_nodes_gate/seed31337.json)
- [Pooled comparison](../results/lock_nodes_gate/pooled.json)

Across 1,500 visible digits, the final rule reduces matched median error by
**0.058054 px/frame** (95% paired interval
0.052236–0.065310) and misses by **23.80
percentage points** (95% interval 21.07–26.53).

Paired bootstrap intervals resample whole sequences, using the same indices and
visibility masks for each estimator and LK (2,000 replicates, seed 2024).
Intervals are descriptive 95% percentile intervals, unadjusted for selection
among estimator variants. Dataset medians are not averaged.

## Accuracy regression tests

Two opt-in tests evaluate the first **12** sequences of the seed-2024 and
seed-777 draws above, through the production cell. They retain the full draw
size before slicing and process batches of two. The local test helper implements
rectangular Hungarian assignment with shortest augmenting paths, without
importing the standalone harness or SciPy. It matches all digits first, then
applies the same visibility filter and lower median.

| Seed | Matched median reference | Miss-rate reference |
| --- | ---: | ---: |
| 2024 | 0.015656080097 | 0.150000005960 |
| 777 | 0.023351069540 | 0.183333337307 |

Each metric has a **one-sided** tolerance: fail only when it exceeds its
final-rule reference by more than 10% relative. Improvements pass. The former
bit-parity test against the historical cross-node rule is retired. Full
100-sequence evidence remains in the table above, independently of the smaller
regression samples. `pytest -m slow` selects these two cases;
`KWM_SLOW=1 pytest -q` includes them in the full suite.

## Historical ablations and numerical tolerance

The following measurements use the earlier cross-node texture/damping and
cumulative-camera arithmetic; they are not the final-rule baseline.

| Rule, pooled over three seeds | Matched median | Miss rate |
| --- | ---: | ---: |
| LK | 0.0818 | 42.1% |
| Lock, heavier mode, ±1.0 grid | 0.0238 | 18.4% |
| Lock, heavier mode, ±1.5 grid | 0.0231 | 18.3% |
| Lock, mode nearest the current estimate, ±1.0 grid | 0.0314 | 27.1% |
| Lock, mode nearest the current estimate, ±1.5 grid | 0.0311 | 24.9% |

The smaller grid traded two additional misses among 1,500 digits for a 1.50×
CPU estimator speedup. Historical batch-16, single-thread medians were 1.1165
s/step for the heavier-mode ±1.0 grid and 1.6773 s/step for ±1.5, after nine
warmup steps and eleven measured steps on an Intel Xeon Gold 5220R. These exclude
feature recurrence and training. Whole-model GPU costs appear in the method doc.

All searches exclude minima on either grid boundary: an edge minimum has no
two-sided refinement, and at long lags a clamped pixel displacement can otherwise
pass the velocity-size gate. With this exclusion, adding lag 8 to `(1,2,4)`
improved pooled median error but increased misses for the two-mode variants:
nearest-mode 27.1% to 30.5%, heavier-mode 21.6% to 22.8%. The single-mode miss
rate fell slightly, 31.2% to 31.0%; the effect is not universal.

Switching SSD from explicit channel differences to norm/dot arithmetic produced
a maximum component difference of 6.9975853e-5 over 20 historical seed-2024
sequences (3.0755997e-5 at the final step). This is accepted at **1e-4** for that
arithmetic comparison; bit parity and 1e-5 agreement are not expected. The final
per-node rule intentionally changes estimator decisions. Separately, sharing
snapshot FFTs and caching masks must preserve final-rule node trajectories to
**1e-6**; that optimization has its own regression check.

A historical prediction-input proxy used 40 seed-778 sequences and a 200-step
LK checkpoint (SHA-256
`b1156def12dd0a3f1664f0aaeb38b5cbf6cc3abfda2ef6629bc054fa007fef03`).
After 30 true and 20 shared predicted frames, LK measured 0.0947 / 48.0% and the
earlier ±1 lock rule 0.0377 / 21.0%. Prediction MSE was 0.119518. The estimator
did not change the generator's predictions; this proxy is not an end-to-end
forecast comparison. The final-rule training comparison remains pending.

## Reproduction

Install the repository requirements plus **SciPy**, which is used only by the
optional standalone harness's assignment metrics. Run from the repository root:

```bash
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
VARIANTS=A_LK,D_lock_LK_2mode_grid1_heavier_pernode
python scripts/estimator_gate.py --n 100 --dataset-size 300 --seed 2024 --batch-size 2 --estimators "$VARIANTS" --skip-benchmark --out results/lock_nodes_gate/seed2024.json
python scripts/estimator_gate.py --n 100 --seed 777 --batch-size 2 --estimators "$VARIANTS" --skip-benchmark --out results/lock_nodes_gate/seed777.json
python scripts/estimator_gate.py --n 100 --seed 31337 --batch-size 2 --estimators "$VARIANTS" --skip-benchmark --out results/lock_nodes_gate/seed31337.json
python scripts/estimator_gate.py --pool-results results/lock_nodes_gate/seed2024.json results/lock_nodes_gate/seed777.json results/lock_nodes_gate/seed31337.json --out results/lock_nodes_gate/pooled.json
```

The harness stores inference history in its own estimator object. Production
uses explicit recurrent state with fp32 estimation under autocast, detached
search and mode selection, locked-LK displacement gradients, and no host scalar
conversions. The standalone historical parity utility defaults to 1e-4 for the
norm/dot comparison; it does not compare the intentionally revised final rule.
