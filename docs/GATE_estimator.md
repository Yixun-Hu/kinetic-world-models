# No-learning estimator gate

**Gate passed. Recommend `D_lock_LK_2mode` for the next node-update PR:**
local search at lag 1 when unlocked, LK when locked, and two-mode pooling of
whichever displacement field is active. Use the settings below unchanged:
agreement threshold 0.1, lock residual norm 0.3, step clip 0.3, offset bound
0.75, residual-size gate 0.6, minimum mode mass 10%, and minimum mode separation
0.1 px/frame. Keep per-sample snapshots and `s=1`; no learned estimator weights
are needed.

On **all 300 sequences**, matched median error falls from **0.0821 to 0.0294
px/frame**, an absolute reduction of 0.0527 (**64.2%**). Closest-node median falls
from 0.0816 to 0.0293. Matched errors above 0.1 fall from **42.4% to 25.2%**
(**17.2 percentage points**, 258 fewer misses among 1,500 digits). Same-cell
pairs resolved by distinct nodes rise from **28/133 to 41/133** (21.1% to 30.8%).
A paired, sequence-level bootstrap with 2,000 resamples gives descriptive 95%
intervals of **0.0480–0.0587 px/frame** for the median reduction and
**14.3–19.9 percentage points** for the miss-rate reduction. These intervals
are not adjusted for selecting among estimators.

The tradeoffs are real: median first lock is step **6 versus 5**, although
never-locked digits fall from 23.3% to 20.1%. CPU time is **2.646 s/step at
batch 16**, about **4.5×** the equally snapshot-based LK control (0.584 s).
The original LK row includes snapshot replay overhead, as explained below.
Same-cell separation remains incomplete: even the winner's own-cell primary
and secondary modes resolve only **21/133 pairs (15.8%)**. This gate establishes
no-learning node accuracy; end-to-end forecasting and CUDA throughput remain
unmeasured.

Two-mode B is the runner-up (0.0492 median, 28.8% misses). Plain D also beats LK
(0.0677, 34.8%); plain B's gain is smaller (0.0765, 40.3%). C's two-mode variants
beat LK's median but have more misses and substantially higher cost than
the winner. Adding lag 8 raises their miss rate from 31.9% to 36.6%. Removing
LK's step clip does not help. The window-level correlation controls remain
poor (76–91% misses). Snapshot-only LK is slightly worse than original LK,
so changing image storage alone does not explain the winner's improvement.

Every estimator shares an identical frozen, randomly initialized
`AdaptiveNodeCell(C=32, num_v=25)` imprint (`K=8`, initialization seed 0, fixed
projection, `s=1`). There is no training or access to digit labels in any
estimator other than the explicitly labelled oracle.

The dataset is one CPU `make_batch(300, 30, 0,
EnvConfig(real_velocity=True, real_camera=True), split='test',
gen=torch.Generator().manual_seed(2024))` call. It contains five digits per
sequence, no walls, exact camera actions, and real velocities in `[-2,2]^2`.
Generating once before splitting into processing batches makes the sequences
independent of estimator selection and batch size. All arithmetic is fp32
(FFT coefficients are complex64); autocast and TF32 are off. No GPU was available
in the execution environment, so reported measurements are CPU measurements.
All 1,500 digits pass the final-ten-frame visibility filter. There are 133
same-cell pairs in 124/300 sequences (41.3%). A separate binary Hungarian
reachability check finds 9/1,500 digits (0.6%) cannot simultaneously receive
distinct nodes under the ±0.75 bound, in nine sequences. This is a geometric
ceiling, independent of image-based estimation.

estimator | n | closest | matched | >.1 | lock* | never | pair | 2mode | s/step@16
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
fixed_lattice | 300 | 0.4061 | 0.4183 | 97.1% | 30 | 97.1% | 0.0% | — | —
oracle_25 | 300 | 0.0000 | 0.0000 | 0.0% | 0 | 0.0% | 100.0% | — | —
A_LK | 300 | 0.0816 | 0.0821 | 42.4% | 5 | 23.3% | 21.1% | — | 1.5769
A_LK_no_clip | 300 | 0.0823 | 0.0829 | 42.6% | 5 | 23.0% | 21.1% | — | 1.4942
A_LK_snapshot | 300 | 0.0863 | 0.0868 | 44.3% | 5 | 23.3% | 23.3% | — | 0.5839
B_local | 300 | 0.0764 | 0.0765 | 40.3% | 5 | 20.5% | 18.8% | — | 2.2772
B_local_2mode | 300 | 0.0491 | 0.0492 | 28.8% | 6 | 20.9% | 26.3% | 9.0% | 2.4706
C_lags124 | 300 | 0.0768 | 0.0768 | 43.4% | 5 | 28.2% | 14.3% | — | 6.3416
C_lags124_2mode | 300 | 0.0428 | 0.0434 | 31.9% | 5 | 26.4% | 27.1% | 7.5% | 6.2288
C_lags1248 | 300 | 0.0765 | 0.0768 | 44.6% | 5 | 33.1% | 15.8% | — | 7.5523
C_lags1248_2mode | 300 | 0.0408 | 0.0417 | 36.6% | 5 | 31.5% | 21.8% | 4.5% | 7.4590
D_lock_LK | 300 | 0.0677 | 0.0677 | 34.8% | 5 | 17.8% | 24.1% | — | 2.8180
D_lock_LK_2mode | 300 | 0.0293 | 0.0294 | 25.2% | 6 | 20.1% | 30.8% | 15.8% | 2.6461
E_window_nearest124 | 300 | 0.1795 | 0.1805 | 76.1% | 16 | 40.7% | 3.8% | — | 0.5253
E_window_strongest124 | 300 | 0.1912 | 0.1929 | 77.9% | 16 | 40.3% | 3.0% | — | 0.4318
E_window_nearest1 | 300 | 0.2909 | 0.2935 | 90.6% | 30 | 71.5% | 2.3% | — | 0.1647

Errors are Euclidean px/frame. `closest` and `matched` are medians across digits
visible at least once in frames 20–29. Matching uses SciPy's Hungarian assignment
on **all five digits**, then filters visibility, exactly as `pr1_sim.py` does.
`>.1` is the fraction of those digits whose matched error exceeds 0.1.
`lock*` is the median first **visible** step at which any node is within 0.1 of
that digit, with zero-based steps 0–29 and never-locked digits censored to 30.
`never` reports that censoring fraction. Locking is a first hit, not a guarantee
that the node stays locked or that every digit has a distinct node at that step.

`pair` counts final same-Voronoi-cell digit pairs for which **both distinct
Hungarian-assigned nodes** have error at most 0.1. Both digits must satisfy the
same visibility filter. `2mode` is a separate, stricter local diagnostic: can
the original lattice cell's actual node and its reported secondary mode match
both digits, in either order, within 0.1? Secondary modes never become extra
nodes in the main error or pair metrics. JSON includes their velocities and
validity masks. A dash means not applicable (or no qualifying pairs).

The `s/step@16` column is the median of three timed steps, frames 9–11, following
nine warmup frames at batch 16 on the same dataset. It includes observation
encoding, snapshot handling, transport, search and the node update; it excludes
data generation, transfers, metrics and the unrelated recurrent feature
convolution. Fixed/oracle references have no timed update. These are wall-clock
measurements on a shared CPU host, not GPU throughput estimates.

The estimators use the following fixed settings:

- **A, LK:** original `lk_displacement`, damping, texture gate, displacement
  gate 0.6, agreement gate, evidence running mean, componentwise step clip 0.3,
  and componentwise lattice-offset bound 0.75. `A_LK_no_clip` removes only the
  step clip. To respect the prohibition on a persistent per-node imprint bank,
  A reconstructs the original overwrite/0.9-decay imprint recurrence from
  per-sample snapshots and past node velocities. Sequential spectral shifts
  are replayed exactly, including overwrite masks and Nyquist effects. Replay
  is included in A's timing; that timing is **not** a claim about the cost of
  the production cell with its existing cached imprint.
- **A snapshot control:** identical LK weighting and update, but uses only the
  immediately previous snapshot and its overlap support. This separates changes
  in the displacement estimator from changes in observation memory.
- **B, local SSD:** transport the previous snapshot by `nu - camera_action`;
  compare each transported pixel with the new imprint at 49 residual offsets
  in `[-1.5,1.5]^2`, spaced by 0.5 px. Fractional candidates use the repository's
  spectral shift. SSD sums the eight imprint channels, then uses a separable
  Gaussian with sigma 2, radius 6, normalized by observed overlap. A per-pixel
  minimum gets axis-wise three-point parabolic refinement. Pool using KWM's
  imprint mass, texture threshold, residual-size gate and **unshifted**
  agreement gate, then use the same running mean, clip and bound as A.
  Overlap masks use bilinear periodic sampling to avoid Fourier mask ringing.
- **C, multi-lag SSD:** B at lags `(1,2,4)` or `(1,2,4,8)`. For lag `L`, transport
  the old snapshot by `nu*L - (camera_t-camera_{t-L})`; the velocity measurement
  is `nu + pooled_residual/L`. Search radius stays 1.5 **pixels**, not 1.5
  px/frame. Fit constant velocity using displacement least squares (weights
  proportional to evidence times `L²`) and two Cauchy reweightings with velocity
  scale 0.15, as in `fit_motion`. Accumulate the summed lag evidence in KWM's
  running mean. The size gate is applied to residual velocity. All lagged images
  are shared per-sample snapshots; transported node images are temporary.
- **D, lock then refine:** compute snapshot LK and its evidence first. A node
  is locked when coherence-weighted evidence / textured imprint mass is at
  least 0.1 and pooled LK residual norm is at most 0.3. Use LK there and local
  SSD only at unlocked nodes; re-evaluate the condition every frame so nodes
  can unlock. Keep A's running mean, clip and bound.
- **Two-mode B/C/D variants:** weighted k=2 along the displacement covariance's
  principal axis, five iterations, restricted to velocities within the node's
  reachable lattice box (±0.75 per component). Accept a split only if each mode
  has at least 10% of the weight and their separation is at least 0.1 px/frame.
  Choose the mode nearest the current node velocity; independently accumulate
  the second mode as a diagnostic, without feeding it back. C robustly fits
  each selected mode across lags. D clusters the active displacement field:
  LK at locked nodes and local SSD at unlocked nodes, with the same primary
  mode selection and secondary-mode report in both cases.
- **E, literal window correlation:** masked normalized FFT correlation on the
  50×50 world, local-maximum candidates in the lattice node's reachable box,
  nearest peak among those with at least 60% of the largest score, three-step
  Newton refinement with the source's acceptance checks, robust lag fit, and
  evidence running mean with bound 0.75 and no step clip. Also report the
  source harness's strongest-peak and single-lag controls. Helpers are copied
  with attribution; the delivered script does not import another repository.
- **Fixed/oracle:** the 25-node integer lattice is the floor. The diagnostic
  oracle has exactly five true digit velocities plus the first 20 vx-major
  lattice velocities. It is an information reference, not a feasible bounded
  update or a claim that all five velocities can always get distinct lattice
  cells.

Run the full gate from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/yixunhu/miniconda3/envs/flowm/bin/python scripts/estimator_gate.py \
  --n 300 --device cpu --estimators A,B,C,D,E \
  --out results/gate_estimator.json
```

`--device cuda` restricts visibility to physical GPU 1 before importing torch;
GPU 0 is never used. Processing batches must be at most 32, and the CUDA
allocator is capped at 4.5 GiB to leave room below the 6 GB process budget.
Default processing batch size is 4. Timing always uses batch 16.
`--slow-n N` evaluates B/C/D on a common prefix and emits corresponding control
prefix rows; `--max-eval-seconds S` retains completed batches if a row reaches
its time budget. Neither option changes dataset generation. Exact row names
are accepted by `--estimators` to run independent subsets. The JSON is saved
after every completed row and contains per-digit errors, assignments, lock
steps, final nodes, secondary modes, counts, configuration and dataset hash.

Implementation validation used `--n 2 --batch-size 2 --self-check
--skip-benchmark`. It checks six-step trajectory parity against the actual
`AdaptiveNodeCell` with and without clipping; synthetic translation recovery
with camera compensation at lags 1, 4 and 8; robust-fit units; and a known
two-mode field. An independent run of the supplied `pr1_sim.py` on two
sequences matched all 25 final node vectors: maximum absolute differences
were 8.4e-7 for A and at most 4.8e-7 for all three E variants. The copied Newton
helper also matched the original helper, including acceptance decisions.
Blank observations preserved nodes and zero evidence for all 14 estimators;
their persistent image histories contained only per-sample tensors. A metric
check confirmed that two close digits sharing one node are not counted as a
resolved same-cell pair. The locked-stage routing check forces a bimodal LK
field and verifies that D selects the nearer mode rather than the unclustered
mean. A real validation sequence exercising the all-negative correlation-peak
fallback also matched literal PR1 to 7.2e-7 in every final node component.

All 16 rows (14 estimators/variants plus two references) use **300 sequences**;
no sample reduction or evaluation time cap was needed. The measured runs used
batch 4 for A/E and batch 2 for B/C/D, with independent estimator subsets running
concurrently, each with one OpenBLAS/OMP/PyTorch thread. Timing always used batch
16. CPU: Intel Xeon Gold 5220R, 2.20 GHz; PyTorch 2.9.0+cu128. CUDA was unavailable.
The subset outputs were merged only after verifying identical dataset hashes.
The single-process command above computes the same matrix but takes longer on CPU.

The saved [JSON](../results/gate_estimator.json) contains all metrics, per-digit
errors and assignments, lock steps, secondary modes, timings, source-run
configurations and the paired bootstrap. Its frames-plus-actions SHA-256 is
`09fafa77ba078b7562bfd2c0851987cd657ba0c4df611ed1949a144ab0b49024`. Bootstrap code is included in the delivered script and runs whenever
A_LK and comparison rows are present. No existing repository files were modified
and no commit was created.
