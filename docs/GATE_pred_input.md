# Prediction-input pilot gate

**No frozen variant passes on either pilot (0/10).** These policies remain opt-in experiments. The default is unchanged.

## Protocol and checkpoint identity

Use the first **16** sequences from the reviewer's existing 32-sequence selection
cache, 30 observations and 150 predictions. All rows use the same sequences,
noisy camera actions, fp32 CPU inference, no autocast, and `torch.set_num_threads(8)`
as in `.pytest_cache/frozen_eval.py`. Predictions are clamped to [0,1]. MSE@H
is averaged over **all first H frames**, sequences and viewport pixels, not just
frame H. The frame cache is loaded directly with `map_location='cpu'`; it is
never regenerated. The cache SHA-256 is
`1d3f58ce06e2c743f9d26135e47fa8d4aa66da98f08baa27f8764597bcf266f4`.

The requested `runs/pilot_lock/last.pt` and `runs/pilot_lk/last.pt` are being
updated by ongoing training. This gate loaded **lock step 10000 and LK step
13000**, once each, and retained the loaded model across its seven policies.
The reviewer's saved output used steps **9000 and 11000** and 32 sequences.
Thus the protocol matches, but these are **not numerical reproductions of the
reviewer's older-checkpoint/32-sequence values**. In particular, zero input is
not uniformly better at every horizon on these newer checkpoints.

Weights were checked with exact tensor equality against the read-only archives
`pilot_lock/ckpt_step10000.pt` and `pilot_lk/ckpt_step13000.pt`. Use the pinned
command in [REPRODUCE.md](REPRODUCE.md#pr3-prediction-input-pilot-gate) to recover
the measured weights after `last.pt` advances. Each raw JSON identifies the
actual checkpoint path, step, full-file hash, cache hash and source hashes.

For flowm and zero, prediction node updates are enabled, as in the reviewer
script. Frozen modes ignore this setting and take no prediction measurements.
The naive frozen control uses the final post-activation features and carried
node velocities, with no added activation. Pre-lift K=0 freezes the actual
last mix output; K>0 replays a zero-input tail from the earlier observation
state. See [the exact definitions](PR3_pred_input.md).

## Gate

A frozen variant passes only if **both** MSE@20 and MSE@150 are no greater
than `min(flowm, zero)` at the corresponding horizon, on the same checkpoint
and subset. No tolerance or significance adjustment is applied. MSE@50 is
reported but does not determine pass/fail.

| Checkpoint | Policy | MSE@20 | MSE@50 | MSE@150 | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| pilot_lock | flowm | 0.029367 | 0.044946 | 0.083409 | reference |
| pilot_lock | zero | 0.026780 | 0.042603 | 0.084857 | reference |
| pilot_lock | frozen | 0.060444 | 0.085372 | 0.124119 | FAIL |
| pilot_lock | frozen_prelift K=0 | 0.048426 | 0.073138 | 0.117348 | FAIL |
| pilot_lock | frozen_prelift K=1 | 0.047636 | 0.072569 | 0.117078 | FAIL |
| pilot_lock | frozen_prelift K=2 | 0.045975 | 0.069954 | 0.112937 | FAIL |
| pilot_lock | frozen_prelift K=4 | 0.043644 | 0.065184 | 0.104567 | FAIL |
| pilot_lk | flowm | 0.035295 | 0.058887 | 0.111627 | reference |
| pilot_lk | zero | 0.031235 | 0.054237 | 0.095042 | reference |
| pilot_lk | frozen | 0.084617 | 0.120428 | 0.174394 | FAIL |
| pilot_lk | frozen_prelift K=0 | 0.052015 | 0.085262 | 0.142220 | FAIL |
| pilot_lk | frozen_prelift K=1 | 0.051264 | 0.083321 | 0.139097 | FAIL |
| pilot_lk | frozen_prelift K=2 | 0.052043 | 0.083958 | 0.140917 | FAIL |
| pilot_lk | frozen_prelift K=4 | 0.052710 | 0.080658 | 0.134549 | FAIL |

## Node concentration

Mean nearest-true-velocity-node squared feature energy / total node energy
inside each visible digit's viewport-clipped bounding box, pooled over visible
digit instances at the stated frame. This is a local diagnostic, not object
assignment; overlapping boxes can share energy. Definition and edge cases are
in [PR3_pred_input.md](PR3_pred_input.md#diagnostics-and-output-names).

| Checkpoint | Policy | tau=1 | tau=10 | tau=20 |
| --- | --- | ---: | ---: | ---: |
| pilot_lock | flowm | 0.1211 | 0.1472 | 0.1445 |
| pilot_lock | zero | 0.1578 | 0.1920 | 0.2016 |
| pilot_lock | frozen | 0.1264 | 0.1522 | 0.1439 |
| pilot_lock | frozen_prelift K=0 | 0.1592 | 0.1676 | 0.1602 |
| pilot_lock | frozen_prelift K=1 | 0.1592 | 0.1677 | 0.1599 |
| pilot_lock | frozen_prelift K=2 | 0.1726 | 0.1748 | 0.1702 |
| pilot_lock | frozen_prelift K=4 | 0.1841 | 0.1861 | 0.1752 |
| pilot_lk | flowm | 0.0808 | 0.0830 | 0.0804 |
| pilot_lk | zero | 0.0953 | 0.0995 | 0.0924 |
| pilot_lk | frozen | 0.0846 | 0.1035 | 0.0961 |
| pilot_lk | frozen_prelift K=0 | 0.0964 | 0.1047 | 0.0987 |
| pilot_lk | frozen_prelift K=1 | 0.0960 | 0.1041 | 0.0986 |
| pilot_lk | frozen_prelift K=2 | 0.1010 | 0.1064 | 0.1013 |
| pilot_lk | frozen_prelift K=4 | 0.1016 | 0.1030 | 0.1013 |

## Runtime and limitations

The completed 16-sequence sweep took **20.21 minutes of CPU-only wall time**
(sum of the 14 measured rollouts), with **143.14 summed CPU-core minutes**
reported by `time.process_time()`. The 30-minute budget was interpreted as
elapsed CPU-only execution, matching the reviewer's elapsed-second timings,
not summed time across eight threads. Two unfinished single-thread attempts
(32 then 16 sequences) were stopped after about 9.5 minutes combined; no scores
from those attempts are retained. Including those attempts, elapsed evaluation
time was approximately **29.71 minutes**. A literal 30 CPU-core-minute
budget was not met; the wall-time budget was the operative interpretation.

The permitted reduction to 16 sequences was necessary for runtime. This is a
selection-cache diagnostic on one checkpoint per estimator, not held-out
multi-seed evidence. Changing checkpoint steps and sample size prevents a
numerical comparison with the old reviewer table. Analytic frozen transport
is ballistic and periodic; it does not predict future collisions or new
observations. A short zero-input tail reduces some copy energy but does not
make any measured frozen policy meet the gate.

Raw data: [results/pred_input_gate](../results/pred_input_gate), one JSON per
checkpoint/policy, including per-sequence prefix MSE, framewise MSE curves,
concentration, pass/fail, and CPU/wall timing. No checkpoint or cache was written
outside this worktree.
