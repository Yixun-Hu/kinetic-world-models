# Order-2 estimator gate

100 sequences, seed 777, batches of 2; frozen random C=32 encoder (seed 0), K=8.
CPU, fp32, MNIST test images; Hungarian assignment before visibility filtering.

Null denominator: all 30 node-steps for nodes with cumulative evidence at frame 29.

| Inflation | κ | Null node-step rate | At frame 29 |
| --- | --- | --- | --- |
| 1 | 4 | 4.421% | 6.240% |
| 1 | 9 | 3.628% | 5.040% |
| 2 | 4 | 3.740% | 5.280% |
| 2 | 9 | 2.963% | 4.280% |
| 4 | 4 | 3.067% | 4.320% |
| 4 | 9 | 2.161% | 2.960% |

Chosen inflation: 4; calibration passed: False.

| Estimator | Matched median velocity error | Mean |
| --- | --- | --- |
| order1 | 0.060391 | 0.112824 |
| order2 | 0.058931 | 0.112322 |
| order1_F0.9 | 0.046026 | 0.101751 |
| order1_F0.95 | 0.052192 | 0.107232 |

Velocity truth is the mean over [29,30], v(29.5); the instantaneous v(29) differs by a/2.

| Acceleration magnitude bin | Visible digits | Gate open |
| --- | --- | --- |
| [0, 1e-07) | 224 | 4.464% |
| [0.002, 0.004) | 97 | 10.309% |
| [0.004, 0.008) | 100 | 1.000% |
| [0.008, 0.02) | 79 | 12.658% |

Single-digit a=(.004,0): velocity error 0.003079; acceleration error 0.000717; gate 0.972892.

Synthetic camera: v0 plus U[-.25,.25]² per step, keeping the digit visible.

Total elapsed CPU wall time: 653.59 s (one thread).

Fit/gate measurements are independent of feature-memory transport. The order-1 and order-2 arms
share identical PR1 measurements; the two forgetting arms rerun the production estimator.
Null fits are reused across gate settings, since the gate never feeds back into the measurement.
This is an estimator diagnostic, not evidence of trained forecasting improvement.
