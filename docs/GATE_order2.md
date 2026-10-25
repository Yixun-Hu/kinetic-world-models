# Order-2 estimator gate

**NO-GO**: null < 2% and at least 10% lower displacement proxy mean than best F are both required.
Selected order 2: κ=9, null 0.396%, proxy improvement -0.30%.

100 five-digit sequences per dataset, seed 777; frozen C32/K8 encoder seed 0; CPU fp32, one thread.
Prediction policy: off. F sweep: 0.6, 0.7, 0.8, 0.9, 0.95; κ sweep: 4, 9.
All arms share the order-1 Hungarian assignment among nodes with evidence in every arm, then the last-ten-frame visibility filter.

| F under fit | κ | Null node-step rate | Eligible (n_eff ≥ 4) rate | Final-frame rate | Final n_eff p10 / median / p90 |
| --- | --- | --- | --- | --- | --- |
| 1 | 4 | 2.074% | 6.027% | 2.520% | 1.10 / 4.50 / 9.92 |
| 1 | 9 | 0.396% | 1.149% | 0.320% | 1.10 / 4.50 / 9.92 |
| 0.9 | 4 | 2.524% | 7.438% | 4.200% | 1.08 / 4.37 / 9.70 |
| 0.9 | 9 | 0.566% | 1.667% | 0.880% | 1.08 / 4.37 / 9.70 |

Null denominator: evidence-bearing node-steps at each time. JSON also records eligible counts and n_eff quantiles over time.

| F under fit | K-accel final n_eff p10 / median / p90 | Fraction n_eff ≥ 4 |
| --- | --- | --- |
| 1 | 1.15 / 4.31 / 9.73 | 53.36% |
| 0.9 | 1.10 / 4.24 / 9.69 | 52.24% |

| Arm | Median velocity error | Mean displacement @20 | @50 | Proxy mean | Obs clamp rate |
| --- | --- | --- | --- | --- | --- |
| order1 | 0.060391 | 2.7498 | 9.1951 | 5.9725 | 0.000% |
| order2_k4 | 0.058527 | 2.7861 | 9.4837 | 6.1349 | 0.021% |
| order2_k9 | 0.060391 | 2.7462 | 9.1829 | 5.9645 | 0.003% |
| order1_F0.6 | 0.069907 | 3.8068 | 11.3799 | 7.5933 | 0.000% |
| order2_F0.6_k4 | 0.070116 | 3.9098 | 11.9540 | 7.9319 | 0.255% |
| order2_F0.6_k9 | 0.069907 | 3.8211 | 11.4534 | 7.6373 | 0.033% |
| order1_F0.7 | 0.055679 | 3.3843 | 10.4475 | 6.9159 | 0.000% |
| order2_F0.7_k4 | 0.056019 | 3.4655 | 10.9734 | 7.2194 | 0.168% |
| order2_F0.7_k9 | 0.056019 | 3.4145 | 10.6157 | 7.0151 | 0.029% |
| order1_F0.8 | 0.050407 | 2.9531 | 9.5026 | 6.2279 | 0.000% |
| order2_F0.8_k4 | 0.051149 | 3.0652 | 10.1292 | 6.5972 | 0.124% |
| order2_F0.8_k9 | 0.051149 | 3.0086 | 9.7917 | 6.4002 | 0.004% |
| order1_F0.9 | 0.050203 | 2.7534 | 9.1405 | 5.9470 | 0.000% |
| order2_F0.9_k4 | 0.050203 | 2.8497 | 9.7273 | 6.2885 | 0.060% |
| order2_F0.9_k9 | 0.050215 | 2.8005 | 9.3965 | 6.0985 | 0.001% |
| order1_F0.95 | 0.052957 | 2.7661 | 9.1845 | 5.9753 | 0.000% |
| order2_F0.95_k4 | 0.053021 | 2.8748 | 9.8676 | 6.3712 | 0.047% |
| order2_F0.95_k9 | 0.052957 | 2.8147 | 9.4726 | 6.1437 | 0.001% |

Best F by proxy mean: 0.9. Best order 2 + that F: order2_F0.9_k9.
Velocity truth is v(29.5). Displacement from p(29) is ντ + ½gατ(τ−1); error is Euclidean distance in pixels, averaged over digits.
The proxy averages τ=20 and 50 equally. It uses final node velocity and acceleration; it omits future clamps, rendering, and association changes.
Clamp rates count evidence-bearing node-steps, include the half-step acceleration, and report order-1 saturation too.

| Arm | Acceleration bin | Visible assigned digits | Gate open |
| --- | --- | --- | --- |
| order2_k4 | [0, 1e-07) | 224 | 2.232% |
| order2_k4 | [0.002, 0.004) | 97 | 0.000% |
| order2_k4 | [0.004, 0.008) | 100 | 4.000% |
| order2_k4 | [0.008, 0.02) | 79 | 5.063% |
| order2_k9 | [0, 1e-07) | 224 | 0.446% |
| order2_k9 | [0.002, 0.004) | 97 | 0.000% |
| order2_k9 | [0.004, 0.008) | 100 | 2.000% |
| order2_k9 | [0.008, 0.02) | 79 | 0.000% |
| order2_F0.9_k4 | [0, 1e-07) | 224 | 1.786% |
| order2_F0.9_k4 | [0.002, 0.004) | 97 | 2.062% |
| order2_F0.9_k4 | [0.004, 0.008) | 100 | 5.000% |
| order2_F0.9_k4 | [0.008, 0.02) | 79 | 2.532% |
| order2_F0.9_k9 | [0, 1e-07) | 224 | 0.446% |
| order2_F0.9_k9 | [0.002, 0.004) | 97 | 0.000% |
| order2_F0.9_k9 | [0.004, 0.008) | 100 | 1.000% |
| order2_F0.9_k9 | [0.008, 0.02) | 79 | 0.000% |

Single-digit paired ensemble: 50 seeds starting at 777; fast frozen C2/K2 encoder, world 32, window 24.
The camera tracks v0 plus U[-.25,.25]²; this is easier than the five-digit random camera. Select the highest-evidence node, without using truth.

| Acceleration | κ | Final gate-open rate |
| --- | --- | --- |
| 0 | 4 | 0.000% |
| 0 | 9 | 0.000% |
| 0.004 | 4 | 100.000% |
| 0.004 | 9 | 94.000% |

Total runtime: 1387.63 seconds on one CPU thread (budget: 1800 seconds).
κ is selected among calibrated settings by proxy mean. The +F summary uses the best order-1 F and best κ; its null rates are reported separately.
Fits are reused only where measurement streams are identical. Forgetting arms rerun the production estimator.
All settings are exploratory in-sample diagnostics; no trained forecasting improvement is established.
