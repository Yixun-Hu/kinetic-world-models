# Results

All models below have 32 hidden channels and 25 velocity nodes. Training
uses batch size 16, 30 observed and 20 predicted frames, learning rate 1e-3
unless noted, and one seed. Validation uses 300 held-out sequences, 30
observed frames, and up to 150 predicted frames. Only the benchmark of the
published weights in Section 1 uses 50 observed frames, as in the paper. The 95% confidence intervals
are about 0.002 for the 20-frame errors. "Integer data" means integer digit
velocities and integer camera motion, which is FloWM's setting. The three
models are FloWM (its cell retrained on our data), the spectral model
(spectral transport, fixed velocity lattice), and the adaptive model (spectral
transport with adaptive velocity nodes).

## 1. The published FloWM weights

We load the authors' 2D checkpoint (epoch 8, reported validation loss 5.9e-4)
into our reimplementation and evaluate it with our environment.

| Data | 20 frames | 150 frames | SSIM, 20 frames | Velocity error (px/frame) |
|---|---|---|---|---|
| Integer data | 0.0007 | 0.0020 | 0.986 | 0.003 |
| Real-valued velocities | 0.158 | 0.160 | 0.28 | 1.78 |
| Real-valued camera motion | 0.098 | 0.127 | 0.45 | 0.97 |
| Both real-valued | 0.158 | 0.158 | 0.25 | 1.80 |
| Predict all zeros | 0.156 | 0.155 | | |

The paper reports 0.0005 and 0.0018 on its own validation split, so the
environment matches. With real-valued velocities the same weights are worse
than predicting zeros from the first frame on. Real-valued camera motion alone
causes a linear drift about thirteen times faster than on integer data.

## 2. Where the difficulty is

Validation error at 20 predicted frames. The oracle sets each sequence's
velocity nodes to the true digit velocities.

| Data | FloWM | Spectral | Oracle nodes |
|---|---|---|---|
| Integer data, 10k steps, lr 1e-4 | 0.0098 | 0.0089 | |
| Both real-valued, 10k steps, lr 3e-4 | 0.077 | 0.061 | 0.0061 |
| Both real-valued, 60k steps, lr 3e-4 | 0.064 | 0.034 | |

On integer data the two models are equal, so spectral transport costs
nothing. On real-valued data FloWM reaches a plateau at 20k steps that more
training does not move. The oracle reaches 0.020 after 1.5k steps. The gap
between the oracle and the spectral model is entirely due to the velocity
lattice.

## 3. Translation and boost subgroups separately

| Data | FloWM | Spectral | Adaptive |
|---|---|---|---|
| Real-valued camera motion only, 10k steps | 0.047 | 0.0028 | 0.0066 (2k steps) |
| Real-valued velocities only, 10k steps | | 0.040 | 0.031 |

Making the camera motion continuous accounts for a factor of seventeen by
itself and brings the spectral model to the integer-data level, with SSIM 0.94
and a velocity error of 0.036 px/frame. Making the object velocities continuous is the harder half
and is where the adaptive nodes act.

## 4. Main comparison on real-valued velocities and camera motion

| Model | Steps | 20 frames | 150 frames | Error at frame 150 | Slope over 150 frames |
|---|---|---|---|---|---|
| FloWM | 20k | 0.067 | 0.145 | 0.31 | 1.3e-3 |
| Spectral | 20k | 0.033 | 0.287 | 0.73 | 5.2e-3 |
| Adaptive | 5k | 0.036 | 0.089 | 0.115 | 5.6e-4 |
| Adaptive | 10k | 0.033 | 0.106 | 0.156 | 8.6e-4 |
| Adaptive, best checkpoint by 20-frame error (19k) | 20k | 0.027 | 0.102 | | |
| Oracle nodes | 10k | 0.006 | 0.043 | | |

The spectral model matches the adaptive model at 20 frames but diverges over
150 frames, with the memory saturating and the predicted view turning white.
The adaptive model is the only trained model whose 150-frame error stays below
the two trivial baselines. Its long-horizon error moves between checkpoints,
so we select checkpoints on the 150-frame validation error in later runs. A
second seed of the adaptive model reaches 0.0397 at 3k steps against 0.0396
for the first.

At the end of the 30 observed frames, the median distance between a node and
the true velocity of the digit it holds is 0.054 px/frame over 320 digits.
For the fixed lattice the median distance from a digit's velocity to the
nearest node is 0.40 over all 1500 validation digits.

![Velocity space](../results/figs/velocity_space.png)

## 5. Equivariance under real-valued boosts

We boost the whole scene and the camera together by a vector $u$, which
leaves the observed frames unchanged and shifts every velocity by $u$. We
compare the predictions with and without the boost on 48 sequences and report
the relative squared error at 10 and 20 predicted frames.

| Model | $u$ on the lattice | $u$ off the lattice |
|---|---|---|
| FloWM | 0.17 / 0.16 | 0.32 / 0.35 |
| Spectral | 0.19 / 0.19 | 0.18 / 0.32 |
| Adaptive | 0.16 / 0.18 | 0.14 / 0.19 |

For the adaptive model an off-lattice boost is no harder than an
on-lattice one. The on-lattice values are not zero for any model. Part of this
comes from the finite velocity lattice, whose edge channels lose content under
a boost, and part from the viewport crop. For the adaptive nodes we compare predictions and not hidden states. Which
of two overlapping cells holds an object is a gauge choice, and a boost can
change it without changing the prediction.

## 6. Generalisation to unseen speeds

Models trained with speeds at most 1 px/frame, evaluated on speeds in (1, 2]
and on scenes boosted by a random vector.

| Model | Seen speeds | Speeds in (1, 2] | Scene boost |
|---|---|---|---|
| FloWM | 0.070 | 0.072 | 0.072 |
| Spectral | 0.044 | 0.045 | 0.046 |

Neither model degrades. Weight sharing across velocity channels gives all
models in this family equivariance to lattice boosts, so unseen speeds cost
nothing. This test does not separate the models.

## 7. Fractional time steps

Spectral transport accepts any real time step, so a model trained at
$\Delta t=1$ can predict frames at $\Delta t=0.5$ without retraining. For the
adaptive model at 5k steps the error at the half steps is 0.066, against
0.077 for repeating the previous frame and 0.065 for linear interpolation
between predicted frames. The model matches interpolation but does not beat it.

## 8. Known limitations

- All results are one or two seeds with a small model. FloWM's own
  configuration (64 channels, batch 32, 50 observed frames) has not been run.
- The adaptive model is five times worse than the oracle at 20 frames.
  Digits whose velocities fall in the same lattice cell compete for one node,
  and some digits are not locked by the end of the observation.
- Its long-horizon error is not monotone in training steps.
- Reflecting walls, where an object's velocity changes while out of view, are
  implemented but not yet evaluated on a model that predicts sharply.
