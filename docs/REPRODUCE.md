# Reproducing the results

Every command runs from the repository root. Times are for one GPU (RTX PRO 6000 Blackwell / RTX A6000) with
the defaults (32 channels, batch 16, bf16 autocast). MNIST is downloaded to `data/` on first use. Validation
sets are generated once and cached in `results/val_cache_*.pt`. Evaluation results are written to
`results/<run name>/eval[_<tag>].json` together with a per-frame MSE curve and sample frames.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # tested with torch 2.14, torchvision 0.29, Python 3.12
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
pytest -q                                # 95 passed, 2 slow cases skipped; about 53 s on CPU
KWM_SLOW=1 pytest -q                     # 97 passed, about 129 s; pytest -m slow selects only the two slow cases
```

## Command reference

```bash
python -m kwm.train --help
python -m kwm.eval --help
```

| Option | Meaning |
|---|---|
| `--model flowm` | FloWM: integer velocity lattice, memory moved by `torch.roll` |
| `--model spectral` | memory moved by a Fourier phase shift, fixed lattice of 25 velocities |
| `--model adaptive` | spectral transport with adaptive velocity nodes (the main model) |
| `--node_update lock` | opt-in lock-then-refine estimator; default `lk`; evaluation defaults to the saved rule |
| `--imprint_gamma` | LK imprint decay; no effect with `--node_update lock` |
| `--model oracle` | spectral transport with the nodes set to each sequence's true digit velocities (diagnostic) |
| `--real_velocity`, `--real_camera` | real-valued digit velocities / camera motion (default: integers, FloWM's setting) |
| `--walls`, `--odometry_noise` | reflecting walls, noisy actions (implemented, not used in RESULTS.md) |
| `--preset flowm` | FloWM's sizes: 64 channels, batch 32, 50 observed frames, lr 1e-4 |

Training writes `best.pt` (best validation MSE over 20 predicted frames), `best150.pt` (best over 150 frames)
and `last.pt`; `--keep_ckpts` also keeps `ckpt_step<N>.pt` every `--ckpt_every` steps. `kwm.eval --ckpt`
takes any of these names.

The defaults are the configuration behind RESULTS.md. `--readout_rel on` (observer-relative readout) and
`--node_update_pred off` (adaptive nodes frozen while predicting) are experimental and not used there.

To run several commands one after another on one GPU, put them in a text file, one per line, and use

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_queue.sh jobs.txt > queue.log 2>&1 &
```

## Section 1: the published FloWM weights

Downloads the checkpoint from Hugging Face (`flowm123/mnistworld-models`, 0.9 MB) into `ext/flowm_official/`,
wraps it as `runs/flowm_official_<data>/best.pt` and evaluates it on four data settings with 50 observed frames.
About 10 min.

```bash
python scripts/benchmark_official.py
# results/flowm_official_{integer,real_velocity,real_camera,real_both}/eval.json
```

## Section 2: where the difficulty is

Integer data (lr 1e-4, 10k steps, about 1.5 h each):

```bash
python -m kwm.train --model flowm    --lr 1e-4 --steps 10000 --out runs/flowm_integer
python -m kwm.train --model spectral --lr 1e-4 --steps 10000 --out runs/spectral_integer
python -m kwm.eval --run runs/flowm_integer
python -m kwm.eval --run runs/spectral_integer
```

Real velocities and camera motion, lr 3e-4 (10k steps: 1.2 / 1.7 / 0.5 h; 60k steps: 7 / 10 h):

```bash
python -m kwm.train --model flowm    --real_velocity --real_camera --lr 3e-4 --steps 10000 --out runs/flowm_lr3e-4
python -m kwm.train --model spectral --real_velocity --real_camera --lr 3e-4 --steps 10000 --out runs/spectral_lr3e-4
python -m kwm.train --model oracle   --real_velocity --real_camera --lr 3e-4 --steps 10000 --val_every 500 --out runs/oracle
python -m kwm.train --model flowm    --real_velocity --real_camera --lr 3e-4 --steps 60000 --out runs/flowm_60k
python -m kwm.train --model spectral --real_velocity --real_camera --lr 3e-4 --steps 60000 --out runs/spectral_60k
for r in flowm_lr3e-4 spectral_lr3e-4 oracle flowm_60k spectral_60k; do python -m kwm.eval --run runs/$r; done
```

## Section 3: camera motion and object velocities separately

lr 1e-3, 10k steps (1.2 / 1.7 / 3.2 h):

```bash
python -m kwm.train --model flowm    --real_camera --steps 10000 --out runs/flowm_camera
python -m kwm.train --model spectral --real_camera --steps 10000 --out runs/spectral_camera
python -m kwm.train --model adaptive --real_camera --steps 10000 --out runs/adaptive_camera
python -m kwm.train --model spectral --real_velocity --steps 10000 --out runs/spectral_velocity
python -m kwm.train --model adaptive --real_velocity --steps 10000 --out runs/adaptive_velocity
for r in flowm_camera spectral_camera adaptive_camera spectral_velocity adaptive_velocity; do
  python -m kwm.eval --run runs/$r
done
```

## Section 4: main comparison

lr 1e-3, 20k steps (flowm 2.4 h, spectral 3.3 h, adaptive 6.5 h on Blackwell; about 1.5x on A6000).
The oracle run is the one from Section 2.

```bash
python -m kwm.train --model flowm    --real_velocity --real_camera --out runs/flowm
python -m kwm.train --model spectral --real_velocity --real_camera --out runs/spectral
python -m kwm.train --model adaptive --real_velocity --real_camera --keep_ckpts --out runs/adaptive
python -m kwm.eval --run runs/flowm    --equiv_boost
python -m kwm.eval --run runs/spectral --ckpt last --equiv_boost
python -m kwm.eval --run runs/adaptive --ckpt ckpt_step5000  --equiv_boost --tag step5000
python -m kwm.eval --run runs/adaptive --ckpt ckpt_step10000 --equiv_boost --tag step10000
python -m kwm.eval --run runs/adaptive --ckpt best150 --equiv_boost --tag best150
```

Each evaluation takes 2 to 5 min. A second seed: add `--seed 1` and a different `--out`. The node velocity
error is `node_velocity_error_median` in the adaptive model's eval.json.

## Section 5: equivariance under boosts

Part of the evaluations above (`--equiv_boost`, 48 sequences). The table uses
`equiv_output.comoving_on` / `comoving_off`, field `nmse` at `t10` and `t20`. To add it to an existing
evaluation without recomputing the rest:

```bash
python -m kwm.eval --run runs/adaptive --ckpt ckpt_step5000 --tag step5000 --only_equiv
```

## Section 6: unseen speeds

Train with speeds at most 1 px/frame (lr 1e-3, 10k steps), then evaluate on speeds in (1, 2] and on scenes
boosted by a random vector in [-1, 1]^2:

```bash
for m in flowm spectral; do
  python -m kwm.train --model $m --real_velocity --real_camera --vmax_train 1 --steps 10000 --out runs/${m}_vmax1
  python -m kwm.eval --run runs/${m}_vmax1
  python -m kwm.eval --run runs/${m}_vmax1 --skip_tp --skip_ft --vmax_test 2 --v_ring --v_ring_lo 1 --tag ring
  python -m kwm.eval --run runs/${m}_vmax1 --skip_tp --skip_ft --vmax_test 1 --scene_boost 1 --tag scene_boost
done
```

## Section 7: fractional time steps

Part of every evaluation (omit `--skip_ft`): eval.json field `fractional_time`, with `model_dt05` (the model
streamed at dt = 0.5), `copy` (repeat the previous frame) and `interp` (linear interpolation between predicted
frames).

## Figures

Need the runs of Sections 1, 2 and 4 under `runs/` (`flowm_official_real_both`, `flowm`, `spectral`,
`adaptive`, `oracle`). About 5 min each; output in `results/figs/`.

```bash
python scripts/qual_figs.py        # qual_rollout.png, mse_vs_horizon.png, qual_table.json
python scripts/panorama_figs.py    # panorama*.png, velocity_space*.png, rollout*.gif, panorama_meta.json
```

## PR2: acceleration law and second-order estimator

These are CPU diagnostics with the existing dependencies. MNIST must already be
available under `data/`. The law and ceiling use the production sampler and renderer.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python scripts/accel_law.py --accel 2e-2 --n 400000 --ceiling \
  --out results/order2_gate/accel_law.json
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python scripts/estimator_gate.py --accel 2e-2 --n 100 --seed 777 --batch-size 2
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 KWM_SLOW=1 python -m pytest -q
```

The gate writes `results/order2_gate/{null,detection,synthetic,summary}.json` and
`docs/GATE_order2.md`. Its default setting is a documented conservative fallback:
none of inflation 1, 2, or 4 passed the specified 2% null target at κ=9. The
single-digit check keeps its digit visible using a real camera moving at v0 plus
uniform jitter in [-0.25,0.25]²; the 100-sequence null/detection sets retain the
standard random camera. `--reuse-null` reuses a compatible completed calibration;
`--reuse` reuses calibration and detection and refreshes the single-digit check.

Example training and evaluation (not run for this PR):

```bash
python -m kwm.train --model adaptive --node_update lock --order 2 \
  --real_velocity --real_camera --accel .02 --out runs/order2_s0
python -m kwm.eval --run runs/order2_s0 --ckpt last --test_seed 777
python -m kwm.eval --run runs/order2_s0 --ckpt last --test_seed 777 \
  --order 1 --nu_forget .95 --tag forget095
python -m kwm.eval --run runs/order2_s0 --ckpt last --test_seed 777 \
  --accel_kappa inf --tag gate_closed
```
