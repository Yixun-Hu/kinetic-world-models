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

## PR3 prediction-input pilot gate

Use the existing review selection cache and read-only pilot checkpoints. The
script writes only to `results/pred_input_gate/` in this checkout. MSE@H means
the mean over the **first H** predicted frames, with predictions clamped to
[0,1]. It also records nearest-node concentration, per-pixel effective node
count, checkpoint/cache hashes, per-sequence errors and CPU/core time. Use
`--variants` to reproduce the script-only de-ghosting, mix-once and
zero-then-freeze experiments. See [the gate](GATE_pred_input.md).

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  /home/yixunhu/miniconda3/envs/flowm/bin/python scripts/pred_input_gate.py \
  --runs /home/yixunhu/codespace/kinetic-world-models/runs \
  --cache /home/yixunhu/codespace/kinetic-world-models/results/val_cache_w50_vel1_cam1_walls0_n32_T30+150.pt \
  --n 16 --threads 4 --lock_ckpt ckpt_step10000 --lk_ckpt ckpt_step13000 --variants
```

The defaults pin those archives and 16 sequences. The default pre-lift sweep is
K=0/4; use `--ks` only for an explicit additional sweep. The observation pass is
shared across policies, each fixed anchor's FFT is cached, LK's carried
velocity metadata is compacted for frozen decoding, the frozen and
mix-once rows share transports, and the zero-then-freeze row reuses zero's first
20 steps. Matching flowm/zero JSONs are reused from `--out` (or `--baseline-dir`)
after checking checkpoint/cache hashes, sample count, numerical-dependency
source hashes, software versions and thread count; original baseline provenance
is retained. Use `--refresh-baselines` to recompute. Older last.pt JSONs are refreshed
once because their archive hashes and effective-node diagnostics are absent.
`runtime.json` reports actual process totals without double-counting shared
work. The reviewer's older robustness checks used steps 9000/11000 and 32
sequences; they are identified separately from this first-16 reproduction.

For general evaluation of one policy:

```bash
CUDA_VISIBLE_DEVICES="" python -m kwm.eval --run runs/pilot_lock --ckpt last \
  --device cpu --dec_input frozen_prelift --frozen_k 0 --skip_tp
```

Frozen overrides are experimental negative controls that failed the pilot
gate. Training accepts `--dec_input flowm|zero` only; `--frozen_k` is an
evaluation sweep knob and defaults to 0.
