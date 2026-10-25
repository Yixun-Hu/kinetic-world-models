"""Evaluation.

python -m kwm.eval --run runs/<name> [--ckpt best] [--n_val 1000] [--bs 50]

Outputs results/<name>/eval.json + mse_curve.png, ft_curve.png, samples.png:
  * MSE / PSNR / SSIM over the first 20 and all 150 predicted frames (mean, 95% CI over sequences)
  * per-frame MSE curve + linear slope (drift diagnostic)
  * bounce-specific local MSE (digit bbox window, 10 frames after each bounce in the
    prediction range), split by in-view / out-of-view, plus a no-bounce control
  * velocity probe: the readout's mean-velocity field (and the velocity of the argmax node) inside each
    digit's ink vs the true digit velocity (l2), over the first 20 predicted frames
  * fractional time: dt=0.5 data (2x frame rate); kinetic models stream with dt=0.5,
    every model also gets frame-copy / linear-interp baselines built from its dt=1 rollout
  * throughput: train step (fwd+bwd, bs=--tp_bs, T=70) ms, peak memory; inference ms/seq
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .env import EnvConfig, subsample
from .model import build_model, count_params, make_wall0, oracle_nodes_from
from .train import ROOT, VAL_SEED, VAL_T_OBS, VAL_T_PRED, val_cache, val_cache_description, env_cfg_from_args
from .provenance import source_hashes, env_fingerprint

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


# ------------------------------------------------------------- metrics ----
def _gauss_win(size=11, sigma=1.5, device="cpu"):
    g = torch.exp(-(torch.arange(size, device=device).float() - size // 2) ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(x, y, data_range=1.0):
    """Gaussian SSIM (Wang et al. 2004, 11x11, sigma 1.5, valid padding). x,y [N,1,H,W] -> [N]."""
    w = _gauss_win(device=x.device)
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mx, my = F.conv2d(x, w), F.conv2d(y, w)
    sxx = F.conv2d(x * x, w) - mx ** 2
    syy = F.conv2d(y * y, w) - my ** 2
    sxy = F.conv2d(x * y, w) - mx * my
    s = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))
    return s.mean((1, 2, 3))


def ci95(a):
    a = np.asarray(a, dtype=np.float64)
    if len(a) < 2:
        return float(a.mean()) if len(a) else float("nan"), float("nan")
    return float(a.mean()), float(1.96 * a.std(ddof=1) / math.sqrt(len(a)))


def digit_masks(cent, half, cam, world, win):
    """Window-pixel masks of digit ink bboxes. cent [N,nd,2], half [N,nd,2], cam [N,2] -> [N,nd,win,win] bool."""
    j = torch.arange(win, device=cent.device).float()
    wx = cam[:, None, 0, None] + j                                          # [N,1,win] world x of window cols
    wy = cam[:, None, 1, None] + j
    dx = torch.remainder(wx - cent[..., 0, None] + world / 2, world) - world / 2   # [N,nd,win]
    dy = torch.remainder(wy - cent[..., 1, None] + world / 2, world) - world / 2
    mx = dx.abs() <= half[..., 0, None] + 0.5
    my = dy.abs() <= half[..., 1, None] + 0.5
    return my[..., :, None] & mx[..., None, :]


# ------------------------------------------------------------ rollouts ----
def load_model(run, which, device, node_update=None, return_provenance=False, order=None, accel_kappa=None, nu_forget=None):
    ck = torch.load(os.path.join(run, f"{which}.pt"), map_location=device, weights_only=False)
    args = dict(ck["args"])
    if node_update is not None:
        args["node_update"] = node_update
    for key, value in (("order", order), ("accel_kappa", accel_kappa), ("nu_forget", nu_forget)):
        if value is not None:
            args[key] = value
    m = build_model(args).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    result = (m, args, ck["step"])
    return result + (ck.get("provenance"),) if return_provenance else result


@torch.no_grad()
def rollout(model, frames, an, cam0, T_obs, T_pred, dt, wall_band, amp, device, aux=False, vel0=None):
    """vel0 [B,nd,2]: true digit velocities, used only by --oracle_nodes models."""
    obs, ain, aout = frames[:, :T_obs], an[:, :T_obs], an[:, T_obs:T_obs + T_pred]
    wall0 = make_wall0(cam0, model.world, wall_band) if model.collision_kind == "bounce" else None
    nodes = oracle_nodes_from(vel0.to(frames.device)) if model.oracle else None
    with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=amp):
        out = model(obs, ain, aout, dt=dt, wall0=wall0, return_aux=aux, nodes=nodes)
    if aux:
        return out[0].float(), {k: v.float() for k, v in out[1].items()}
    return out.float()


@torch.no_grad()
def equiv_memory(model, args, cfg, n, device, ts=(10, 30), seed=777, chunk=16, modes=("comoving", "scene")):
    """Boost equivariance error of the memory (normalised L2 on the feature channels).

    Two versions of "the same scene, boosted by nu_hat":
      comoving: the observer is boosted too (camera velocity + nu_hat): the frames are identical and only the
                actions change, a_t -> a_t + nu_hat*dt. Exact Galilean symmetry of the partially observed task;
                the memory (ego frame) should transform by the velocity action alone (no shift).
      scene:    only the digits are boosted (camera unchanged), re-rendered from the same seed; the memory
                should transform by the velocity action plus a displacement nu_hat*t. NOT an exact symmetry under
                partial observation (what enters the fixed window changes), so it has a floor even on-lattice.
    nu_hat: 'on' = lattice vectors k*Delta, k in {-1,0,1}^2 minus 0; 'off' = U[-1,1]^2.
    The comparison excludes edge nodes whose source node (nu - round(nu_hat)) is off the lattice (equiv.target_mask).
    Torus data only; the adaptive model is handled by equiv_memory_nodes."""
    from .env import make_batch
    from .equiv import boost_transform, equiv_error, target_mask, interior_mask
    v = args["model"]
    if cfg.walls or model.oracle or model.collision_kind != "none":
        return {"skipped": "needs torus data and a fixed lattice without collisions (flowm, spectral)"}
    if v == "adaptive":
        return equiv_memory_nodes(model, args, cfg, n, device, ts, seed, chunk)
    spacing = 1.0 if v == "flowm" else args.get("node_spacing", 1.0)
    nodes = model.nodes().detach().float()
    T = max(ts) + 1
    g = torch.Generator().manual_seed(seed)
    ks = torch.tensor([[i, j] for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)], dtype=torch.float32)
    boosts = {"on": ks[torch.randint(0, len(ks), (n,), generator=g)] * spacing,
              "off": torch.rand(n, 2, generator=g) * 2 - 1}
    out = {"spacing": spacing}
    for em in modes:
        for kind, nu in boosts.items():
            errs = {t: [] for t in ts}
            for i in range(0, n, chunk):
                nb = nu[i:i + chunk]
                B = len(nb)
                seed_i = seed * 1000 + i
                b0 = make_batch(B, T, 1, cfg, device, torch.Generator().manual_seed(seed_i), split="test")
                if em == "scene":
                    b1 = make_batch(B, T, 1, cfg, device, torch.Generator().manual_seed(seed_i), split="test", boost=nb)
                    f1, a1 = b1["frames"], b1["actions_noisy"]
                else:
                    f1 = b0["frames"]
                    a1 = b0["actions_noisy"].clone()
                    a1[:, 1:] += nb.to(device)[:, None] * cfg.dt
                runs = []
                for f, act in ((b0["frames"], b0["actions_noisy"]), (f1, a1)):
                    _, aux = model(f[:, :T], act[:, :T], act[:, T:T + 1], return_states=ts)
                    runs.append(aux["states"])
                for t in ts:
                    disp_t = t if em == "scene" else 0
                    ht = boost_transform(runs[0][t], nodes, spacing, model.C, nb.to(device), disp_t)
                    mask = target_mask(nodes, spacing, nb.to(device))
                    mask = mask & interior_mask(nodes, spacing).to(mask.device)[None]   # interior nodes only
                    errs[t] += equiv_error(ht, runs[1][t], model.C, mask).tolist()
            out[f"{em}_{kind}"] = {f"t{t}": ci95(errs[t]) for t in ts}
    return out


@torch.no_grad()
def equiv_memory_nodes(model, args, cfg, n, device, ts=(10, 30), seed=777, chunk=16):
    """Memory equivariance for the adaptive model: comoving boost only (frames identical, actions + u); memory transformed by
    equiv.boost_transform_nodes (all adaptive node velocities + u). 'on' = integer u, 'off' = U[-1,1]^2."""
    from .env import make_batch
    from .equiv import boost_transform_nodes, equiv_error, interior_mask
    if cfg.walls:
        return {"skipped": "needs torus data"}
    spacing = args.get("node_spacing", 1.0)
    nodes = model.nodes().detach().float()
    T = max(ts) + 1
    g = torch.Generator().manual_seed(seed)
    ks = torch.tensor([[i, j] for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)], dtype=torch.float32)
    boosts = {"on": ks[torch.randint(0, len(ks), (n,), generator=g)] * spacing,
              "off": torch.rand(n, 2, generator=g) * 2 - 1}
    out = {"spacing": spacing}
    for kind, nu in boosts.items():
        errs = {t: [] for t in ts}
        for i in range(0, n, chunk):
            nb = nu[i:i + chunk]
            B = len(nb)
            b0 = make_batch(B, T, 1, cfg, device, torch.Generator().manual_seed(seed * 1000 + i), split="test")
            a1 = b0["actions_noisy"].clone()
            a1[:, 1:] += nb.to(device)[:, None] * cfg.dt
            runs = []
            for act in (b0["actions_noisy"], a1):
                _, aux = model(b0["frames"][:, :T], act[:, :T], act[:, T:T + 1], return_states=ts)
                runs.append(aux["states"])
            for t in ts:
                ht, mask = boost_transform_nodes(runs[0][t], nodes, spacing, model.C, nb.to(device),
                                                 model.cell.bound)
                mask = mask & interior_mask(nodes, spacing).to(mask.device)[None]
                errs[t] += equiv_error(ht, runs[1][t], model.C, mask).tolist()
        out[f"comoving_{kind}"] = {f"t{t}": ci95(errs[t]) for t in ts}
    return out


@torch.no_grad()
def equiv_output(model, args, cfg, n, device, ts=(10, 20), seed=777, chunk=16):
    """Output-level boost equivariance (end-to-end flow equivariance; insensitive to the internal gauge).
    Comoving boost u: the observer moves with the scene, so the observed frames are identical and every action
    (observation and prediction phase) becomes a_t + u*dt. The predicted frames must then be identical to the
    unboosted predictions (the ego-frame action of the boost on frames is the identity in the comoving case).
    Reports, at prediction steps ts, mean((p' - p)^2) ('mse') and that normalised by mean(p^2) ('nmse'),
    for 'on' = lattice u (integer for spacing 1) and 'off' = U[-1,1]^2; all models."""
    from .env import make_batch
    if cfg.walls:
        return {"skipped": "needs torus data"}
    T_obs, T_pred = args.get("t_obs", VAL_T_OBS), max(ts)
    spacing = args.get("node_spacing", 1.0) if args["model"] != "flowm" else 1.0
    g = torch.Generator().manual_seed(seed)
    ks = torch.tensor([[i, j] for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)], dtype=torch.float32)
    boosts = {"on": ks[torch.randint(0, len(ks), (n,), generator=g)] * spacing,
              "off": torch.rand(n, 2, generator=g) * 2 - 1}
    out = {"kind": "comoving, predictions", "t_obs": T_obs,
           "interior": "scene speeds |v|<=1 so that v and v+u stay inside the +-2 lattice (edge channels unused)"}
    cfg_int = EnvConfig(**dict(cfg.to_dict(), vmax=1.0, v_ring=0.0, scene_boost=0.0))
    sets = [(k, nu, cfg, k) for k, nu in boosts.items()] + [(k, nu, cfg_int, k + "_interior") for k, nu in boosts.items()]
    for kind, nu, cfg_k, key in sets:
        mse = {t: [] for t in ts}; nmse = {t: [] for t in ts}
        for i in range(0, n, chunk):
            nb = nu[i:i + chunk].to(device)
            B = len(nb)
            b0 = make_batch(B, T_obs, T_pred, cfg_k, device, torch.Generator().manual_seed(seed * 1000 + i), split="test")
            fr, a0 = b0["frames"], b0["actions_noisy"]
            a1 = a0.clone()
            a1[:, 1:] += nb[:, None] * cfg.dt
            preds = []
            for act in (a0, a1):
                nodes = None
                if model.oracle:
                    from .model import oracle_nodes_from
                    nodes = oracle_nodes_from(b0["digit_states"]["vel"][:, 0] + (nb[:, None] if act is a1 else 0))
                preds.append(model(fr[:, :T_obs], act[:, :T_obs], act[:, T_obs:T_obs + T_pred], nodes=nodes).float())
            for t in ts:
                p, q = preds[0][:, t - 1], preds[1][:, t - 1]
                e = (q - p).pow(2).mean((1, 2, 3))
                mse[t] += e.tolist()
                nmse[t] += (e / p.pow(2).mean((1, 2, 3)).clamp_min(1e-8)).tolist()
        out[f"comoving_{key}"] = {f"t{t}": {"nmse": ci95(nmse[t]), "mse": ci95(mse[t])} for t in ts}
    return out


@torch.no_grad()
def node_velocity_error(model, cache, T_obs, device, n=64, bs=16):
    """adaptive: at the end of the observation, distance |nu_i - v| of the closest adaptive node to each digit's true
    velocity (digits visible at least once in the last 10 observed frames); returns median, mean and count."""
    errs = []
    n = min(n, cache["frames"].shape[0])
    for i in range(0, n, bs):
        sl = slice(i, min(i + bs, n))
        fr = cache["frames"][sl].float().to(device); an = cache["actions_noisy"][sl].to(device)
        _, aux = model(fr[:, :T_obs], an[:, :T_obs], an[:, T_obs:T_obs + 1], record_nodes=True)
        nu = aux["node_traj"][:, T_obs - 1]                                        # [B,V,2] after the last obs
        v = cache["vel"][sl, T_obs - 1].to(device)                                 # [B,nd,2]
        vis = cache["visible"][sl, T_obs - 10:T_obs].any(1).to(device)             # [B,nd]
        d = (nu[:, None] - v[:, :, None]).norm(dim=-1).amin(-1)                    # [B,nd]
        errs += d[vis].tolist()
    t = torch.tensor(errs)
    return {"median": float(t.median()), "mean": float(t.mean()), "n": len(errs)}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", default="best", help="checkpoint basename in the run dir: best | last | ckpt_step2000 ...")
    p.add_argument("--n_val", type=int, default=None, help="default: the run's --n_val")
    p.add_argument("--test_seed", type=int, default=None,
                   help="main and fractional-time cache seed (default: selection seed 12345)")
    p.add_argument("--bs", type=int, default=50)
    # the run's preset (T_obs, C, n_val, ...) is read from its checkpoint args
    p.add_argument("--device", default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--skip_ft", action="store_true")
    p.add_argument("--skip_tp", action="store_true")
    p.add_argument("--tp_bs", type=int, default=32)
    p.add_argument("--out", default=None)
    # alternative validation sets (unseen speeds) -> written as eval_<tag>.json / <tag>_*.png
    p.add_argument("--vmax_test", type=float, default=None, help="override the run's validation speed bound")
    p.add_argument("--v_ring", action="store_true", help="validation speeds in (v_ring_lo, vmax_test]")
    p.add_argument("--v_ring_lo", type=float, default=None, help="inner ring bound (default: the run's vmax_train)")
    p.add_argument("--scene_boost", type=float, default=None, help="validation: per-sequence boost u~U[-b,b]^2")
    p.add_argument("--tag", default=None, help="suffix for the output files (eval_<tag>.json)")
    p.add_argument("--node_update_pred", default=None, choices=[None, "on", "off"],
                   help="adaptive: override whether nodes/imprints are updated during the prediction phase")
    p.add_argument("--node_update", default=None, choices=["lk", "lock"],
                   help="adaptive estimator override; default: checkpoint's saved value (lk for older checkpoints)")
    p.add_argument("--order", type=int, choices=[1, 2], default=None)
    p.add_argument("--accel_kappa", type=float, default=None, help="override gate strength; inf closes it")
    p.add_argument("--nu_forget", type=float, default=None)
    # boost equivariance (on / off the velocity lattice)
    p.add_argument("--equiv_boost", action="store_true", help="boost equivariance of the memory (t=10,30) and of the predictions (t=10,20)")
    p.add_argument("--n_equiv", type=int, default=48)
    p.add_argument("--only_equiv", action="store_true",
                   help="skip everything but the equivariance tests; the result is MERGED into an existing eval<tag>.json")
    p.add_argument("--equiv_boost_output", action="store_true",
                   help="output-level equivariance only (end-to-end flow equivariance of the predictions; also implied by --equiv_boost)")
    a = p.parse_args(argv)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    amp = a.amp and device.startswith("cuda")
    name = os.path.basename(os.path.normpath(a.run))
    out_dir = a.out or os.path.join(ROOT, "results", name)
    os.makedirs(out_dir, exist_ok=True)

    model, args, step, ckpt_provenance = load_model(a.run, a.ckpt, device, node_update=a.node_update,
                                                    return_provenance=True, order=a.order, accel_kappa=a.accel_kappa, nu_forget=a.nu_forget)
    if a.node_update_pred is not None:
        model.node_update_pred = a.node_update_pred == "on"
        args = dict(args, node_update_pred=a.node_update_pred)
    for k_, v_ in (("vmax_test", a.vmax_test), ("scene_boost", a.scene_boost), ("v_ring_lo", a.v_ring_lo)):
        if v_ is not None:
            args[k_] = v_
    if a.v_ring:
        args["v_ring"] = True
    ns = argparse.Namespace(**args)
    n_val = a.n_val or args.get("n_val", 1000)
    cfg = env_cfg_from_args(args, split="val")
    if cfg.accel and model.oracle:
        raise ValueError("the constant-velocity oracle model is undefined on K-accel")
    test_seed = VAL_SEED if a.test_seed is None else a.test_seed
    sfx = (f"_seed{test_seed}_{a.ckpt}" if test_seed != VAL_SEED else "") + (f"_{a.tag}" if a.tag else "")
    W, win, band = cfg.world, cfg.window, args.get("wall_band", 3)
    cache = val_cache(cfg, n_val, T_obs=args.get("t_obs", VAL_T_OBS), device=device, seed=test_seed)
    T_obs, T_pred = args.get("t_obs", VAL_T_OBS), VAL_T_PRED
    kinetic = args["model"] != "flowm"
    provenance = dict(source_hashes(), eval_env=env_fingerprint(cfg))
    if ckpt_provenance is not None and ckpt_provenance.get("files") != provenance["files"]:
        print("[kwm] WARNING: evaluation source file hashes differ from checkpoint provenance.")
    res = dict(run=name, ckpt=a.ckpt, step=step, model=args["model"], params=count_params(model),
               num_v=model.num_v, env=cfg.to_dict(), n_val=n_val, tag=a.tag, seed=args.get("seed"),
               test_seed=test_seed, provenance=provenance, ckpt_provenance=ckpt_provenance, sets={},
               node_update=model.node_update, node_update_pred=getattr(model, "node_update_pred", None),
               readout_rel=getattr(model, "readout_rel", None), order=model.order,
               accel_kappa=("inf" if args.get("accel_kappa", 9.) == float('inf') else args.get("accel_kappa", 9.)),
               accel_var_inflation=args.get("accel_var_inflation", 4.),
               nu_forget=args.get("nu_forget", 1.), pred_shrink=args.get("pred_shrink", "none"))
    role = "selection" if test_seed == VAL_SEED else "heldout"
    cache_set = dict(seed=test_seed, mnist_split="test", role=role, n=n_val, t_obs=T_obs, t_pred=T_pred, chunk=50)

    # ---------------- adaptive: node velocity error at the end of the observation ----------------
    if args["model"] == "adaptive" and not cfg.walls:
        res["sets"]["node_velocity_error"] = dict(cache_set, n=min(64, n_val))
        ne = node_velocity_error(model, cache, args.get("t_obs", VAL_T_OBS), device)
        res["node_velocity_error_median"], res["node_velocity_error_mean"], res["node_velocity_error_n"] = \
            ne["median"], ne["mean"], ne["n"]
        print("[node_velocity_error]", ne)

    # ---------------- boost equivariance of the memory / of the predictions ----------------
    if a.equiv_boost or (a.only_equiv and not a.equiv_boost_output):
        res["sets"]["equiv"] = dict(seed=777, mnist_split="test", role="heldout", n=a.n_equiv, chunk=16,
                                     sequence_seed="seed * 1000 + chunk_start")
        res["equiv"] = equiv_memory(model, args, cfg, a.n_equiv, device)
        if args["model"] == "adaptive" and isinstance(res["equiv"], dict) and "comoving_off" in res["equiv"]:
            res["equiv"]["note"] = ("comoving_off is gauge-dependent for the adaptive model: with overlapping cells an object may be held "
                                    "by either of two neighbouring nodes; see equiv_output for the gauge-free check")
        print("[equiv]", json.dumps(res["equiv"]))
    if a.equiv_boost or a.equiv_boost_output:
        res["sets"]["equiv_output"] = dict(seed=777, mnist_split="test", role="heldout", n=a.n_equiv, chunk=16,
                                            sequence_seed="seed * 1000 + chunk_start")
        res["equiv_output"] = equiv_output(model, args, cfg, a.n_equiv, device)
        print("[equiv-output]", json.dumps(res["equiv_output"]))
    if a.only_equiv:
        path = os.path.join(out_dir, f"eval{sfx}.json")
        old = json.load(open(path)) if os.path.exists(path) else {}
        old.update({k: v for k, v in res.items() if k in ("equiv", "equiv_output")})
        # A merged file may contain metrics from an earlier invocation. Keep their
        # provenance, and identify the invocation that produced each new diagnostic.
        diagnostic_keys = ("provenance", "ckpt_provenance", "seed", "test_seed", "sets")
        for key in ("equiv", "equiv_output"):
            if key in res:
                old.setdefault("diagnostic_provenance", {})[key] = dict(
                    {k: res[k] for k in diagnostic_keys}, ckpt=a.ckpt, step=step)
        for k, v in res.items():
            if k not in diagnostic_keys:
                old.setdefault(k, v)
        with open(path, "w") as f:
            json.dump(old, f, indent=1)
        return

    # ---------------- main rollout (150 frames) ----------------
    res["sets"]["main"] = dict(cache_set, cache=val_cache_description(cache))
    mse_f, psnr_f, ssim_f = [], [], []       # [N, T_pred] each
    bounce_rows, ctrl_rows, vel_rows = [], [], []
    sample = None
    for i in range(0, n_val, a.bs):
        sl = slice(i, i + a.bs)
        fr = cache["frames"][sl].float().to(device)
        an = cache["actions_noisy"][sl].to(device)
        cam = cache["cam"][sl].to(device)
        pred, aux = rollout(model, fr, an, cam[:, 0], T_obs, T_pred, 1.0, band, amp, device, aux=True,
                            vel0=cache["vel"][sl, 0])
        pred = pred.clamp(0, 1)
        gt = fr[:, T_obs:T_obs + T_pred]
        N = len(fr)
        se = (pred - gt) ** 2                                                # [N,T,1,win,win]
        mse = se.mean((2, 3, 4))
        mse_f.append(mse.cpu())
        psnr_f.append((10 * torch.log10(1.0 / mse.clamp_min(1e-10))).cpu())
        ssim_f.append(ssim(pred.reshape(-1, 1, win, win), gt.reshape(-1, 1, win, win)).reshape(N, T_pred).cpu())
        if sample is None:
            sample = (fr[0].cpu(), pred[0].cpu())

        cent = cache["cent"][sl].to(device)
        half = cache["half"][sl].to(device)
        vel = cache["vel"][sl].to(device)
        nd = cent.shape[2]
        # ---- bounce-specific local MSE ----
        ev = cache["bounce_events"]
        ev = ev[(ev[:, 0] >= i) & (ev[:, 0] < i + N) & (ev[:, 2] >= T_obs)]
        for b, d, t, iv in ev.tolist():
            bl = b - i
            vals = []
            for k in range(10):
                tt = t + k
                if tt >= T_obs + T_pred:
                    break
                m = digit_masks(cent[bl, tt][None, d:d + 1], half[bl][None, d:d + 1], cam[bl, tt][None], W, win)[0, 0]
                if m.any():
                    vals.append(se[bl, tt - T_obs, 0][m].mean().item())
            if vals:
                bounce_rows.append((iv, float(np.mean(vals))))
        if cfg.walls:
            # control: digit-frames in the prediction range with no bounce in the last 10 frames
            bnc = cache["bounce"][sl].to(device)                              # [N,T,nd]
            recent = torch.stack([bnc[:, max(0, t - 9):t + 1].any(1) for t in range(bnc.shape[1])], 1)
            g = torch.Generator().manual_seed(0)
            for _ in range(4 * N):
                bl = int(torch.randint(0, N, (1,), generator=g))
                d = int(torch.randint(0, nd, (1,), generator=g))
                tt = int(torch.randint(T_obs, T_obs + T_pred, (1,), generator=g))
                if recent[bl, tt, d]:
                    continue
                m = digit_masks(cent[bl, tt][None, d:d + 1], half[bl][None, d:d + 1], cam[bl, tt][None], W, win)[0, 0]
                if m.any():
                    ctrl_rows.append(se[bl, tt - T_obs, 0][m].mean().item())
        # ---- velocity probe (first 20 predicted frames) ----
        for k in range(20):
            tt = T_obs + k
            masks = digit_masks(cent[:, tt], half, cam[:, tt], W, win)       # [N,nd,win,win]
            other = masks.sum(1, keepdim=True) > 1
            ink = gt[:, k, 0] > 0.2
            msk = masks & ~other & ink[:, None]
            cnt = msk.sum((2, 3)).float()                                    # [N,nd]
            for key in ("mean_velocity", "argmax_velocity"):
                f = aux[key][:, k]                                           # [N,2,win,win]
                est = (f[:, None] * msk[:, :, None]).sum((3, 4)) / cnt.clamp_min(1)[..., None]
                err = (est - vel[:, tt]).norm(dim=-1)                        # [N,nd]
                ok = cnt >= 5
                vel_rows += [(key, e) for e in err[ok].tolist()]

    mse_f, psnr_f, ssim_f = torch.cat(mse_f), torch.cat(psnr_f), torch.cat(ssim_f)
    for H in (20, 150):
        res[f"mse{H}"], res[f"mse{H}_ci"] = ci95(mse_f[:, :H].mean(1).numpy())
        res[f"per_seq_mse{H}"] = mse_f[:, :H].mean(1).tolist()
        res[f"psnr{H}"], res[f"psnr{H}_ci"] = ci95(psnr_f[:, :H].mean(1).numpy())
        res[f"ssim{H}"], res[f"ssim{H}_ci"] = ci95(ssim_f[:, :H].mean(1).numpy())
    curve = mse_f.mean(0).numpy()
    res["mse_curve"] = curve.tolist()
    res["slope150"] = float(np.polyfit(np.arange(T_pred), curve, 1)[0])
    res["slope20"] = float(np.polyfit(np.arange(20), curve[:20], 1)[0])
    if cfg.walls:
        for lab, iv in (("in_view", 1), ("out_of_view", 0)):
            v = [x for f, x in bounce_rows if f == iv]
            res[f"bounce_mse_{lab}"], res[f"bounce_mse_{lab}_ci"] = ci95(v)
            res[f"bounce_n_{lab}"] = len(v)
        res["nobounce_ctrl_mse"], res["nobounce_ctrl_mse_ci"] = ci95(ctrl_rows)
    for key in ("mean_velocity", "argmax_velocity"):
        v = [e for k_, e in vel_rows if k_ == key]
        res[f"velocity_probe_{key}_l2"], res[f"velocity_probe_{key}_l2_ci"] = ci95(v)
        res[f"velocity_probe_{key}_n"] = len(v)

    # ---------------- fractional time (dt = 0.5) ----------------
    if not a.skip_ft:
        cfg_h = EnvConfig(**dict(cfg.to_dict(), dt=0.5))
        Th_obs, Th_pred = 2 * T_obs, 40
        res["sets"]["fractional_time"] = dict(seed=test_seed, mnist_split="test", role=role, n=n_val,
                                                t_obs=Th_obs, t_pred=Th_pred, dt=0.5, chunk=50)
        ch = val_cache(cfg_h, n_val, T_obs=Th_obs, T_pred=Th_pred, device=device, seed=test_seed)
        res["sets"]["fractional_time"]["cache"] = val_cache_description(ch)
        ft = {"model_dt05": [], "copy": [], "interp": [], "dt1_even": []}
        for i in range(0, n_val, a.bs):
            sl = slice(i, i + a.bs)
            fr = ch["frames"][sl].float().to(device)
            an = ch["actions_noisy"][sl].to(device)
            cam = ch["cam"][sl].to(device)
            gt = fr[:, Th_obs:]                                              # 40 half-frames, times T_obs .. T_obs+19.5
            # dt=1 rollout on the even (integer-time) frames
            b1 = subsample(dict(frames=fr, actions=ch["actions"][sl].to(device), actions_noisy=an,
                                bounce=ch["bounce"][sl], bounce_events=torch.zeros(0, 4, dtype=torch.long),
                                digit_states=dict(cent=ch["cent"][sl], vel=ch["vel"][sl], half=ch["half"][sl],
                                                  cam=cam, visible=ch["visible"][sl])), 2)
            p1 = rollout(model, b1["frames"], b1["actions_noisy"], cam[:, 0], T_obs, 20, 1.0, band, amp, device,
                         vel0=ch["vel"][sl, 0]).clamp(0, 1)
            # half-frame k has time T_obs + k/2: even k -> integer time -> p1[k//2];
            # odd k -> between p1[k//2] and p1[k//2+1] (copy = previous integer frame)
            nxt = lambda k: p1[:, min(k // 2 + 1, p1.shape[1] - 1)]
            copy = torch.stack([p1[:, k // 2] for k in range(Th_pred)], 1)
            interp = torch.stack([p1[:, k // 2] if k % 2 == 0 else 0.5 * (p1[:, k // 2] + nxt(k))
                                  for k in range(Th_pred)], 1)
            ft["copy"].append(((copy - gt) ** 2).mean((2, 3, 4)).cpu())
            ft["interp"].append(((interp - gt) ** 2).mean((2, 3, 4)).cpu())
            ft["dt1_even"].append(((p1 - b1["frames"][:, T_obs:T_obs + 20]) ** 2).mean((2, 3, 4)).cpu())
            if kinetic:
                ph = rollout(model, fr, an, cam[:, 0], Th_obs, Th_pred, 0.5, band, amp, device,
                             vel0=ch["vel"][sl, 0]).clamp(0, 1)
                ft["model_dt05"].append(((ph - gt) ** 2).mean((2, 3, 4)).cpu())
        ftres = {}
        for k, v in ft.items():
            if v:
                v = torch.cat(v)
                ftres[k] = dict(mse=ci95(v.mean(1).numpy()), curve=v.mean(0).tolist())
        if kinetic:
            ftres["ratio_dt05_over_dt1"] = ftres["model_dt05"]["mse"][0] / ftres["dt1_even"]["mse"][0]
        res["fractional_time"] = ftres

    # ---------------- throughput / memory ----------------
    if not a.skip_tp:
        tpb = a.tp_bs if device.startswith("cuda") else 2
        res["sets"]["throughput"] = dict(cache_set, n=min(tpb, n_val))
        fr = cache["frames"][:tpb].float().to(device)
        an = cache["actions_noisy"][:tpb].to(device)
        cam0 = cache["cam"][:tpb, 0].to(device)
        wall0 = make_wall0(cam0, W, band) if model.collision_kind == "bounce" else None
        tp = args.get("t_pred", 20)
        obs, ain, aout, tgt = fr[:, :T_obs], an[:, :T_obs], an[:, T_obs:T_obs + tp], fr[:, T_obs:T_obs + tp]
        tp_nodes = oracle_nodes_from(cache["vel"][:tpb, 0].to(device)) if model.oracle else None
        model.train()
        sync = (lambda: torch.cuda.synchronize()) if device.startswith("cuda") else (lambda: None)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        ts = []
        for r in range(3):
            sync(); t0 = time.time()
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=amp):
                pr = model(obs, ain, aout, target=tgt, tf_ratio=0.0, wall0=wall0, nodes=tp_nodes)
            F.mse_loss(pr.float(), tgt).backward()
            model.zero_grad(set_to_none=True)
            sync(); ts.append(time.time() - t0)
        model.eval()
        mem = torch.cuda.max_memory_allocated() / 2 ** 20 if device.startswith("cuda") else None
        ti = []
        for r in range(3):
            sync(); t0 = time.time()
            rollout(model, fr, an, cam0, T_obs, tp, 1.0, band, amp, device, vel0=cache["vel"][:tpb, 0])
            sync(); ti.append(time.time() - t0)
        res["throughput"] = dict(bs=tpb, train_step_ms=1000 * float(np.median(ts[1:])),
                                 peak_mem_mb=mem, infer_ms_per_seq=1000 * float(np.median(ti[1:])) / tpb,
                                 grad_ckpt=bool(model.grad_ckpt), amp=amp, device=device)

    with open(os.path.join(out_dir, f"eval{sfx}.json"), "w") as f:
        json.dump(res, f, indent=1)

    # ---------------- figures ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(np.arange(1, T_pred + 1), curve, color=COLORS[0], lw=2, label=name)
    ax.set_xlabel("predicted frame"); ax.set_ylabel("MSE"); ax.set_title(f"per-frame MSE ({name})")
    ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"mse_curve{sfx}.png"), dpi=120); plt.close(fig)
    if "fractional_time" in res:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        tt = 0.5 * np.arange(0, 40)
        for c, k in zip(COLORS, ("model_dt05", "copy", "interp")):
            if k in res["fractional_time"]:
                ax.plot(tt, res["fractional_time"][k]["curve"], color=c, lw=2, label=k)
        ax.plot(np.arange(0, 20), res["fractional_time"]["dt1_even"]["curve"], color=COLORS[3], lw=2, ls="--", label="dt=1 (integer frames)")
        ax.set_xlabel("predicted time"); ax.set_ylabel("MSE"); ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"ft_curve{sfx}.png"), dpi=120); plt.close(fig)
    fr0, pr0 = sample
    idx = [0, 4, 9, 19, 49, 99, 149]
    fig, axs = plt.subplots(2, len(idx), figsize=(1.6 * len(idx), 3.4))
    for c, k in enumerate(idx):
        axs[0, c].imshow(fr0[T_obs + k, 0], cmap="gray", vmin=0, vmax=1)
        axs[1, c].imshow(pr0[k, 0], cmap="gray", vmin=0, vmax=1)
        axs[0, c].set_title(f"+{k + 1}", fontsize=8)
        for r in range(2):
            axs[r, c].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"samples{sfx}.png"), dpi=120); plt.close(fig)
    print(json.dumps({k: v for k, v in res.items()
                      if not k.startswith("per_seq_")
                      and k not in ("mse_curve", "fractional_time", "provenance", "ckpt_provenance")}, indent=1))


if __name__ == "__main__":
    main()
