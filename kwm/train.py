"""Training CLI.

python -m kwm.train --model adaptive --real_velocity --real_camera --steps 20000 --out runs/adaptive

Data is generated online (kwm.env.make_batch) from a CPU generator seeded by
--seed; the validation set is generated once with a fixed seed (VAL_SEED, MNIST
test split) and cached in results/val_cache_<cfg>_n<n>_T<T>.pt.
Logs: <out>/train.csv (every --log_every), <out>/val.csv, <out>/config.json,
<out>/last.pt, best.pt (best validation MSE over --t_pred frames) and best150.pt
(best validation MSE over 150 frames). Re-running with the same --out resumes.
"""
import argparse
import csv
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from .env import EnvConfig, make_batch
from .model import MODELS, COLLISIONS, build_model, count_params, make_wall0, oracle_nodes_from

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_SEED = 12345
VAL_T_OBS, VAL_T_PRED = 30, 150     # validation: T_obs (= the run's --t_obs) observed + 150 predicted frames

# The defaults are the configuration of every result in docs/RESULTS.md (32 channels, batch 16,
# 30 observed + 20 predicted frames, lr 1e-3, bf16 autocast). --preset flowm restores FloWM's own sizes;
# flags given explicitly still win.
PRESETS = {
    "flowm": dict(bs=32, t_obs=50, t_pred=20, C=64, steps=40000, lr=1e-4, n_val=1000, val_n_quick=256),
}


def env_cfg_from_args(a, dt=1.0, split="train"):
    """split='train': training distribution (vmax_train); 'val': validation set (vmax_test, v_ring, scene_boost).
    With the defaults (vmax_train = vmax_test = 2, no ring/boost) both are the same distribution."""
    g = (lambda k, d=None: a.get(k, d)) if isinstance(a, dict) else (lambda k, d=None: getattr(a, k, d))
    vtr = g("vmax_train", 2.0)
    kw = dict(world=g("world"), window=g("window"), real_velocity=g("real_velocity"), real_camera=g("real_camera"),
              walls=g("walls"), odometry_noise=g("odometry_noise"), odometry_bias=g("odometry_bias"), dt=dt,
              bounce_rule=g("bounce_rule"), vmax=vtr)
    if split == "val":
        vte = g("vmax_test", None) or vtr
        lo = g("v_ring_lo", None) or vtr
        kw.update(vmax=vte, v_ring=(lo if g("v_ring", False) else 0.0), scene_boost=g("scene_boost", 0.0))
    return EnvConfig(**kw)


def val_cache(cfg, n, T_obs=VAL_T_OBS, T_pred=VAL_T_PRED, device="cpu", chunk=50):
    """Load or build the fixed validation set. Frames are stored as float16 on CPU."""
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    path = os.path.join(ROOT, "results", f"val_cache_{cfg.tag()}_n{n}_T{T_obs}+{T_pred}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    gen = torch.Generator().manual_seed(VAL_SEED)
    parts = []
    for i in range(0, n, chunk):
        b = make_batch(min(chunk, n - i), T_obs, T_pred, cfg, device, gen, split="test")
        ev = b["bounce_events"].clone()
        ev[:, 0] += i
        parts.append(dict(frames=b["frames"].half().cpu(), actions=b["actions"].cpu(),
                          actions_noisy=b["actions_noisy"].cpu(), bounce=b["bounce"].cpu(),
                          bounce_events=ev,
                          **{k: v.cpu() for k, v in b["digit_states"].items()}))
    cache = {k: torch.cat([p[k] for p in parts], 0) for k in parts[0]}
    cache["cfg"] = cfg.to_dict()
    cache["T_obs"], cache["T_pred"] = T_obs, T_pred
    torch.save(cache, path)
    return cache


def model_inputs(frames, actions_noisy, T_obs, T_pred):
    return (frames[:, :T_obs], actions_noisy[:, :T_obs], actions_noisy[:, T_obs:T_obs + T_pred],
            frames[:, T_obs:T_obs + T_pred])


@torch.no_grad()
def quick_val(model, cache, n, T_pred, bs, device, amp, wall_band, T_obs=VAL_T_OBS):
    model.eval()
    tot, cnt = 0.0, 0
    for i in range(0, n, bs):
        fr = cache["frames"][i:i + bs].float().to(device)
        an = cache["actions_noisy"][i:i + bs].to(device)
        obs, ain, aout, tgt = model_inputs(fr, an, T_obs, T_pred)
        wall0 = make_wall0(cache["cam"][i:i + bs, 0].to(device), model.world, wall_band)
        nodes = oracle_nodes_from(cache["vel"][i:i + bs, 0].to(device)) if model.oracle else None
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=amp):
            p = model(obs, ain, aout, wall0=wall0, nodes=nodes)
        tot += F.mse_loss(p.float(), tgt, reduction="sum").item() / tgt[0].numel()
        cnt += len(fr)
    model.train()
    return tot / cnt


def get_parser():
    p = argparse.ArgumentParser(description="Train a kinetic world model on Kinetic MNIST World.")
    p.add_argument("--preset", default=None, choices=list(PRESETS),
                   help="flowm: FloWM's sizes (64 channels, batch 32, 50 observed frames, lr 1e-4, 40k steps)")
    p.add_argument("--model", default="adaptive", choices=list(MODELS),
                   help="flowm: FloWM (integer roll) | spectral: Fourier-phase transport on a fixed velocity lattice | "
                        "adaptive: spectral + adaptive velocity nodes | oracle: spectral with the nodes set to the "
                        "true digit velocities (diagnostic)")
    # data
    g = p.add_argument_group("data")
    g.add_argument("--real_velocity", action="store_true", help="real-valued digit velocities (default: integers)")
    g.add_argument("--real_camera", action="store_true", help="real-valued camera motion (default: integers)")
    g.add_argument("--walls", action="store_true", help="reflecting walls (default: periodic world)")
    g.add_argument("--odometry_noise", type=float, default=0.0, help="std of additive noise on the actions")
    g.add_argument("--odometry_bias", type=float, default=0.0, help="relative scale error of the actions")
    g.add_argument("--bounce_rule", default="bbox", choices=["centroid", "bbox"], help="walls: what is confined")
    g.add_argument("--vmax_train", type=float, default=2.0, help="training digit speed bound ||v||_inf")
    g.add_argument("--vmax_test", type=float, default=None, help="validation speed bound (default = vmax_train)")
    g.add_argument("--v_ring", action="store_true", help="validation speeds in the ring (vmax_train, vmax_test]")
    g.add_argument("--scene_boost", type=float, default=0.0, help="validation: add one boost u~U[-b,b]^2 per sequence")
    g.add_argument("--world", type=int, default=50)
    g.add_argument("--window", type=int, default=32)
    g.add_argument("--t_obs", type=int, default=30, help="observed frames per training sequence")
    g.add_argument("--t_pred", type=int, default=20, help="predicted frames per training sequence")
    # model
    g = p.add_argument_group("model")
    g.add_argument("--C", type=int, default=32, help="hidden channels")
    g.add_argument("--node_spacing", type=float, default=1.0, help="velocity lattice spacing (1.0: 25 nodes)")
    g.add_argument("--readout", default=None, choices=[None, "max", "moments"],
                   help="default: max for flowm, moments otherwise")
    g.add_argument("--readout_rel", default="off", choices=["on", "off"],
                   help="experimental (on): moment readout with observer-relative velocities v - a/dt "
                        "(Galilean invariant); off: absolute node velocities, as in the reported results")
    g.add_argument("--imprint_k", type=int, default=8, help="adaptive: observation imprint channels per node")
    g.add_argument("--imprint_gamma", type=float, default=0.9, help="adaptive: imprint decay per step")
    g.add_argument("--node_update", default="lk", choices=["lk", "lock"],
                   help="adaptive: existing LK update or snapshot lock-then-refine update")
    g.add_argument("--node_update_pred", default="on", choices=["on", "off"],
                   help="adaptive: update the nodes also from the model's own predictions (on, as in the "
                        "reported results) or only from observations (off, experimental)")
    g.add_argument("--dec_input", default="flowm", choices=["flowm", "zero"],
                   help="prediction-phase input: the previous prediction (FloWM) or zeros")
    # walls (experimental)
    g = p.add_argument_group("collision operators (experimental)")
    g.add_argument("--collision", default="none", choices=list(COLLISIONS),
                   help="bounce: bounce-back on walls | vmix: learned velocity mixing | mlp: learned collision")
    g.add_argument("--wall_mode", default="known", choices=["known", "learned"])
    g.add_argument("--bounce", default="rigid", choices=["rigid", "pointwise"], help="bounce-back operator")
    g.add_argument("--bounce_mode", default="directional", choices=["directional", "symmetric"],
                   help="pointwise bounce-back variant")
    g.add_argument("--wall_band", type=int, default=3)
    g.add_argument("--rigid_K", type=int, default=28)
    g.add_argument("--rigid_eps", type=float, default=1e-3)
    g.add_argument("--rigid_eps_rel", type=float, default=0.05)
    # optimisation
    g = p.add_argument_group("optimisation")
    g.add_argument("--steps", type=int, default=20000)
    g.add_argument("--bs", type=int, default=16)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--clip", type=float, default=1.0)
    g.add_argument("--tf", type=float, default=0.125, help="teacher forcing ratio")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--no_amp", dest="amp", action="store_false", help="disable bf16 autocast (CUDA)")
    g.add_argument("--grad_ckpt", action="store_true", help="checkpoint every recurrence step")
    g.add_argument("--accum", type=int, default=1, help="split --bs into this many micro-batches (same effective batch)")
    # io
    g = p.add_argument_group("logging and validation")
    g.add_argument("--out", default=None, help="run directory (default runs/<model>_s<seed>)")
    g.add_argument("--n_val", type=int, default=300, help="validation sequences")
    g.add_argument("--val_n_quick", type=int, default=300, help="sequences used by the periodic validation")
    g.add_argument("--val150_n", type=int, default=64,
                   help="sequences for the periodic 150-step validation (0 = off); best150.pt = best by it")
    g.add_argument("--val_every", type=int, default=1000)
    g.add_argument("--ckpt_every", type=int, default=1000)
    g.add_argument("--keep_ckpts", action="store_true", help="also keep ckpt_step<N>.pt at every --ckpt_every")
    g.add_argument("--log_every", type=int, default=50)
    g.add_argument("--device", default=None)
    return p


def parse_args(argv=None):
    """Two-pass parse: --preset changes the defaults; flags given explicitly still win."""
    p = get_parser()
    pre, _ = p.parse_known_args(argv)
    if pre.preset:
        p.set_defaults(**PRESETS[pre.preset])
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if a.out is None:
        a.out = os.path.join("runs", f"{a.model}_s{a.seed}")
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    amp = a.amp and device.startswith("cuda")

    cfg = env_cfg_from_args(a)                       # training distribution
    vcfg = env_cfg_from_args(a, split="val")         # quick-val / val cache distribution
    model = build_model(vars(a)).to(device)
    n_params = count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    print(f"[kwm] model={a.model} |V|={model.num_v} params={n_params} env={cfg.tag()} val={vcfg.tag()} device={device} amp={amp}")

    step0 = 0
    best = math.inf
    best150 = math.inf
    last = os.path.join(a.out, "last.pt")
    gen = torch.Generator().manual_seed(a.seed)
    if os.path.exists(last):
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step0, best = ck["step"], ck.get("best", math.inf)
        best150 = ck.get("best150", math.inf)
        gen.set_state(ck["gen_state"].cpu())          # ckpt was loaded with map_location=device
        random.setstate(ck["py_rng"])
        print(f"[kwm] resumed from step {step0}")
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(dict(vars(a), n_params=n_params, num_v=model.num_v, env=cfg.to_dict(), device=device), f, indent=1)

    cache = val_cache(vcfg, a.n_val, T_obs=a.t_obs, device=device)
    n_quick = min(a.val_n_quick, a.n_val)

    tr_f = open(os.path.join(a.out, "train.csv"), "a", newline="")
    va_f = open(os.path.join(a.out, "val.csv"), "a", newline="")
    tr_w, va_w = csv.writer(tr_f), csv.writer(va_f)
    if step0 == 0:
        tr_w.writerow(["step", "loss", "sec_step", "sec_data", "peak_mem_mb", "beta"])
        va_w.writerow(["step", "val_mse_tpred", "best", "val_mse150", "best150"])

    def save(path, step):
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, best=best, best150=best150,
                        args=vars(a), gen_state=gen.get_state(), py_rng=random.getstate()), path)

    model.train()
    t_acc, d_acc, l_acc, n_acc = 0.0, 0.0, 0.0, 0
    for step in range(step0 + 1, a.steps + 1):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        mb = a.bs // a.accum
        loss_sum, t_data = 0.0, 0.0
        for _ in range(a.accum):
            td = time.time()
            b = make_batch(mb, a.t_obs, a.t_pred, cfg, device, gen, split="train")
            t_data += time.time() - td
            obs, ain, aout, tgt = model_inputs(b["frames"], b["actions_noisy"], a.t_obs, a.t_pred)
            wall0 = make_wall0(b["digit_states"]["cam"][:, 0], a.world, a.wall_band) if model.collision_kind == "bounce" else None
            nodes = oracle_nodes_from(b["digit_states"]["vel"][:, 0]) if model.oracle else None
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=amp):
                pred = model(obs, ain, aout, target=tgt, tf_ratio=a.tf, wall0=wall0, nodes=nodes)
            loss = F.mse_loss(pred.float(), tgt)
            (loss / a.accum).backward()
            loss_sum += loss.item() / a.accum
        t1 = t0 + t_data
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t2 = time.time()
        t_acc += t2 - t0; d_acc += t1 - t0; l_acc += loss_sum; n_acc += 1
        if step % a.log_every == 0 or step == a.steps or step == step0 + 1:
            mem = torch.cuda.max_memory_allocated() / 2 ** 20 if device.startswith("cuda") else 0
            beta = model.log_beta.exp().item() if hasattr(model, "log_beta") else ""
            tr_w.writerow([step, l_acc / n_acc, t_acc / n_acc, d_acc / n_acc, round(mem), beta])
            tr_f.flush()
            print(f"step {step} loss {l_acc / n_acc:.5f} {t_acc / n_acc:.3f}s/step (data {d_acc / n_acc:.3f}s) mem {mem:.0f}MB", flush=True)
            t_acc, d_acc, l_acc, n_acc = 0.0, 0.0, 0.0, 0
        if step % a.val_every == 0 or step == a.steps:
            v = quick_val(model, cache, n_quick, a.t_pred, max(a.bs // a.accum, 4), device, amp, a.wall_band, T_obs=a.t_obs)
            if v < best:
                best = v
                save(os.path.join(a.out, "best.pt"), step)
            v150 = ""
            if a.val150_n:
                v150 = quick_val(model, cache, min(a.val150_n, a.n_val), VAL_T_PRED, max(a.bs // a.accum, 4), device, amp,
                                 a.wall_band, T_obs=a.t_obs)
                if v150 < best150:
                    best150 = v150
                    save(os.path.join(a.out, "best150.pt"), step)
            va_w.writerow([step, v, best, v150, best150 if a.val150_n else ""])
            va_f.flush()
            print(f"[val] step {step} mse{a.t_pred} {v:.5f} best {best:.5f}" + (f" mse150 {v150:.5f} best150 {best150:.5f}" if a.val150_n else ""), flush=True)
        if step % a.ckpt_every == 0 or step == a.steps:
            save(last, step)
            if a.keep_ckpts:
                save(os.path.join(a.out, f"ckpt_step{step}.pt"), step)
    tr_f.close(); va_f.close()
    print("[kwm] done")


if __name__ == "__main__":
    main()
