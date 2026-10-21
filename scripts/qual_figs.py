"""Comparison figures on the validation set with real-valued velocities and camera motion (periodic world,
300 sequences, seed 12345, 30 observed + 150 predicted frames, 32x32 window):
  results/figs/qual_rollout.png    2 sequences x (GT + 5 models) x predicted frames 1,5,10,20,50,100,150
  results/figs/mse_vs_horizon.png  per-frame MSE (1-150), log y, all-zero / observation-mean baselines
  results/figs/qual_table.json     MSE@20 / MSE@150 (mean +- 95% CI) per model and baseline
Run directories (see docs/REPRODUCE.md): runs/flowm_official_real_both, runs/flowm, runs/spectral, runs/adaptive,
runs/oracle. The adaptive model uses best150.pt if the run has one, else the ckpt_step*.pt with the lowest
150-step MSE on the first 64 validation sequences (the rule train.py uses for best150.pt).
usage: python scripts/qual_figs.py [--device cuda] [--runs runs]"""
import argparse, glob, json, os, re, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kwm.eval import load_model, rollout, ci95
from kwm.env import EnvConfig
from kwm.train import ROOT, VAL_T_PRED, val_cache, quick_val

MODELS = [  # (label, run, ckpt)
    ("FloWM (published weights)", "flowm_official_real_both", "best"),
    ("FloWM (retrained)", "flowm", "best"),
    ("Spectral", "spectral", "last"),
    ("Adaptive nodes", "adaptive", "best150"),
    ("Oracle nodes", "oracle", "best"),
]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # categorical slots 1-5, fixed order
FRAMES = [1, 5, 10, 20, 50, 100, 150]
T_OBS, N_VAL, N150 = 30, 300, 64


def pick_best150(run_dir, cache, device):
    if os.path.exists(os.path.join(run_dir, "best150.pt")):
        return "best150", None
    scores = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "ckpt_step*.pt"))):
        ck = os.path.basename(p)[:-3]
        m, args, step = load_model(run_dir, ck, device)
        scores[step] = (ck, quick_val(m, cache, N150, VAL_T_PRED, 16, device, True, args.get("wall_band", 3), T_obs=T_OBS))
        print("best150 scan", ck, f"{scores[step][1]:.5f}", flush=True)
    best = min(scores.values(), key=lambda x: x[1])
    return best[0], {str(k): v[1] for k, v in sorted(scores.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=50)
    ap.add_argument("--runs", default=os.path.join(ROOT, "runs"), help="directory holding the run directories")
    a = ap.parse_args()
    dev = a.device
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    cfg = EnvConfig(real_velocity=True, real_camera=True)
    cache = val_cache(cfg, N_VAL, T_obs=T_OBS)
    gt = cache["frames"][:, T_OBS:T_OBS + VAL_T_PRED].float()                 # [N,150,1,32,32]
    obs = cache["frames"][:, :T_OBS].float()
    out = os.path.join(ROOT, "results", "figs"); os.makedirs(out, exist_ok=True)
    meta = dict(val_cache=f"{cfg.tag()} n{N_VAL} T{T_OBS}+{VAL_T_PRED}", models={})

    curves, preds_show, table = {}, {}, {}
    # two fixed sequences: the first two whose GT window has ink in every shown frame
    ink = (gt[:, [f - 1 for f in FRAMES]].amax((2, 3, 4)) > 0.3).all(1)
    show = [int(i) for i in torch.nonzero(ink)[:2, 0]]
    for label, run, ck in MODELS:
        rd = os.path.join(a.runs, run)
        scan = None
        if ck == "best150" and not os.path.exists(os.path.join(rd, "best150.pt")):
            ck, scan = pick_best150(rd, cache, dev)
        model, args, step = load_model(rd, ck, dev)
        mse = []
        for i in range(0, N_VAL, a.bs):
            sl = slice(i, i + a.bs)
            fr = cache["frames"][sl].float().to(dev)
            p = rollout(model, fr, cache["actions_noisy"][sl].to(dev), cache["cam"][sl, 0].to(dev), T_OBS, VAL_T_PRED,
                        1.0, args.get("wall_band", 3), False, dev, vel0=cache["vel"][sl, 0]).clamp(0, 1).cpu()
            mse.append(((p - gt[sl]) ** 2).mean((2, 3, 4)))
            for j in show:
                if i <= j < i + a.bs:
                    preds_show[(label, j)] = p[j - i]
        mse = torch.cat(mse)                                                  # [N,150]
        curves[label] = mse.mean(0).numpy()
        table[label] = dict(run=run, ckpt=ck, step=step, mse20=ci95(mse[:, :20].mean(1).numpy()),
                            mse150=ci95(mse.mean(1).numpy()))
        meta["models"][label] = dict(run=run, ckpt=ck, step=step, best150_scan=scan)
        print(label, run, ck, step, table[label]["mse20"], table[label]["mse150"], flush=True)
        del model; torch.cuda.empty_cache()
    zero = (gt ** 2).mean((2, 3, 4))
    omean = ((obs.mean(1, keepdim=True) - gt) ** 2).mean((2, 3, 4))
    base = {"All-zero": zero, "Observation mean": omean}
    for k, v in base.items():
        curves[k] = v.mean(0).numpy()
        table[k] = dict(mse20=ci95(v[:, :20].mean(1).numpy()), mse150=ci95(v.mean(1).numpy()))

    # ---------------- qual_rollout.png ----------------
    rows = ["Ground truth"] + [m[0] for m in MODELS]
    nc, nr = len(FRAMES), len(rows)
    cell = 0.86
    fig = plt.figure(figsize=(1.75 + 2 * nc * cell + 0.3, 0.55 + nr * cell + 0.25), dpi=150)
    W, H = fig.get_size_inches()
    for s, j in enumerate(show):
        x0 = 1.75 + s * (nc * cell + 0.3)
        fig.text((x0 + nc * cell / 2) / W, 1 - 0.14 / H, f"Validation sequence #{j}", ha="center", va="top",
                 fontsize=9.5, color="#1f1f1e", weight="bold")
        for r, lab in enumerate(rows):
            for c, f in enumerate(FRAMES):
                ax = fig.add_axes([(x0 + c * cell + 0.02) / W, 1 - (0.55 + (r + 1) * cell - 0.02) / H,
                                   (cell - 0.04) / W, (cell - 0.04) / H])
                img = gt[j, f - 1, 0] if r == 0 else preds_show[(lab, j)][f - 1, 0]
                ax.imshow(img.numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if r > 0:
                    e = float(((img - gt[j, f - 1, 0]) ** 2).mean())
                    ax.text(0.97, 0.03, f"{e:.3f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6,
                            color="white", path_effects=[pe.withStroke(linewidth=1.6, foreground="black")])
                if r == 0:
                    ax.set_title(f"t+{f}", fontsize=8, color="#55554f", pad=2)
                if s == 0 and c == 0:
                    fig.text((x0 - 0.08) / W, 1 - (0.55 + (r + 0.5) * cell) / H, lab, ha="right", va="center",
                             fontsize=8, color="#1f1f1e")
    fig.savefig(os.path.join(out, "qual_rollout.png"), dpi=150); plt.close(fig)

    # ---------------- mse_vs_horizon.png ----------------
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=150)
    t = np.arange(1, VAL_T_PRED + 1)
    for (label, _, _), col in zip(MODELS, COLORS):
        ax.plot(t, curves[label], color=col, lw=2, label=label, solid_capstyle="round")
    for k, ls in (("All-zero", (0, (5, 3))), ("Observation mean", (0, (1.5, 2.5)))):
        y = float(curves[k].mean())
        ax.axhline(y, color="#8a897f", lw=1.2, ls=ls, zorder=0)
        ax.text(VAL_T_PRED + 1.5, y, f"{k} {y:.3f}", va="center", ha="left", fontsize=7.5, color="#55554f")
    ax.set_yscale("log")
    ax.set_xlim(1, VAL_T_PRED); ax.set_xticks(FRAMES)
    ax.set_xlabel("Prediction horizon (frames after the last observation)", fontsize=9, color="#1f1f1e")
    ax.set_ylabel("Per-frame MSE (log scale)", fontsize=9, color="#1f1f1e")
    ax.set_title(f"MSE vs horizon: real velocities and camera motion, {N_VAL} sequences, {T_OBS} observed frames",
                 fontsize=10, color="#1f1f1e", loc="left")
    ax.grid(True, which="major", color="#e4e3dc", lw=0.6); ax.grid(True, which="minor", color="#efeee9", lw=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#b9b8ae")
    ax.tick_params(colors="#55554f", labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="lower right", ncol=2, labelcolor="#1f1f1e")
    fig.subplots_adjust(right=0.78)
    fig.savefig(os.path.join(out, "mse_vs_horizon.png"), dpi=150); plt.close(fig)

    meta["table"] = table
    meta["curves_at"] = {k: [float(v[f - 1]) for f in FRAMES] for k, v in curves.items()}
    meta["shown_sequences"] = show
    json.dump(meta, open(os.path.join(out, "qual_table.json"), "w"), indent=1)
    print(json.dumps(meta["table"], indent=1))


if __name__ == "__main__":
    main()
