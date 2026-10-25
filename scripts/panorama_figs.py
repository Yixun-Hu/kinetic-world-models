"""Scene-level figures on the validation set with real-valued velocities and camera motion (30 observed + 150
predicted frames, world 50, window 32, seed 12345). For two sequences in which a digit leaves the view during the observation and
re-enters during the prediction (out of view at the end of the observation), writes into results/figs/:
  panorama{sfx}.png       columns t in TIMES; rows: true world + viewport, true view, predicted views (FloWM, spectral,
                          adaptive nodes),
                          memory maps (max over velocity nodes of the feature norm) mapped back to world coordinates
                          with the known actions, overlaid with the true digit outlines and the viewport
  velocity_space{sfx}.png adaptive node velocities nu_i(t), t=0..29, vs the true digit velocities; the fixed lattice
  rollout{sfx}.gif        150 predicted frames: world + viewport | true view | adaptive nodes | FloWM, with t and MSE
sfx = '' for the first sequence, '_seq2' for the second.
Run directories (see docs/REPRODUCE.md): runs/flowm, runs/spectral, runs/adaptive.
usage: python scripts/panorama_figs.py [--device cuda] [--runs runs]"""
import argparse, json, math, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kwm.env import make_batch, _fourier_shift
from kwm.env import EnvConfig
from kwm.eval import load_model, node_velocity_error
from kwm.lock_nodes import features
from kwm.train import ROOT, VAL_SEED, VAL_T_PRED, val_cache

MODELS = [  # (key, label, run, ckpt)
    ("flowm", "FloWM (retrained)", "flowm", "best"),
    ("spectral", "Spectral", "spectral", "last"),
    ("adaptive", "Adaptive nodes", "adaptive", "best150"),
]
T_OBS, N_VAL, CHUNK = 30, 300, 50
TIMES = [0, 15, 29, 40, 60, 100, 150, 179]
DIGIT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
VIEW_COLOR = "#1baf7a"
INK, MUTED = "#1f1f1e", "#55554f"


def find_reentry(vis, T_obs):
    """vis [N,T,nd] -> (seq, digit, t_left, t_back) for out-of-view gaps of >= 8 frames of a digit that was already
    seen, starting by frame 40 and ending (re-entry) in the prediction (frames 30-120). Returns the longest gap and the
    longest gap among those that start during the observation (the digit is carried across the obs/pred boundary by
    memory alone), sorted by gap length."""
    gaps = []
    for i in range(vis.shape[0]):
        for d in range(vis.shape[2]):
            x = vis[i, :, d].tolist()
            t = 1
            while t < len(x):
                if not x[t] and any(x[:t]):
                    s0 = t
                    while t < len(x) and not x[t]:
                        t += 1
                    if t < len(x) and t - s0 >= 8 and T_obs <= t <= 120 and s0 <= 40:
                        gaps.append((i, d, s0, t))
                t += 1
    gaps.sort(key=lambda g: g[2] - g[3])
    first = gaps[0]
    during = [g for g in gaps if g[2] < T_obs and g[0] != first[0]]
    second = during[0] if during else next(g for g in gaps if g[0] != first[0])
    return [first, second]


@torch.no_grad()
def run_model(model, fr, an, T_obs, T_pred):
    """Re-implements KineticWM._forward (no walls, no collision) recording the memory map and the adaptive nodes.
    Returns preds [B,T_pred,1,win,win], mem [B,T,W,W] (ego frame), nodes [B,T_obs,V,2] or None."""
    model._dt = 1.0  # this helper always renders the unit-time rollout
    h = model.initial_state(fr[:, 0])
    mems, nodes, preds = [], [], []
    adaptive = model.kind == "adaptive"
    feat = lambda h: features(h)[:, :, :model.C].float().norm(dim=2).amax(1)
    for t in range(T_obs):
        h, _ = model.step(fr[:, t], h, an[:, t], 1.0, None)
        mems.append(feat(h))
        if adaptive:
            nodes.append(model.diagnostic_velocity(h))
    if adaptive and not model.node_update_pred:
        model.cell.update_nodes = False
    prev = fr[:, T_obs - 1]
    for k in range(T_pred):
        frame = torch.zeros_like(prev) if model.dec_input == "zero" else prev
        h, _ = model.step(frame, h, an[:, T_obs + k], 1.0, None, prediction=True)
        model._rel = an[:, T_obs + k] if model.readout_rel else None
        out = model.decode(h)
        preds.append(out); mems.append(feat(h)); prev = out
    if hasattr(model.cell, "update_nodes"):
        model.cell.update_nodes = True
    return torch.stack(preds, 1).clamp(0, 1), torch.stack(mems, 1), (torch.stack(nodes, 1) if adaptive else None)


def view_rects(ax, cam, W, win, color=VIEW_COLOR, lw=1.4):
    """Draw the (torus-wrapped) viewport whose top-left is cam=(x,y) on a WxW image axis."""
    from matplotlib.patches import Rectangle
    x0, y0 = float(cam[0]) % W, float(cam[1]) % W
    for dx in (0, -W):
        for dy in (0, -W):
            ax.add_patch(Rectangle((x0 + dx - 0.5, y0 + dy - 0.5), win, win, fill=False, ec=color, lw=lw, clip_on=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--runs", default=os.path.join(ROOT, "runs"), help="directory holding the run directories")
    a = ap.parse_args()
    dev = a.device
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.colors import LinearSegmentedColormap
    from PIL import Image, ImageDraw, ImageFont

    cfg = EnvConfig(real_velocity=True, real_camera=True)
    cache = val_cache(cfg, N_VAL, T_obs=T_OBS)
    W, win, T = cfg.world, cfg.window, T_OBS + VAL_T_PRED
    hits = find_reentry(cache["visible"].cpu(), T_OBS)
    seqs = hits[:2]
    print("re-entry candidates:", len(hits), "chosen:", seqs, flush=True)

    # regenerate the full world canvases of the chosen sequences (same generator stream as val_cache)
    gen = torch.Generator().manual_seed(VAL_SEED)
    worlds = {}
    for c0 in range(0, N_VAL, CHUNK):
        need = [s for s, *_ in seqs if c0 <= s < c0 + CHUNK]
        b = make_batch(min(CHUNK, N_VAL - c0), T_OBS, VAL_T_PRED, cfg, "cpu", gen, split="test", return_world=bool(need))
        for s in need:
            worlds[s] = b["world"][s - c0, :, 0]
            err = (b["frames"][s - c0] - cache["frames"][s].float()).abs().max().item()
            assert err < 1e-2, f"regenerated frames differ from the val cache ({err})"
        if len(worlds) == len(seqs):
            break

    idx = [s for s, *_ in seqs]
    fr = cache["frames"][idx].float().to(dev)
    an = cache["actions_noisy"][idx].to(dev)
    cam = cache["cam"][idx].float()                                            # [B,T,2] true camera
    cam_hat = cache["cam"][idx, 0].float()[:, None] + torch.cumsum(cache["actions_noisy"][idx].float(), 1)  # known actions
    gt = fr[:, T_OBS:].cpu()
    res = {}
    for key, label, run, ck in MODELS:
        model, args, step = load_model(os.path.join(a.runs, run), ck, dev)
        model._nodes_ovr = None
        p, mem, nodes = run_model(model, fr, an, T_OBS, VAL_T_PRED)
        ref_p = model(fr[:, :T_OBS], an[:, :T_OBS], an[:, T_OBS:]).clamp(0, 1)
        assert (ref_p - p).abs().max().item() < 1e-4, "custom rollout differs from model.forward"
        memw = _fourier_shift(mem.cpu(), cam_hat[:, :, None, :].expand(-1, -1, 1, -1)[:, :, 0]).clamp_min(0)   # ego -> world
        if key == "adaptive":
            node_err = node_velocity_error(model, cache, T_OBS, dev)
        res[key] = dict(label=label, step=step, pred=p.cpu(), mem=memw,
                        nodes=None if nodes is None else nodes.cpu(),
                        base_nodes=model.nodes().detach().float().cpu())
        print(key, run, ck, "ok", flush=True)
        del model; torch.cuda.empty_cache()

    out = os.path.join(ROOT, "results", "figs"); os.makedirs(out, exist_ok=True)
    cmap_mem = LinearSegmentedColormap.from_list("mem", ["#0b0b0a", "#7a2f10", "#eb6834", "#fbd9c8"])
    meta = {}
    for b, (s, d_re, t_left, t_back) in enumerate(seqs):
        sfx = "" if b == 0 else "_seq2"
        world = worlds[s]
        cent, vel = cache["cent"][s].float(), cache["vel"][s, 0].float()
        # check: memory (world coords) correlates with the true world at the end of the observation
        corr = {k: float(np.corrcoef(r["mem"][b, T_OBS - 1].flatten(), world[T_OBS - 1].flatten())[0, 1]) for k, r in res.items()}
        mse = {k: ((r["pred"][b] - gt[b]) ** 2).mean((1, 2, 3)) for k, r in res.items()}

        # ---------------- panorama ----------------
        rows = [("True world + viewport", "world"), ("True view", "gt")] + \
               [(f"Predicted view: {r['label']}", ("pred", k)) for k, r in res.items()] + \
               [(f"Memory (world frame): {r['label']}", ("mem", k)) for k, r in res.items()]
        nc, nr, cell = len(TIMES), len(rows), 1.42
        left, top = 2.05, 0.62
        fig = plt.figure(figsize=(left + nc * cell + 0.1, top + nr * cell + 0.42), dpi=150)
        Wf, Hf = fig.get_size_inches()
        fig.text(0.1 / Wf, 1 - 0.12 / Hf,
                 f"Validation sequence #{s}: digit {d_re + 1} is out of view for t={t_left}-{t_back - 1} and re-enters at t={t_back}  "
                 f"(observation t=0-29, prediction t=30-179)", fontsize=9.5, color=INK, weight="bold", va="top")
        scales = {k: float(np.quantile(r["mem"][b].numpy(), 0.995)) for k, r in res.items()}
        for c, t in enumerate(TIMES):
            for r_i, (lab, kind) in enumerate(rows):
                ax = fig.add_axes([(left + c * cell + 0.03) / Wf, 1 - (top + (r_i + 1) * cell - 0.03) / Hf,
                                   (cell - 0.06) / Wf, (cell - 0.06) / Hf])
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if kind == "world":
                    ax.imshow(world[t].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                    view_rects(ax, cam[b, t], W, win)
                    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(W - 0.5, -0.5)
                elif kind == "gt":
                    ax.imshow(fr[b, t, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                    for sp in ax.spines.values():
                        sp.set_visible(True); sp.set_color(VIEW_COLOR); sp.set_linewidth(1.4)
                elif kind[0] == "pred":
                    if t < T_OBS:
                        ax.set_facecolor("#efeee9")
                        ax.text(0.5, 0.5, "obs", ha="center", va="center", fontsize=8, color=MUTED, transform=ax.transAxes)
                    else:
                        img = res[kind[1]]["pred"][b, t - T_OBS, 0]
                        ax.imshow(img.numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                        ax.text(0.97, 0.03, f"{float(mse[kind[1]][t - T_OBS]):.3f}", transform=ax.transAxes, ha="right",
                                va="bottom", fontsize=6, color="white",
                                path_effects=[pe.withStroke(linewidth=1.6, foreground="black")])
                else:
                    k = kind[1]
                    ax.imshow(res[k]["mem"][b, t].numpy(), cmap=cmap_mem, vmin=0, vmax=scales[k], interpolation="nearest")
                    ax.contour(world[t].numpy(), levels=[0.35], colors="#ffffff", linewidths=0.45, alpha=0.75)
                    view_rects(ax, cam[b, t], W, win, lw=1.0)
                    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(W - 0.5, -0.5)
                if r_i == 0:
                    tl = f"t={t}" + (" (obs end)" if t == T_OBS - 1 else " (obs)" if t < T_OBS else f"  (+{t - T_OBS + 1})")
                    ax.set_title(tl, fontsize=7.5, color=INK if t >= T_OBS else MUTED, pad=2)
                if c == 0:
                    fig.text((left - 0.06) / Wf, 1 - (top + (r_i + 0.5) * cell) / Hf, lab.replace(": ", ":\n"),
                             ha="right", va="center", fontsize=7.5, color=INK)
        fig.text(0.1 / Wf, 0.1 / Hf,
                 "Viewport (32x32) in aqua; memory rows: max over velocity nodes of the feature norm, mapped ego->world with "
                 "the known actions; white = true digit outlines. Numbers = frame MSE.", fontsize=7, color=MUTED)
        fig.savefig(os.path.join(out, f"panorama{sfx}.png"), dpi=150); plt.close(fig)

        # ---------------- velocity space ----------------
        nod = res["adaptive"]["nodes"][b]                                          # [T_obs,V,2]
        visl = cache["visible"][s, T_OBS - 10:T_OBS].any(0)
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 5.6), dpi=150, gridspec_kw=dict(width_ratios=[1.6, 1]))
        lat = res["adaptive"]["base_nodes"]
        errs = []
        for ax_ in (ax, ax2):
            ax_.scatter(lat[:, 0], lat[:, 1], s=16, color="#b9b8ae", zorder=1, label="lattice nodes (25)")
        for d in range(vel.shape[0]):
            col = DIGIT_COLORS[d]
            dist = (nod[-1] - vel[d]).norm(dim=-1)
            i = int(dist.argmin())
            if visl[d]:
                errs.append(float(dist[i]))
            ax.plot(nod[:, i, 0], nod[:, i, 1], color=col, lw=1.6, zorder=2)
            ax.scatter(nod[-1, i, 0], nod[-1, i, 1], s=36, color=col, zorder=3, edgecolors="white", linewidths=0.8)
            mk = dict(marker="x", s=70, linewidths=2.2, color=col, zorder=4)
            ax.scatter(vel[d, 0], vel[d, 1], **mk)
            ax2.scatter(vel[d, 0], vel[d, 1], **mk)
            j = int((lat - vel[d]).norm(dim=-1).argmin())
            ax2.plot([vel[d, 0], lat[j, 0]], [vel[d, 1], lat[j, 1]], color=col, lw=1, ls=(0, (2, 2)), zorder=2)
            tag = f"digit {d + 1}" + ("" if visl[d] else " (not in view, last 10 obs)")
            ax.text(vel[d, 0] + 0.09, vel[d, 1] + 0.09, tag, fontsize=7, color=INK)
        q = [float((lat - vel[d]).norm(dim=-1).min()) for d in range(vel.shape[0]) if visl[d]]
        for ax_, ttl in ((ax, f"Adaptive nodes: node velocities nu_i(t), t=0-29 (line), t=29 (dot)\n"
                               f"node velocity error, median {np.median(errs):.3f} this sequence, "
                               f"{node_err['median']:.3f} over {node_err['n']} digits"),
                         (ax2, f"Spectral: fixed lattice (cannot move)\n"
                               f"distance to the nearest node, median {np.median(q):.3f} this sequence")):
            ax_.set_xlim(-2.5, 2.5); ax_.set_ylim(-2.5, 2.5); ax_.set_aspect("equal")
            ax_.set_xlabel("v_x (px/frame)", fontsize=8.5, color=INK); ax_.set_ylabel("v_y (px/frame)", fontsize=8.5, color=INK)
            ax_.set_title(ttl, fontsize=8.5, color=INK, loc="left")
            ax_.grid(True, color="#e4e3dc", lw=0.5); ax_.tick_params(labelsize=7.5, colors=MUTED)
            for sp in ax_.spines.values():
                sp.set_color("#b9b8ae")
        ax.scatter([], [], marker="x", color=MUTED, label="true digit velocity")
        ax.legend(fontsize=7, frameon=False, loc="lower left")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"velocity_space{sfx}.png"), dpi=150); plt.close(fig)

        # ---------------- GIF ----------------
        S, G, LAB, FOOT = 210, 14, 24, 30
        font_p = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
        f1, f2 = ImageFont.truetype(font_p, 13), ImageFont.truetype(font_p, 14)
        heads = ["True world + viewport", "True view", "Adaptive nodes", "FloWM (retrained)"]
        Wg = 4 * S + 5 * G
        frames_gif = []
        to_img = lambda x, n: Image.fromarray((x.clamp(0, 1).numpy() * 255).astype(np.uint8)).convert("RGB").resize((S, S), Image.NEAREST)
        for k in range(VAL_T_PRED):
            t = T_OBS + k
            im = Image.new("RGB", (Wg, LAB + S + FOOT), (252, 252, 251))
            dr = ImageDraw.Draw(im)
            tiles = [to_img(world[t], W), to_img(gt[b, k, 0], win), to_img(res["adaptive"]["pred"][b, k, 0], win),
                     to_img(res["flowm"]["pred"][b, k, 0], win)]
            # viewport on the world tile (torus-wrapped)
            dw = ImageDraw.Draw(tiles[0]); sc = S / W
            x0, y0 = float(cam[b, t, 0]) % W, float(cam[b, t, 1]) % W
            for dx in (0, -W):
                for dy in (0, -W):
                    dw.rectangle([(x0 + dx) * sc, (y0 + dy) * sc, (x0 + dx + win) * sc, (y0 + dy + win) * sc],
                                 outline=(27, 175, 122), width=2)
            for j_, tile in enumerate(tiles):
                x = G + j_ * (S + G)
                im.paste(tile, (x, LAB))
                dr.text((x, 4), heads[j_], fill=(31, 31, 30), font=f1)
            dr.rectangle([G + S + G - 2, LAB - 2, G + 2 * S + G + 1, LAB + S + 1], outline=(27, 175, 122), width=2)
            dr.text((G, LAB + S + 7), f"t = {t}  (prediction frame {k + 1}/150)     frame MSE:  adaptive nodes {float(mse['adaptive'][k]):.3f}"
                                      f"   FloWM (retrained) {float(mse['flowm'][k]):.3f}", fill=(31, 31, 30), font=f2)
            frames_gif.append(np.asarray(im))
        # fixed palette: 32 grey levels + viewport colour + background + text (keeps each GIF under 4 MB)
        cols = [(round(255 * i / 31),) * 3 for i in range(32)] + [(27, 175, 122), (252, 252, 251), (31, 31, 30)]
        pal = Image.new("P", (1, 1))
        pal.putpalette([c for rgb in cols for c in rgb] + [0] * (768 - 3 * len(cols)))
        ims = [Image.fromarray(f).quantize(palette=pal, dither=Image.Dither.NONE) for f in frames_gif]
        ims[0].save(os.path.join(out, f"rollout{sfx}.gif"), save_all=True, append_images=ims[1:], duration=100,
                    loop=0, optimize=True)
        meta[f"seq{sfx or '1'}"] = dict(seq=s, reentry_digit=d_re + 1, t_left=t_left, t_back=t_back, mem_world_corr_t29=corr,
                                        node_velocity_error_seq_median=float(np.median(errs)),
                                        node_velocity_error_median_64seq=node_err["median"],
                                        mse20={k: float(v[:20].mean()) for k, v in mse.items()},
                                        mse150={k: float(v.mean()) for k, v in mse.items()})
        print(sfx or "seq1", json.dumps(meta[f"seq{sfx or '1'}"]), flush=True)
    meta["models"] = {k: dict(run=run, ckpt=ck) for k, _, run, ck in MODELS}
    json.dump(meta, open(os.path.join(out, "panorama_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
