#!/usr/bin/env python3
"""CPU selection gate. Extra --variants are experiments, never kwm.eval modes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from kwm.diagnostics import (concentration_at, digit_masks, effective_count_at,
                             effective_count_summary, prediction_truth)
from kwm.eval import load_model
from kwm.lock_nodes import LockState, features
from kwm.provenance import source_hashes
from kwm.transport import _phase


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FrozenTransport:
    """Cache the fixed anchor FFT; preserve stream()'s phase and reduction order."""
    def __init__(self, model, anchor, activate=False):
        self.model, self.anchor, self.activate = model, anchor, activate
        feat = features(anchor)[:, :, :model.C]
        self.shape = feat.shape[-2:]
        self.spectrum = torch.fft.rfft2(feat.float())
        self.nodes = model.node_velocity(anchor)[..., 0, 0]
        if not isinstance(anchor, LockState) and model.kind == "adaptive":
            # Frozen decode needs features and carried velocity, not LK's large
            # spatial metadata/imprint planes. Preserve offsets()'s reduction
            # exactly, then carry it compactly; no recurrent step sees this state.
            offsets = model.cell.offsets(anchor)
            compact = torch.cat([offsets, offsets.new_zeros(*offsets.shape[:2], 2)], -1)
            self.anchor = LockState(feat, compact, feat.new_empty(0))

    def __call__(self, cumulative_action, elapsed):
        displacement = self.nodes * elapsed - cumulative_action[:, None]
        phase = _phase(*self.shape, displacement[:, :, None], self.nodes.device)
        feat = torch.fft.irfft2(self.spectrum * phase, s=self.shape)
        if self.activate:
            feat = self.model.cell.act(feat)
        return self.model.with_features(self.anchor, feat)


def wta_mask(feat):
    energy = feat.float().square().sum(2)
    return torch.zeros_like(energy).scatter_(1, energy.argmax(1, keepdim=True), 1.)


def oracle_mask(model, anchor, cache, n, t_obs):
    """Truth is used ONLY in this explicitly labelled script-local oracle."""
    feat = features(anchor)[:, :, :model.C]
    keep = wta_mask(feat)
    masks = digit_masks(cache["cent"][:n, t_obs - 1], cache["half"][:n],
                        cache["cam"][:n, t_obs - 1], model.world, model.window)
    nodes = model.node_velocity(anchor)[..., 0, 0]
    closest = (nodes[:, None] - cache["vel"][:n, t_obs - 1, :, None]).square().sum(-1).argmin(-1)
    for digit in range(masks.shape[1]):
        mask = masks[:, digit] & cache["visible"][:n, t_obs - 1, digit, None, None]
        onehot = feat.new_zeros(n, model.num_v).scatter_(1, closest[:, digit:digit + 1], 1.)
        win = keep[..., :model.window, :model.window]
        # As in the reviewer script, later digits win overlapping boxes.
        win.copy_(torch.where(mask[:, None], onehot[:, :, None, None], win))
    return keep


def observation_pass(model, frames, actions, ks):
    anchors = {}
    h = model.initial_state(frames[:, 0])
    for t in range(frames.shape[1]):
        if t == frames.shape[1] - 1:
            h, _, prelift = model.step(frames[:, t], h, actions[:, t], 1., None, return_prelift=True)
        else:
            h, _ = model.step(frames[:, t], h, actions[:, t], 1., None)
        for k in ks:
            if k and t == frames.shape[1] - k - 1:
                anchors[k] = h
    return h, prelift, anchors


class Measurements:
    def __init__(self, model, truth, target, observation_count):
        self.model, self.truth, self.target = model, truth, target
        self.mse, self.counts, self.concentration = [], [observation_count], {}

    def add(self, t, h, pred):
        self.mse.append((pred.clamp(0, 1) - self.target[:, t]).square().mean((1, 2, 3)))
        self.counts.append(effective_count_at(self.model, h))
        if t + 1 in (1, 10, 20):
            c = concentration_at(self.model, h, self.truth, t)
            c = c[torch.isfinite(c)]
            self.concentration[str(t + 1)] = float(c.mean()) if c.numel() else None

    def result(self):
        mse = torch.stack(self.mse, 1)
        full_mse = mse
        if mse.shape[1] < self.target.shape[1]:
            # Preserve the full rollout's row stride for prefix reductions.
            # A contiguous [B,20] mean can round differently from [B,150][:,:20]
            # even with identical per-frame errors. The cache/gate stays exact.
            full_mse = mse.new_zeros(mse.shape[0], self.target.shape[1])
            full_mse[:, :mse.shape[1]] = mse
            mse = full_mse[:, :mse.shape[1]]
        out = dict(mse_curve=full_mse.mean(0)[:mse.shape[1]].tolist(), node_concentration=self.concentration,
                   effective_node_count=effective_count_summary(torch.stack(self.counts, 1)))
        for horizon in (20, 50, 150):
            if horizon <= mse.shape[1]:
                out[f"mse{horizon}"] = float(mse[:, :horizon].mean())
                out[f"per_seq_mse{horizon}"] = mse[:, :horizon].mean(1).tolist()
        return out

    def prefix(self, n):
        out = Measurements(self.model, self.truth, self.target, self.counts[0])
        out.mse, out.counts = self.mse[:n], self.counts[:n + 1]
        out.concentration = {k: v for k, v in self.concentration.items() if int(k) <= n}
        return out


def recurrent(model, anchor, last_frame, actions, measurements, mode, steps):
    h, prev, at20, first = anchor, last_frame, None, None
    for t in range(steps):
        frame = torch.zeros_like(prev) if mode == "zero" else prev
        h, _ = model.step(frame, h, actions[:, t], 1., None)
        model._rel = actions[:, t] if model.readout_rel else None
        prev = model.decode(h)
        measurements.add(t, h, prev)
        if t == 0:
            first = prev
        if t == 19:
            at20 = h
    return at20, first


def frozen(model, anchor, actions, measurements, activate=False, start=0, mix_measurements=None):
    transport = FrozenTransport(model, anchor, activate)
    cumulative = torch.zeros_like(actions[:, 0])
    mix_first = None
    for t in range(start, actions.shape[1]):
        cumulative = cumulative + actions[:, t]
        h = transport(cumulative, float(t - start + 1))
        if t == start:
            # Guard the optimization against the public evaluation implementation.
            reference = model.transport_frozen(anchor, cumulative, 1., activate)
            assert torch.equal(features(h)[:, :, :model.C], features(reference)[:, :, :model.C]), "cached FFT transport changed values"
            assert torch.equal(model.node_velocity(h), model.node_velocity(reference)), "compact state changed velocities"
        model._rel = actions[:, t] if model.readout_rel else None
        measurements.add(t, h, model.decode(h))
        if mix_measurements is not None:
            mixed = model.with_features(h, model.cell.act(model.cell.mix(features(h)[:, :, :model.C])))
            pred = model.decode(mixed)
            mix_measurements.add(t, mixed, pred)
            if t == 0:
                mix_first = pred
    return mix_first


def reusable_baseline(path, common, mode):
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    # Old last.pt rows lack the pinned archive identity / new diagnostic and
    # must be refreshed once. Repeated sweeps reuse only exactly matched data.
    keys = ("checkpoint_sha256", "cache_sha256", "n", "t_obs", "t_pred", "seed", "threads")
    # These are the numerical dependencies of the shared observation/recurrent
    # path. Unrelated CLI, training and plotting edits do not invalidate scores;
    # retain the complete original provenance in baseline_source when reusing.
    files = tuple(f"kwm/{name}.py" for name in ("model", "cells", "transport", "collision", "lock_nodes", "diagnostics"))
    old_prov, new_prov = row.get("provenance", {}), common["provenance"]
    if (any(row.get(k) != common[k] for k in keys) or row.get("dec_input") != mode
            or row.get("node_update_pred") is not True or "effective_node_count" not in row
            or any(old_prov.get("files", {}).get(f) != new_prov["files"].get(f) for f in files)
            or old_prov.get("software") != new_prov.get("software")):
        return None
    return row


@torch.no_grad()
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--lock_ckpt", default="ckpt_step10000")
    p.add_argument("--lk_ckpt", default="ckpt_step13000")
    p.add_argument("--n", type=int, default=16, choices=(16, 32))
    p.add_argument("--threads", type=int, default=4, choices=range(1, 5))
    p.add_argument("--ks", type=int, nargs="+", default=[0, 4], help="explicit prelift sweep; default: 0 4")
    p.add_argument("--variants", action="store_true", help="also run the reviewer's script-only experiments")
    p.add_argument("--out", type=Path, default=ROOT / "results/pred_input_gate")
    p.add_argument("--baseline-dir", type=Path, help="reuse matching flowm/zero JSONs (default: --out)")
    p.add_argument("--refresh-baselines", action="store_true")
    a = p.parse_args(argv)
    if not a.out.resolve().is_relative_to(ROOT):
        p.error("--out must be inside this worktree")
    if any(k < 0 or k >= 30 for k in a.ks):
        p.error("--ks must be in [0, 30)")
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    cache = torch.load(a.cache, map_location="cpu", weights_only=False)
    if len(cache["frames"]) < a.n or cache["T_obs"] != 30 or cache["T_pred"] != 150:
        p.error("expected the reviewer's 30+150 selection cache with enough sequences")
    fr, an = cache["frames"][:a.n].float(), cache["actions_noisy"][:a.n]
    obs, ain, aout = fr[:, :30], an[:, :30], an[:, 30:180]
    truth = prediction_truth(cache, slice(0, a.n), 30, 150, "cpu")
    provenance, cache_hash = source_hashes(), sha256(a.cache)
    script_hash = sha256(__file__)
    rows, observations = [], []
    for name, which in (("pilot_lock", a.lock_ckpt), ("pilot_lk", a.lk_ckpt)):
        checkpoint_path = a.runs / name / f"{which}.pt"
        model, saved, step = load_model(a.runs / name, which, "cpu")
        if model.collision_kind != "none" or model.kind != "adaptive":
            p.error("this pilot gate expects adaptive, collision-free checkpoints")
        # Enabled for recurrent baselines; frozen rows ignore prediction node updates.
        model.node_update_pred = True
        common = dict(run=name, step=step, n=a.n, t_obs=30, t_pred=150,
                      role="selection", seed=12345, device="cpu", threads=a.threads, batch_size=a.n,
                      cache=str(a.cache), cache_sha256=cache_hash,
                      checkpoint=str(checkpoint_path), checkpoint_sha256=sha256(checkpoint_path),
                      provenance=provenance, script_sha256=script_hash)
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        hT, prelift, anchors = observation_pass(model, obs, ain, a.ks)
        count0 = effective_count_at(model, hT)
        fT = features(hT)[:, :, :model.C]
        # Reviewer statistic pools energy over the window BEFORE entropy; keep
        # it separate from the requested per-pixel effective_node_count.
        energy = fT[..., :model.window, :model.window].square().sum((2, 3, 4))
        share = energy / energy.sum(1, keepdim=True).clamp_min(1e-30)
        anchor_stats = dict(effective_node_count=float(count0.mean()),
                            window_pooled_effective_nodes=float((-(share * share.clamp_min(1e-30).log()).sum(1)).exp().mean()),
                            largest_node_share=float(share.max(1).values.mean()))
        if isinstance(hT, LockState):
            anchor_stats["locked_nodes_per_sample"] = float((hT.nodes[..., 3] > .5).float().sum(1).mean())
        observations.append(dict(run=name, wall_seconds=time.perf_counter() - start_wall,
                                 cpu_seconds=time.process_time() - start_cpu, **anchor_stats))
        common["observation_statistics"] = anchor_stats
        print(f"{name} step {step}: shared observation {observations[-1]['wall_seconds']:.1f}s", flush=True)

        def measurements():
            return Measurements(model, truth, fr[:, 30:180], count0)

        def save(label, stats, start, mode=None, k=None, **extra):
            row = dict(common, policy=label, dec_input=mode, frozen_k=k,
                       node_update_pred=mode in ("flowm", "zero"),
                       node_update_pred_ignored=mode not in ("flowm", "zero"),
                       experiment=mode is None, wall_seconds=time.perf_counter() - start[0],
                       cpu_seconds=time.process_time() - start[1], **stats, **extra)
            baselines = [r for r in rows if r["run"] == name and r["dec_input"] in ("flowm", "zero")]
            row["gate_pass"] = (None if mode in ("flowm", "zero") else
                                all(row[f"mse{h}"] <= min(r[f"mse{h}"] for r in baselines) for h in (20, 150)))
            rows.append(row)
            (a.out / f"{name}_{label}.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
            print(f"{name} {label:27s} " + " / ".join(f"{row[f'mse{h}']:.6f}" for h in (20, 50, 150))
                  + f" CPU {row['cpu_seconds']:.1f}s wall {row['wall_seconds']:.1f}s pass={row['gate_pass']}", flush=True)
            return row

        clock = lambda: (time.perf_counter(), time.process_time())
        zero_metrics, zero20, zero_first = None, None, None
        for mode in ("flowm", "zero"):
            start = clock()
            cached = None if a.refresh_baselines else reusable_baseline(
                (a.baseline_dir or a.out) / f"{name}_{mode}.json", common, mode)
            metrics = measurements()
            if cached is None:
                at20, first = recurrent(model, hT, obs[:, -1], aout, metrics, mode, 150)
                save(mode, metrics.result(), start, mode=mode, baseline_reused=False)
            else:
                if mode == "zero" and a.variants:
                    at20, first = recurrent(model, hT, obs[:, -1], aout, metrics, mode, 20)
                    prefix = metrics.result()
                    assert prefix["mse20"] == cached["mse20"], "cached zero MSE@20 changed"
                    assert prefix["per_seq_mse20"] == cached["per_seq_mse20"], "cached zero sequence errors changed"
                    assert prefix["mse_curve"] == cached["mse_curve"][:20], "cached zero frame errors changed"
                stats = {k: v for k, v in cached.items() if k.startswith(("mse", "per_seq_mse"))
                         or k in ("node_concentration", "effective_node_count")}
                save(mode, stats, start, mode=mode, baseline_reused=True,
                     baseline_source=dict(path=str((a.baseline_dir or a.out) / f"{name}_{mode}.json"),
                                          sha256=sha256((a.baseline_dir or a.out) / f"{name}_{mode}.json"),
                                          provenance=cached["provenance"]))
            if mode == "zero" and a.variants:
                zero_metrics, zero20, zero_first = metrics.prefix(20), at20, first

        start = clock()
        metrics, mix_metrics = measurements(), measurements() if a.variants else None
        mix_first = frozen(model, hT, aout, metrics, mix_measurements=mix_metrics)
        save("frozen", metrics.result(), start, mode="frozen",
             timing_group="frozen_and_mix_once" if a.variants else "frozen")
        if a.variants:
            # Compare unclamped predictions, not just the scalar MSE.
            exact = torch.equal(mix_first, zero_first)
            maxdiff = float((mix_first - zero_first).abs().max())
            assert maxdiff <= 1e-6, "mix-once no longer matches zero at tau=1"
            save("mix_once", mix_metrics.result(), start, timing_group="frozen_and_mix_once",
                 tau1_zero_bit_equal=exact, tau1_zero_max_abs=maxdiff)

        for k in sorted(set(a.ks)):
            start, metrics = clock(), measurements()
            if k == 0:
                anchor = model.with_features(hT, prelift)
            else:
                anchor = anchors[k]
                for t in range(30 - k, 30):
                    anchor, _ = model._step(torch.zeros_like(obs[:, -1]), anchor, ain[:, t], 1., None, update_nodes=False)
            frozen(model, anchor, aout, metrics, activate=True)
            save(f"frozen_prelift_k{k}", metrics.result(), start, mode="frozen_prelift", k=k)

        if a.variants:
            experiments = [("wta_postlift", fT * wta_mask(fT)[:, :, None], False),
                           ("wta_prelift", prelift * wta_mask(prelift)[:, :, None], True),
                           ("oracle_node_postlift", fT * oracle_mask(model, hT, cache, a.n, 30)[:, :, None], False)]
            if isinstance(hT, LockState):
                keep = (hT.nodes[..., 3] > .5).float()
                keep = torch.where(keep.sum(1, keepdim=True) > 0, keep, torch.ones_like(keep))
                experiments.append(("locked_nodes_only", fT * keep[:, :, None, None, None], False))
            for label, feat, activate in experiments:
                start, metrics = clock(), measurements()
                frozen(model, model.with_features(hT, feat), aout, metrics, activate=activate)
                save(label, metrics.result(), start)
            start = clock()
            frozen(model, zero20, aout, zero_metrics, start=20)
            save("zero20_then_frozen", zero_metrics.result(), start,
                 prefix_source="zero: first 20 predictions and recurrent state reused",
                 node_updates_first20=True)

    manifest = dict(provenance=provenance, script_sha256=script_hash, variants=a.variants,
                    ks=a.ks, n=a.n, threads=a.threads, observations=observations,
                    wall_seconds=time.perf_counter() - wall_start, cpu_seconds=time.process_time() - cpu_start,
                    baseline_reused=[r["run"] + ":" + r["policy"] for r in rows if r.get("baseline_reused")],
                    rows=[r["run"] + "_" + r["policy"] + ".json" for r in rows],
                    timing_note="Use these process totals; frozen/mix_once share a timing group, and zero20 reuses the zero prefix.")
    (a.out / "runtime.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"TOTAL: {manifest['cpu_seconds']/60:.2f} CPU-core min; {manifest['wall_seconds']/60:.2f} wall min", flush=True)


if __name__ == "__main__":
    main()
