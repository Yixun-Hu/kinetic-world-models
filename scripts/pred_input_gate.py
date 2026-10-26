#!/usr/bin/env python3
"""CPU pilot gate, matching the reviewer's prefix-MSE selection-cache protocol."""
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

from kwm.diagnostics import prediction_truth, concentration_summary
from kwm.eval import load_model, configure_prediction, prediction_metadata
from kwm.provenance import source_hashes


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--lock_ckpt", default="last", help="lock checkpoint basename; pin an archive to reproduce a step")
    p.add_argument("--lk_ckpt", default="last", help="LK checkpoint basename; pin an archive to reproduce a step")
    p.add_argument("--n", type=int, default=32, choices=(16, 32))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", type=Path, default=ROOT / "results/pred_input_gate")
    a = p.parse_args(argv)
    if not a.out.resolve().is_relative_to(ROOT):
        p.error("--out must be inside this worktree")
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    cache = torch.load(a.cache, map_location="cpu", weights_only=False)
    if len(cache["frames"]) < a.n or cache["T_obs"] != 30 or cache["T_pred"] != 150:
        p.error("expected the reviewer's 30+150 selection cache with enough sequences")
    fr = cache["frames"][:a.n].float()
    an = cache["actions_noisy"][:a.n]
    truth = prediction_truth(cache, slice(0, a.n), 30, 150, "cpu")
    provenance = source_hashes()
    cache_hash = sha256(a.cache)
    rows = []
    for name in ("pilot_lock", "pilot_lk"):
        which = a.lock_ckpt if name == "pilot_lock" else a.lk_ckpt
        checkpoint_path = a.runs / name / f"{which}.pt"
        model, saved, step = load_model(a.runs / name, which, "cpu")
        checkpoint_hash = sha256(checkpoint_path)
        for mode, k in [("flowm", 1), ("zero", 1), ("frozen", 1)] + [
                ("frozen_prelift", k) for k in (0, 1, 2, 4)]:
            configure_prediction(model, saved, mode, k)
            # Reviewer's baseline explicitly enables prediction-time node updates.
            model.node_update_pred = True
            wall_start, cpu_start = time.perf_counter(), time.process_time()
            with torch.no_grad():
                pred, aux = model(fr[:, :30], an[:, :30], an[:, 30:180], digit_truth=truth)
                mse = (pred.clamp(0, 1) - fr[:, 30:180]).square().mean((2, 3, 4))
            row = dict(run=name, step=step, n=a.n, t_obs=30, t_pred=150,
                       role="selection", seed=12345, device="cpu", threads=a.threads,
                       batch_size=a.n, cache=str(a.cache), cache_sha256=cache_hash,
                       checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_hash,
                       provenance=provenance, **prediction_metadata(model),
                       wall_seconds=time.perf_counter() - wall_start,
                       cpu_seconds=time.process_time() - cpu_start,
                       node_concentration=concentration_summary(aux["node_concentration"]),
                       mse_curve=mse.mean(0).tolist())
            for horizon in (20, 50, 150):
                row[f"mse{horizon}"] = float(mse[:, :horizon].mean())
                row[f"per_seq_mse{horizon}"] = mse[:, :horizon].mean(1).tolist()
            if mode.startswith("frozen"):
                baselines = [r for r in rows if r["run"] == name and r["dec_input"] in ("flowm", "zero")]
                row["gate_pass"] = all(row[f"mse{h}"] <= min(r[f"mse{h}"] for r in baselines)
                                       for h in (20, 150))
            else:
                row["gate_pass"] = None
            rows.append(row)
            label = mode + (f"_k{k}" if mode == "frozen_prelift" else "")
            (a.out / f"{name}_{label}.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
            print(f"{name} {label:20s} " + " ".join(f"MSE@{h} {row[f'mse{h}']:.6f}" for h in (20, 50, 150))
                  + f" CPU {row['cpu_seconds']:.1f}s wall {row['wall_seconds']:.1f}s pass={row['gate_pass']}", flush=True)
    print(f"Total CPU seconds: {sum(r['cpu_seconds'] for r in rows):.1f}", flush=True)


if __name__ == "__main__":
    main()
