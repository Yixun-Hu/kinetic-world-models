#!/usr/bin/env python3
"""PR6 CPU forward/backward cost: batch four, six observations, four predictions."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from kwm.model import KineticWM, count_params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/pr6_c4/cost.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(2024)
    obs = torch.rand(4, 6, 1, 32, 32, generator=g)
    ain, aout = torch.randn(4, 6, 2, generator=g), torch.randn(4, 4, 2, generator=g)
    target = torch.rand(4, 4, 1, 32, 32, generator=g)
    rows = []
    for update in ("lk", "lock"):
        for rot in ("none", "c4"):
            torch.manual_seed(0)
            model = KineticWM(C=32, rot=rot, node_update=update).train()
            times = []
            for step in range(7):
                model.zero_grad(set_to_none=True)
                start = time.perf_counter()
                pred = model(obs, ain, aout)
                (pred - target).square().mean().backward()
                elapsed = time.perf_counter() - start
                if step >= 2:
                    times.append(elapsed)
            row = dict(rot=rot, node_update=update, C=32, parameters=count_params(model),
                       median_seconds=statistics.median(times), mean_seconds=statistics.mean(times), samples=times)
            rows.append(row)
            print(json.dumps(row), flush=True)
            del model, pred
    counts = [dict(C=C, rot=rot, parameters=count_params(KineticWM(C=C, rot=rot)))
              for C in (32, 64) for rot in ("none", "c4")]
    result = dict(batch_size=4, threads=1, warmup=2, measured_steps=5, observation_steps=6,
                  prediction_steps=4, world=50, window=32, dtype="fp32", torch_version=torch.__version__,
                  rows=rows, counts=counts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
