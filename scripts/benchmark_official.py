"""Benchmark the published FloWM 2D checkpoint (docs/RESULTS.md, section 1).

Downloads FloWM's MNIST World checkpoint (Hugging Face flowm123/mnistworld-models,
mnistworld/flowm-fernn/dynamic_po_flowm.pt), wraps it as a run directory of the `flowm` model
(runs/flowm_official_<data>/best.pt; the parameter names match, strict load), and evaluates it with
kwm.eval on four data settings: integer (FloWM's own), real velocities, real camera motion, both.
The checkpoint is read with torch.load(weights_only=True): tensors only, no pickled code.

usage: python scripts/benchmark_official.py [--ckpt PATH] [--data integer real_velocity real_camera real_both]
                                            [--n_val 300] [--skip_eval]
"""
import argparse
import hashlib
import os
import subprocess
import sys
import urllib.request

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from kwm.train import parse_args          # noqa: E402
from kwm.model import build_model         # noqa: E402

URL = ("https://huggingface.co/flowm123/mnistworld-models/resolve/main/"
       "mnistworld/flowm-fernn/dynamic_po_flowm.pt")
DATA = {"integer": [], "real_velocity": ["--real_velocity"], "real_camera": ["--real_camera"],
        "real_both": ["--real_velocity", "--real_camera"]}


def wrap(src, out, data_flags, n_val):
    ck = torch.load(src, map_location="cpu", weights_only=True)
    args = vars(parse_args(["--model", "flowm", "--preset", "flowm", "--n_val", str(n_val)] + data_flags))
    m = build_model(args)
    m.load_state_dict(ck["model_state"], strict=True)
    os.makedirs(out, exist_ok=True)
    torch.save(dict(model=m.state_dict(), step=int(ck["epoch"]), args=args, source=src,
                    official_val_loss=float(ck["val_loss"])), os.path.join(out, "best.pt"))
    print("wrote", out, "epoch", ck["epoch"], "official val_loss", float(ck["val_loss"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(ROOT, "ext", "flowm_official", "dynamic_po_flowm.pt"))
    p.add_argument("--data", nargs="+", default=list(DATA), choices=list(DATA))
    p.add_argument("--n_val", type=int, default=300)
    p.add_argument("--skip_eval", action="store_true")
    a = p.parse_args()
    if not os.path.exists(a.ckpt):
        os.makedirs(os.path.dirname(a.ckpt), exist_ok=True)
        print("downloading", URL)
        urllib.request.urlretrieve(URL, a.ckpt)
    print("sha256", hashlib.sha256(open(a.ckpt, "rb").read()).hexdigest())
    for d in a.data:
        run = os.path.join(ROOT, "runs", f"flowm_official_{d}")
        wrap(a.ckpt, run, DATA[d], a.n_val)
        if not a.skip_eval:
            subprocess.run([sys.executable, "-m", "kwm.eval", "--run", run, "--ckpt", "best", "--n_val", str(a.n_val),
                            "--skip_tp"], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
