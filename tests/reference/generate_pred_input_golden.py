"""Regenerate the tiny CPU golden using the ACTUAL pre-PR3 sources.

Run from the repository root. The output embeds torch.save checkpoint bytes,
inputs and predictions from base kwm.eval.rollout; no current kwm code is used.
The isolated checkout and all temporary files stay under .pytest_cache/.
"""
import io
from pathlib import Path
import subprocess
import sys
import tarfile

BASE = "0c1198a048b4c0d7db310552efa2112bc4dfd9db"
ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "tests/reference/pre_pr3_predictions.npz"

if __name__ == "__main__":
    checkout = ROOT / ".pytest_cache/pre_pr3_golden_source"
    checkout.mkdir(parents=True, exist_ok=True)
    archive = subprocess.check_output(["git", "archive", BASE, "kwm"], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(checkout, filter="data")
    code = r'''
from pathlib import Path
import sys
import numpy as np
import torch
from kwm.model import build_model
from kwm.eval import load_model, rollout

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
g = torch.Generator().manual_seed(72)
obs = torch.rand(1, 5, 1, 8, 8, generator=g)
ain = torch.randn(1, 5, 2, generator=g)
aout = torch.randn(1, 3, 2, generator=g)
fr = torch.cat([obs, obs[:, :3]], 1)
an = torch.cat([ain, aout], 1)
data = dict(frames=fr.numpy(), actions=an.numpy(), base=np.array(sys.argv[2]))
for kind, rule in (("flowm", "lk"), ("spectral", "lk"), ("adaptive", "lk"), ("adaptive", "lock")):
    for mode in ("flowm", "zero"):
        key = f"{kind}_{rule}_{mode}"
        args = dict(model=kind, C=2, world=12, window=8, imprint_k=2,
                    node_update=rule, dec_input=mode)
        torch.manual_seed(71)
        model = build_model(args).eval()
        # Same checkpoint format and torch.save call as base kwm.train.
        checkpoint = dict(args=args, model=model.state_dict(), step=17)
        torch.save(checkpoint, "last.pt")
        data[key + "_checkpoint"] = np.frombuffer(Path("last.pt").read_bytes(), dtype=np.uint8)
        restored, _, _ = load_model(".", "last", "cpu")
        with torch.no_grad():
            pred, _ = rollout(restored, fr, an, torch.zeros(1, 2), 5, 3, 1., 3, False, "cpu", aux=True)
        data[key + "_predictions"] = pred.numpy()
np.savez_compressed(sys.argv[1], **data)
'''
    subprocess.run([sys.executable, "-c", code, str(DEST), BASE], cwd=checkout, check=True)
    print(DEST)
