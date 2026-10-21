"""Lucas-Kanade displacement estimator (used by the adaptive nodes)."""
import torch
import pytest

from kwm.transport import lk_displacement, spectral_shift, stream
from kwm.collision import lattice_nodes


def _blob(H, cx, cy, s=2.5):
    y = torch.arange(H).float()[:, None]
    x = torch.arange(H).float()[None, :]
    return torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s ** 2))


@pytest.mark.parametrize("d", [(0.3, -0.2), (-0.4, 0.1), (0.05, 0.37), (0.0, 0.0)])
def test_lk_one_step_recovers_subpixel_shift(d):
    """Memory streamed with node nu, observation at the true position nu + delta*: LK returns delta*."""
    H = 32
    nodes = lattice_nodes(1.0)
    i = int(((nodes - torch.tensor([1.0, -1.0])).abs().sum(1) == 0).nonzero())
    feats = torch.stack([_blob(H, 12.0, 16.0), 0.5 * _blob(H, 12.0, 16.0, 3.5)])      # C = 2
    h = torch.zeros(1, 25, 2, H, H)
    h[0, i] = feats
    hs = stream(h, nodes, torch.zeros(1, 2), 1.0)                                       # memory: moved by nu
    truth = spectral_shift(feats, nodes[i] + torch.tensor(d))[None]                    # observation: nu + delta*
    est = lk_displacement(hs, truth)[0, i]                                              # [2,H,W]
    m = feats[0] > 0.3                                                                  # textured pixels (blob core)
    err = (est[:, m].mean(1) - torch.tensor(d)).abs().max().item()
    assert err < 0.05, (d, est[:, m].mean(1))


def test_lk_untextured_is_zero():
    h = torch.full((1, 3, 2, 16, 16), 0.7)
    u = torch.full((1, 2, 16, 16), 0.2)                                                 # mismatch but no gradient
    assert lk_displacement(h, u).abs().max() < 1e-4


def test_lk_commutes_with_reflection():
    torch.manual_seed(0)
    h = torch.rand(1, 4, 3, 20, 20)
    u = torch.rand(1, 3, 20, 20)
    d = lk_displacement(h, u)
    dm = lk_displacement(h.flip(-1), u.flip(-1))                                        # mirror x
    assert torch.allclose(dm[:, :, 0], -d[:, :, 0].flip(-1), atol=1e-5)
    assert torch.allclose(dm[:, :, 1], d[:, :, 1].flip(-1), atol=1e-5)
