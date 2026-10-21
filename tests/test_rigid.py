"""Rigid-contact-propagation bounce-back (walls, experimental)."""
import torch
import pytest

from kwm.collision import (lattice_nodes, reflection_perms, rigid_bounce_back, bounce_back,
                           known_wall_field)
from kwm.transport import stream

W = 24
LO = 1                     # env: leftmost ink column confined to >= 1 (wall line at col 0)
BAND = LO + 1              # contact band = wall pixel + 1 -> exact match for |v| = 1


def _node(nodes, v):
    return int(((nodes - torch.tensor(v)).abs().sum(1) < 1e-6).nonzero())


def _rect(e, top=8, w=4, hgt=6):
    img = torch.zeros(W, W)
    img[top:top + hgt, e:e + w] = 1.0
    img[top + 1, e] = 0.5          # asymmetric texture: a mirror image would differ
    return img


def _env_truth(e0, v0, T):
    """Fold-formula trajectory of the left bbox edge (kwm.env bbox rule, x only)."""
    e, v, out = float(e0), float(v0), []
    for _ in range(T):
        e = e + v
        if e < LO:
            e, v = 2 * LO - e, -v
        out.append((int(round(e)), v))
    return out


def _simulate(op, e0=6, v0=-1.0, T=14):
    nodes = lattice_nodes(1.0)
    px, py = reflection_perms(nodes)
    V = nodes.shape[0]
    h = torch.zeros(1, V, 2, W, W)
    i = _node(nodes, [v0, 0.0])
    h[0, i, 0] = _rect(e0)
    h[0, i, 1] = 0.3 * _rect(e0)
    wall = known_wall_field(W, band=BAND)[None]
    wall[:, 1:] = 0                # only the left wall matters here
    a = torch.zeros(1, 2)
    frames = []
    for _ in range(T):
        h = op(h, nodes, px, py, wall)
        h = stream(h, nodes, a, 1.0)
        frames.append(h.clone())
    return frames, nodes


def test_rigid_matches_env_truth_and_pointwise_folds():
    T = 14
    truth = _env_truth(6, -1.0, T)
    rigid, nodes = _simulate(lambda h, n, px, py, w: rigid_bounce_back(h, n, px, py, w, K=28))
    point, _ = _simulate(lambda h, n, px, py, w: bounce_back(h, n, px, py, w, "directional"))
    max_rigid_err, max_point_err = 0.0, 0.0
    for t, (e, v) in enumerate(truth):
        gt = torch.zeros_like(rigid[t][0, :, 0])
        gt[_node(nodes, [v, 0.0])] = _rect(e)                              # all mass in the true velocity group
        max_rigid_err = max(max_rigid_err, (rigid[t][0, :, 0] - gt).abs().max().item())
        max_point_err = max(max_point_err, (point[t][0, :, 0] - gt).abs().max().item())
        # spatial shape (sum over V): rigid keeps the (unmirrored) rectangle
        shape_err = (rigid[t][0, :, 0].sum(0) - _rect(e)).abs().max().item()
        assert shape_err < 1e-4, (t, shape_err)
    assert any(v > 0 for _, v in truth), "test must include a bounce"
    assert max_rigid_err < 1e-4
    assert max_point_err > 0.5          # pointwise: leading edge reverses first -> folding


def test_pointwise_spatially_folds():
    point, nodes = _simulate(lambda h, n, px, py, w: bounce_back(h, n, px, py, w, "directional"))
    truth = _env_truth(6, -1.0, 14)
    errs = [(point[t][0, :, 0].sum(0) - _rect(e)).abs().sum().item() for t, (e, _) in enumerate(truth)]
    assert max(errs) > 3.0              # shape (not only velocity labels) is destroyed


@pytest.mark.parametrize("nodes_kind", ["lattice", "stretched"])
def test_rigid_identity_conservation_mirror(nodes_kind):
    torch.manual_seed(0)
    if nodes_kind == "lattice":
        nodes = lattice_nodes(0.5)
        px, py = reflection_perms(nodes)
    else:
        nodes = lattice_nodes(1.0) * torch.tensor([1.3, 0.7])         # off-lattice, reflection-closed
        px, py = reflection_perms(nodes)
    V = nodes.shape[0]
    h = torch.relu(torch.randn(2, V, 3, 16, 16) - 1.0)                       # sparse blobs
    wall = torch.rand(2, 4, 16, 16) * (torch.rand(2, 4, 16, 16) > 0.7)
    out = rigid_bounce_back(h, nodes, px, py, wall, K=6)
    # identity without walls
    assert (rigid_bounce_back(h, nodes, px, py, torch.zeros_like(wall), K=6) - h).abs().max() == 0
    # mass conservation over V
    assert (out.sum(1) - h.sum(1)).abs().max() < 1e-5
    # commutes with the x mirror (flip x, relabel by R_x, swap left/right walls)
    hm, wm = h.flip(-1)[:, px], wall.flip(-1)[:, [1, 0, 2, 3]]
    om = rigid_bounce_back(hm, nodes, px, py, wm, K=6)
    assert (om - out.flip(-1)[:, px]).abs().max() < 1e-5
    # and with the y mirror
    hm, wm = h.flip(-2)[:, py], wall.flip(-2)[:, [0, 1, 3, 2]]
    om = rigid_bounce_back(hm, nodes, px, py, wm, K=6)
    assert (om - out.flip(-2)[:, py]).abs().max() < 1e-5


def test_rigid_translation_equivariant():
    torch.manual_seed(1)
    nodes = lattice_nodes(1.0)
    px, py = reflection_perms(nodes)
    h = torch.relu(torch.randn(1, 25, 2, 16, 16) - 1.0)
    wall = (torch.rand(1, 4, 16, 16) > 0.8).float()
    out = rigid_bounce_back(h, nodes, px, py, wall, K=5)
    r = lambda x: torch.roll(x, shifts=(3, -5), dims=(-2, -1))
    assert (rigid_bounce_back(r(h), nodes, px, py, r(wall), K=5) - r(out)).abs().max() < 1e-5
