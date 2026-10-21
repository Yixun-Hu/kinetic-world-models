"""Adaptive velocity nodes."""
import torch
import pytest

from kwm.cells import AdaptiveNodeCell
from kwm.collision import lattice_nodes
from kwm.transport import spectral_shift
from kwm.model import build_model

H = 32


def _blob(cx, cy, s=2.5):
    y = torch.arange(H).float()[:, None]
    x = torch.arange(H).float()[None, :]
    dx = (x - cx + H / 2) % H - H / 2
    dy = (y - cy + H / 2) % H - H / 2
    return torch.exp(-(dx ** 2 + dy ** 2) / (2 * s ** 2))


def _run(blobs, T, seed=0, frames=None):
    """blobs: list of ((x0, y0), (vx, vy)); returns per-step node velocities [T, V, 2]."""
    torch.manual_seed(seed)
    cell = AdaptiveNodeCell(C=8, world=H, window=H, num_v=25, spacing=1.0)
    nodes = lattice_nodes(1.0)
    h = torch.zeros(1, 25, 8 + cell.extra + 8, H, H)
    traj = []
    with torch.no_grad():
        for t in range(T):
            if frames is None:
                f = sum(_blob(x0 + vx * t, y0 + vy * t) for (x0, y0), (vx, vy) in blobs)[None, None]
            else:
                f = frames[t]
            h = cell(f, h, torch.zeros(1, 2), nodes, 1.0)
            traj.append(cell.last_nodes[0].clone())
    return torch.stack(traj), nodes, cell, h


def _idx(nodes, v):
    return int((nodes - torch.tensor(v)).norm(dim=1).argmin())


def test_single_blob_three_steps():
    v = (0.7, -1.3)
    traj, nodes, _, _ = _run([((10.0, 20.0), v)], 6)
    i = _idx(nodes, v)
    # frame 0 only writes the first imprint (no evidence yet); 3 evidence steps = after frame 3
    assert (traj[3, i] - torch.tensor(v)).abs().max() <= 0.03, traj[:4, i]


def test_two_blobs_ten_steps():
    vA, vB = (0.7, -1.3), (-1.2, 0.4)
    traj, nodes, _, _ = _run([((8.0, 8.0), vA), ((22.0, 22.0), vB)], 10)
    for v in (vA, vB):
        i = _idx(nodes, v)
        assert (traj[-1, i] - torch.tensor(v)).abs().max() <= 0.03, (v, traj[-1, i])


def test_no_observation_unchanged():
    torch.manual_seed(0)
    cell = AdaptiveNodeCell(C=8, world=H, window=H, num_v=25, spacing=1.0)
    nodes = lattice_nodes(1.0)
    h = torch.zeros(1, 25, 19, H, H)
    h[:, :, 8:10] = (torch.rand(1, 25, 2, 1, 1) - 0.5)
    h[:, :, 10:11] = 3.0
    rho0, W0 = cell.offsets(h), h[:, :, 10].mean((-2, -1))
    with torch.no_grad():
        for _ in range(5):
            h = cell(torch.zeros(1, 1, H, H), h, torch.zeros(1, 2), nodes, 1.0)
    assert (cell.offsets(h) - rho0).abs().max() < 1e-5 and (h[:, :, 10].mean((-2, -1)) - W0).abs().max() < 1e-4


def test_boost_equivariance_of_node_trajectory():
    v, u = (0.7, -1.3), (0.27, -0.41)
    vb = (v[0] + u[0], v[1] + u[1])
    tr0, nodes, _, _ = _run([((10.0, 20.0), v)], 10)
    tr1, _, _, _ = _run([((10.0, 20.0), vb)], 10)
    n0, n1 = tr0[-1, _idx(nodes, v)], tr1[-1, _idx(nodes, vb)]
    assert ((n1 - n0) - torch.tensor(u)).abs().max() <= 0.05, (n0, n1)


def test_adaptive_model_forward_backward():
    torch.manual_seed(2)
    m = build_model(dict(model="adaptive", C=8, world=20, window=12, node_spacing=1.0, node_update_pred="on")).train()
    assert m.state_extra == 3 + 8 and m.cell.bound == 0.75
    p, aux = m(torch.rand(2, 6, 1, 12, 12), torch.randn(2, 6, 2) * 3, torch.randn(2, 3, 2) * 3, record_nodes=True)
    ((p - 0.3) ** 2).mean().backward()
    for n, q in m.named_parameters():
        assert q.grad is not None and torch.isfinite(q.grad).all(), n
    assert aux["node_traj"].shape == (2, 9, 25, 2) and m.cell.n_s.grad.abs() > 0


# ------------------------------------------------- walls on adaptive nodes (experimental) ----
def test_adaptive_bounce_moves_content_and_reflects_offset():
    from kwm.collision import reflection_perms, known_wall_field
    torch.manual_seed(3)
    Hh = 24
    cell = AdaptiveNodeCell(C=2, world=Hh, window=8, num_v=25, spacing=1.0)
    nodes = lattice_nodes(1.0)
    px, py = reflection_perms(nodes)
    i = int(((nodes - torch.tensor([-1.0, 0.0])).abs().sum(1) == 0).nonzero())     # heading left
    j = int(px[i])                                                                  # (+1, 0)
    h = torch.zeros(1, 25, 2 + 3 + 8, Hh, Hh)
    h[0, i, 0, 10:14, 1:5] = 1.0                                                    # rigid block touching the left wall
    h[0, i, 2] = -0.3; h[0, i, 3] = 0.2; h[0, i, 4] = 5.0                          # rho_i = (-0.3, 0.2), W_i = 5
    wall = known_wall_field(Hh, band=2)[None]
    out = cell.bounce(h, nodes, px, py, wall)
    assert out[0, j, 0].sum() == 16 and out[0, i, 0].sum() == 0                     # whole block moved to R nu
    assert torch.allclose(out[0, j, 2:5].mean((-2, -1)), torch.tensor([0.3, 0.2, 5.0]))
    m = build_model(dict(model="adaptive", collision="bounce", C=8, world=20, window=12, node_spacing=1.0, wall_mode="known"))
    from kwm.model import make_wall0
    p = m(torch.rand(2, 4, 1, 12, 12), torch.randn(2, 4, 2), torch.randn(2, 2, 2),
          wall0=make_wall0(torch.rand(2, 2) * 20, 20))
    assert p.shape == (2, 2, 1, 12, 12) and torch.isfinite(p).all()


# ------------------------------------------------------------- boost equivariance ----
def test_boost_transform_nodes_carry_and_permute():
    from kwm.equiv import boost_transform_nodes
    nodes = lattice_nodes(1.0)
    i = _idx(nodes, (0.0, 0.0))
    h = torch.zeros(1, 25, 2 + 3 + 8, 6, 6)
    h[0, i, 0] = 1.0
    h[0, i, 2] = 0.6; h[0, i, 3] = -0.2; h[0, i, 4] = 7.0          # rho = (0.6, -0.2), W = 7
    out, mask = boost_transform_nodes(h, nodes, 1.0, 2, torch.tensor([[1.3, 0.4]]), 0.75)
    # u = (1.3, 0.4): k = (1, 0), frac = (0.3, 0.4) -> rho = (0.9, 0.2) -> carry x: node (2, 0), rho = (-0.1, 0.2)
    j = _idx(nodes, (2.0, 0.0))
    assert out[0, j, 0].sum() == 36 and torch.allclose(out[0, j, 2:5].mean((-2, -1)), torch.tensor([-0.1, 0.2, 7.0]), atol=1e-5)
    assert mask[0, j] and not mask[0, _idx(nodes, (-2.0, 0.0))]


def test_memory_equivariance_comoving_on_lattice():
    from kwm.eval import equiv_memory
    from kwm.env import EnvConfig
    torch.manual_seed(0)
    m = build_model(dict(model="adaptive", C=4, node_spacing=1.0)).eval()
    out = equiv_memory(m, dict(model="adaptive", node_spacing=1.0), EnvConfig(real_velocity=True, real_camera=True), 4, "cpu",
                          ts=(3, 6), chunk=4)
    # not bit-exact: the LK damping and the texture threshold use means over ALL nodes, and a lattice boost changes
    # which nodes are at the lattice edge (measured ~1e-3 on an untrained model)
    assert out["comoving_on"]["t6"][0] < 5e-3
    assert out["comoving_off"]["t6"][0] >= 0


def test_nodes_frozen_in_prediction_by_default():
    """--node_update_pred off (default): nodes move during the observation and stay fixed while predicting."""
    torch.manual_seed(4)
    obs = torch.rand(1, 6, 1, 12, 12)
    a_in, a_out = torch.randn(1, 6, 2), torch.randn(1, 4, 2)
    trajs = {}
    for mode in ("off", "on"):
        torch.manual_seed(5)
        m = build_model(dict(model="adaptive", C=8, world=20, window=12, node_update_pred=mode)).eval()
        with torch.no_grad():
            _, aux = m(obs, a_in, a_out, record_nodes=True)
        trajs[mode] = aux["node_traj"][0]                                   # [10, 25, 2]
        assert m.cell.update_nodes                                          # restored after forward
    pred_off = trajs["off"][6:]
    assert (pred_off - pred_off[:1]).abs().max() < 1e-6                     # (state stored as a spatial mean)
    assert (trajs["off"][:6] - trajs["on"][:6]).abs().max() == 0           # same observation phase
    assert (trajs["on"][6:] - trajs["on"][5:6]).abs().max() > 1e-4
