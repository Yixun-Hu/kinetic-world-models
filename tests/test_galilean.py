"""Galilean boosts: velocity ring / scene boost in the env, the boost action on the memory, and the
equivariance evaluations."""
import torch
import pytest

from kwm.env import EnvConfig, make_batch
from kwm.collision import lattice_nodes
from kwm.transport import stream
from kwm.equiv import boost_transform, equiv_error
from kwm.model import KineticWM, build_model


# ------------------------------------------------------------------ env ----
@pytest.mark.parametrize("real_velocity", [True, False])
def test_v_ring_sampling(real_velocity):
    cfg = EnvConfig(real_velocity=real_velocity, real_camera=True, vmax=2.0, v_ring=1.0)
    b = make_batch(16, 3, 1, cfg, gen=torch.Generator().manual_seed(0))
    v = b["digit_states"]["vel"][:, 0].abs().amax(-1)
    assert (v > 1.0).all() and (v <= 2.0 + 1e-6).all()


def test_scene_boost_and_explicit_boost():
    base = EnvConfig(real_velocity=True, real_camera=True)
    b0 = make_batch(8, 3, 1, base, gen=torch.Generator().manual_seed(1))
    b1 = make_batch(8, 3, 1, EnvConfig(real_velocity=True, real_camera=True, scene_boost=1.0), gen=torch.Generator().manual_seed(1))
    d = b1["digit_states"]["vel"][:, 0] - b0["digit_states"]["vel"][:, 0]          # [B,nd,2]
    assert torch.allclose(d, d[:, :1].expand_as(d), atol=1e-6)                       # same u for every digit
    assert torch.allclose(d[:, 0], b1["scene_boost"], atol=1e-6) and b1["scene_boost"].abs().max() <= 1.0
    assert (b0["actions"] == b1["actions"]).all()                                     # rest of the scene unchanged
    u = torch.tensor([0.4, -0.7])
    b2 = make_batch(8, 3, 1, base, gen=torch.Generator().manual_seed(1), boost=u)
    assert torch.allclose(b2["digit_states"]["vel"][:, 0] - b0["digit_states"]["vel"][:, 0], u.expand(8, 5, 2))
    assert torch.allclose(b2["digit_states"]["cent"][:, 0], b0["digit_states"]["cent"][:, 0])


# ---------------------------------------------- boost action, synthetic ----
def _blob(H, cx, cy, s=2.5):
    y = torch.arange(H).float()[:, None]
    x = torch.arange(H).float()[None, :]
    dx = (x - cx + H / 2) % H - H / 2
    dy = (y - cy + H / 2) % H - H / 2
    return torch.exp(-(dx ** 2 + dy ** 2) / (2 * s ** 2))


def _node(nodes, v):
    return int(((nodes - torch.tensor(v)).abs().sum(1) < 1e-6).nonzero())


@pytest.mark.parametrize("nu", [(1.0, 0.0), (-1.0, 1.0)])
def test_boost_action_fixed_lattice_exact(nu):
    H, T = 48, 10
    nodes = lattice_nodes(1.0)
    h0 = torch.zeros(1, 25, 2, H, H)
    h0[0, _node(nodes, [1.0, 0.0]), 0] = _blob(H, 20, 20)
    h0[0, _node(nodes, [0.0, -1.0]), 1] = _blob(H, 30, 12)
    nu_hat = torch.tensor([nu])
    hb = boost_transform(h0, nodes, 1.0, 2, nu_hat, 0)
    a = torch.tensor([[0.37, -1.6]])
    ho = h0.clone()
    for _ in range(T):
        ho, hb = stream(ho, nodes, a, 1.0), stream(hb, nodes, a, 1.0)
    err = equiv_error(boost_transform(ho, nodes, 1.0, 2, nu_hat, T), hb, 2)
    assert err.max() < 1e-4


def test_offlattice_boost_fixed_lattice_is_not_equivariant():
    """A fixed lattice cannot hold the off-lattice part of the boost: error clearly > 0."""
    H, T = 48, 20
    nodes = lattice_nodes(1.0)
    h0 = torch.zeros(1, 25, 1, H, H)
    h0[0, _node(nodes, [1.0, 0.0]), 0] = _blob(H, 20, 20)
    nu_hat = torch.tensor([[0.4, 0.0]])
    hb = boost_transform(h0, nodes, 1.0, 1, nu_hat, 0)             # = h0 (round(0.4) = 0)
    ho = h0.clone()
    for _ in range(T):
        ho, hb = stream(ho, nodes, torch.zeros(1, 2), 1.0), stream(hb, nodes, torch.zeros(1, 2), 1.0)
    assert equiv_error(boost_transform(ho, nodes, 1.0, 1, nu_hat, T), hb, 1).item() > 0.5


# ------------------------------------------------------------ evaluations ----
def test_memory_equivariance_eval_runs():
    from kwm.eval import equiv_memory
    torch.manual_seed(0)
    m = build_model(dict(model="spectral", C=4, node_spacing=1.0)).eval()
    args = dict(model="spectral", node_spacing=1.0)
    out = equiv_memory(m, args, EnvConfig(real_velocity=True, real_camera=True), 4, "cpu", ts=(3, 6), chunk=4)
    assert {"comoving_on", "comoving_off", "scene_on", "scene_off"} <= set(out)
    # the co-moving boost is an exact symmetry: on-lattice the kinetic memory is exactly equivariant
    assert out["comoving_on"]["t6"][0] < 1e-3
    assert out["comoving_off"]["t6"][0] > 10 * out["comoving_on"]["t6"][0]


def test_output_equivariance():
    """Output-level equivariance is not exactly zero on the lattice, even untrained: (i) the finite velocity lattice -- the boosted
    run's edge channels receive content the unboosted run's lattice cannot hold, and the max / moment readouts pool
    over all channels; (ii) the viewport crop and the nonlinear recurrence. So we only check that it runs, is finite,
    and that the adaptive model stays well below 0.75."""
    from kwm.eval import equiv_output
    torch.manual_seed(0)
    cfg = EnvConfig(real_velocity=True, real_camera=True)
    for v in ("spectral", "flowm"):
        m = build_model(dict(model=v, C=4, node_spacing=1.0)).eval()
        o = equiv_output(m, dict(model=v, node_spacing=1.0, t_obs=8), cfg, 4, "cpu", ts=(3, 5), chunk=4)
        assert all(torch.isfinite(torch.tensor(o[k]["t5"]["nmse"][0])) for k in ("comoving_on", "comoving_off"))
    mn = build_model(dict(model="adaptive", C=4, node_spacing=1.0)).eval()
    on = equiv_output(mn, dict(model="adaptive", node_spacing=1.0, t_obs=8), cfg, 4, "cpu", ts=(3, 5), chunk=4)
    assert on["comoving_off"]["t5"]["nmse"][0] < 0.75


def test_readout_rel_is_galilean_invariant():
    """With content only in interior nodes, a comoving integer boost (relabel nodes by k, action + k) leaves the
    relative-velocity readout -- hence the decoded frame -- unchanged; the absolute readout changes."""
    from kwm.equiv import boost_transform
    torch.manual_seed(0)
    nodes = lattice_nodes(1.0)
    diffs = {}
    for rel in (True, False):
        m = build_model(dict(model="spectral", C=4, world=12, window=8, node_spacing=1.0,
                             readout_rel="on" if rel else "off")).eval()
        h = torch.zeros(1, 25, 4, 12, 12)
        inner = [int(i) for i in range(25) if nodes[i].abs().max() <= 1]
        h[0, inner] = torch.rand(len(inner), 4, 12, 12)
        k = torch.tensor([[1.0, 0.0]])
        hb = boost_transform(h, nodes, 1.0, 4, k, 0)
        a = torch.tensor([[0.7, -1.2]])
        with torch.no_grad():
            m._rel = a if rel else None
            p0 = m.decode(h)
            m._rel = (a + k) if rel else None
            p1 = m.decode(hb)
        d = (p0 - p1).abs().max().item() / p0.abs().max().clamp_min(1e-8).item()
        diffs[rel] = d
    assert diffs[True] < 1e-3 and diffs[False] > 100 * diffs[True], diffs
