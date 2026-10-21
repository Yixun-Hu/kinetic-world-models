import torch
import torch.nn.functional as F
import pytest

from kwm.model import KineticWM, build_model, make_wall0, count_params
from kwm.collision import (lattice_nodes, reflection_perms, bounce_back,
                           known_wall_field)
from kwm.transport import stream


# ---------------------------------------------- flowm == FloWM original ----
def test_flowm_matches_reference_model():
    from flowm_ref import FlowEquivariantRNN
    torch.manual_seed(0)
    world, win, C = 20, 12, 8
    ref = FlowEquivariantRNN(1, C, world, win, decoder_conv_layers=1, cell_type="v_channels", v_range=2)
    ours = KineticWM("flowm", C=C, world=world, window=win)
    sd = ref.state_dict()
    # same parameter names: cell.conv_u, cell.conv_h, decoder_conv.{0,2}
    missing = ours.load_state_dict(sd, strict=True)
    B, T, Tp = 2, 5, 4
    obs = torch.rand(B, T, 1, win, win)
    a_in = torch.randint(-4, 5, (B, T, 2)).float()
    a_out = torch.randint(-4, 5, (B, Tp, 2)).float()
    ref.eval(); ours.eval()
    with torch.no_grad():
        p_ref = ref(obs, a_in, a_out)
        p_our = ours(obs, a_in, a_out)
    assert p_ref.shape == p_our.shape == (B, Tp, 1, win, win)
    assert (p_ref - p_our).abs().max() < 1e-6


def test_flowm_cell_matches_reference_cell():
    from flowm_ref import FERNN_Cell_w_VChannels
    from kwm.cells import FloWMCell
    torch.manual_seed(1)
    ref = FERNN_Cell_w_VChannels(1, 6, world_size=16, window_size=10, v_range=2)
    ours = FloWMCell(1, 6, world=16, window=10, v_range=2)
    ours.load_state_dict(ref.state_dict(), strict=False)
    u = torch.rand(3, 1, 10, 10)
    h = torch.rand(3, 25, 6, 16, 16)
    a = torch.tensor([[3., -2.], [-7., 1.], [0., 5.]])
    assert (ref(u, h, a) - ours(u, h, a)).abs().max() < 1e-6


def test_spectral_equals_flowm_on_integer_data():
    """Spectral streaming on the integer lattice with integer actions == torch.roll (same weights)."""
    torch.manual_seed(2)
    flowm = KineticWM("flowm", C=8, world=20, window=12)
    spec = KineticWM("spectral", C=8, world=20, window=12, node_spacing=1.0, readout="max")
    spec.load_state_dict(flowm.state_dict(), strict=False)
    obs = torch.rand(2, 4, 1, 12, 12)
    a = torch.randint(-3, 4, (2, 4, 2)).float()
    ao = torch.randint(-3, 4, (2, 3, 2)).float()
    with torch.no_grad():
        assert (flowm(obs, a, ao) - spec(obs, a, ao)).abs().max() < 1e-4


# ------------------------------------------------------- bounce-back ----
def _mirror_x(h, wall, perm_x):
    """Spatial flip x -> W-1-x, relabel velocities by R_x, swap left/right walls."""
    hm = h.flip(-1)[:, perm_x]
    wm = wall.flip(-1)[:, [1, 0, 2, 3]]
    return hm, wm


@pytest.mark.parametrize("mode", ["directional", "symmetric"])
@pytest.mark.parametrize("nodes_kind", ["lattice", "stretched"])
def test_bounce_back_commutes_with_reflection(mode, nodes_kind):
    torch.manual_seed(3)
    if nodes_kind == "lattice":
        nodes = lattice_nodes(0.5)
        px, py = reflection_perms(nodes)
    else:
        nodes = lattice_nodes(1.0) * torch.tensor([1.3, 0.7])         # off-lattice, reflection-closed
        px, py = reflection_perms(nodes)
        # closure check
        assert torch.allclose(nodes[px], nodes * torch.tensor([-1., 1.]))
        assert torch.allclose(nodes[py], nodes * torch.tensor([1., -1.]))
    V = nodes.shape[0]
    h = torch.rand(2, V, 3, 12, 12)
    wall = torch.rand(2, 4, 12, 12)
    out = bounce_back(h, nodes, px, py, wall, mode)
    hm, wm = _mirror_x(h, wall, px)
    out_m = bounce_back(hm, nodes, px, py, wm, mode)
    ref_m, _ = _mirror_x(out, wall, px)
    assert (out_m - ref_m).abs().max() < 1e-5
    # identity when no wall
    assert (bounce_back(h, nodes, px, py, torch.zeros_like(wall), mode) - h).abs().max() == 0
    # mass conservation over V
    assert (out.sum(1) - h.sum(1)).abs().max() < 1e-4 if mode == "directional" else True


def test_bounce_back_reflects_into_wall_population():
    nodes = lattice_nodes(1.0)
    px, py = reflection_perms(nodes)
    V = nodes.shape[0]
    h = torch.zeros(1, V, 1, 8, 8)
    i = int(((nodes - torch.tensor([-2., 1.])).abs().sum(1) == 0).nonzero())   # heading left
    j = int(px[i])
    h[0, i, 0, 4, 1] = 1.0
    wall = known_wall_field(8, band=3)[None]
    out = bounce_back(h, nodes, px, py, wall, "directional")
    assert out[0, j, 0, 4, 1] == 1 and out[0, i, 0, 4, 1] == 0
    # a population moving away from the wall is untouched (no trapping)
    h2 = torch.zeros_like(h); h2[0, j, 0, 4, 1] = 1.0
    assert (bounce_back(h2, nodes, px, py, wall, "directional") - h2).abs().max() == 0


# --------------------------------------------- moment readout (sanity [3]) ----
def _blob(H, W, cx, cy, s=2.0):
    y = torch.arange(H).float()[:, None]
    x = torch.arange(W).float()[None, :]
    dx = (x - cx + W / 2) % W - W / 2
    dy = (y - cy + H / 2) % H - H / 2
    return torch.exp(-(dx ** 2 + dy ** 2) / (2 * s ** 2))


def _moment_est(nodes, spectral, T=12, temp=0.05):
    H = W = 50
    v = torch.tensor([0.7, -1.3])
    V = nodes.shape[0]
    h = torch.zeros(1, V, 1, H, W)
    zero_a = torch.zeros(1, 2)
    for t in range(T):
        f = _blob(H, W, 25 + v[0] * t, 25 + v[1] * t)
        if spectral:
            hs = stream(h, nodes, zero_a, 1.0)
        else:
            hs = torch.stack([torch.roll(h[:, i], shifts=(int(round(float(n[1]))), int(round(float(n[0])))), dims=(-2, -1))
                              for i, n in enumerate(nodes)], 1)
        h = F.relu(0.9 * hs + f)
    e = (h ** 2).sum((2, 3, 4))[0]
    w = torch.softmax(e / (e.max() * temp), 0)
    return (w[:, None] * nodes).sum(0), v


def test_moment_readout_subnode_velocity():
    est, v = _moment_est(lattice_nodes(0.5), spectral=True)
    assert (est - v).norm() < 0.1
    est_i, _ = _moment_est(lattice_nodes(1.0), spectral=False)
    assert (est_i - v).norm() > 0.2      # integer lattice + roll cannot resolve it


def test_model_moment_field_matches_formula():
    torch.manual_seed(4)
    m = KineticWM("spectral", C=4, world=16, window=8, node_spacing=1.0)
    h = torch.rand(2, m.num_v, 4, 16, 16)
    aux = {"mean_velocity": [], "argmax_velocity": []}
    m.decode(h, aux)
    w = m.velocity_weights(h[..., :8, :8])
    assert torch.allclose(w.sum(1), torch.ones(2, 8, 8), atol=1e-5)
    mean = aux["mean_velocity"][0]
    assert mean.shape == (2, 2, 8, 8) and mean.abs().max() <= 2.0 + 1e-5


# ----------------------------------------------- shapes and gradients ----
CONFIGS = [
    dict(model="flowm"),
    dict(model="spectral", node_spacing=0.5),
    dict(model="spectral", node_spacing=1.0, readout="max"),
    dict(model="spectral", node_spacing=1.0, readout_rel="on"),
    dict(model="adaptive", node_spacing=1.0, readout_rel="on", node_update_pred="off"),
    dict(model="adaptive", node_spacing=1.0),
    dict(model="spectral", node_spacing=1.0, collision="bounce", wall_mode="known"),
    dict(model="spectral", node_spacing=1.0, collision="bounce", wall_mode="learned"),
    dict(model="spectral", node_spacing=1.0, collision="bounce", wall_mode="known", bounce="pointwise",
         bounce_mode="symmetric"),
    dict(model="spectral", node_spacing=1.0, collision="bounce", wall_mode="known", bounce="pointwise"),
    dict(model="spectral", node_spacing=1.0, collision="bounce", wall_mode="learned", bounce="rigid", rigid_K=4),
    dict(model="spectral", node_spacing=1.0, collision="vmix"),
    dict(model="spectral", node_spacing=1.0, collision="mlp"),
    dict(model="spectral", node_spacing=1.0, grad_ckpt=True, collision="mlp"),
    dict(model="adaptive", node_spacing=1.0, grad_ckpt=True),
]


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: "-".join(f"{k}={v}" for k, v in c.items()))
def test_model_forward_backward(cfg):
    torch.manual_seed(5)
    cfg = dict(cfg, C=8, world=20, window=12)
    m = build_model(cfg)
    if cfg.get("collision") == "mlp":     # zero-init residual: perturb so all params get gradient
        with torch.no_grad():
            m.coll.fc2.weight.normal_(0, 0.1)
    m.train()
    B, T, Tp = 2, 4, 3
    obs = torch.rand(B, T, 1, 12, 12)
    a = torch.randn(B, T, 2) * 3
    ao = torch.randn(B, Tp, 2) * 3
    if cfg["model"] == "flowm":
        a, ao = a.round(), ao.round()
    wall0 = make_wall0(torch.rand(B, 2) * 20, 20)
    p = m(obs, a, ao, target=torch.rand(B, Tp, 1, 12, 12), tf_ratio=0.5, wall0=wall0)
    assert p.shape == (B, Tp, 1, 12, 12)
    ((p - 0.3) ** 2).mean().backward()
    for n, q in m.named_parameters():
        if q.numel() == 0:
            continue
        assert q.grad is not None and q.grad.abs().sum() > 0, n
    assert count_params(m) > 0


def test_fractional_dt_runs():
    m = KineticWM("spectral", C=4, world=16, window=8, node_spacing=1.0)
    p = m(torch.rand(1, 3, 1, 8, 8), torch.randn(1, 3, 2), torch.randn(1, 2, 2), dt=0.5)
    assert p.shape == (1, 2, 1, 8, 8)
    with pytest.raises(ValueError):
        KineticWM("flowm", C=4, world=16, window=8)(torch.rand(1, 3, 1, 8, 8), torch.zeros(1, 3, 2),
                                                 torch.zeros(1, 2, 2), dt=0.5)
