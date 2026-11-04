"""CPU geometry, masked statistics, residual nulls, accuracy and gradients."""
import importlib.util
import math
from pathlib import Path

import pytest
import torch

import kwm.lock_nodes as lock
from kwm.collision import named_nodes
from kwm.lock_nodes import (LocalSearch, crop_box, gaussian, lock_measurement, measure_nodes,
                            pad_world, shifted_window_mask, shift_snapshot)
from kwm.transport import lk_displacement, lk_support, spectral_shift


def cone_scene():
    y, x = torch.meshgrid(torch.arange(19).float() - 9, torch.arange(17).float(), indexing="ij")
    # Any of the 5x5 subcell samples inside a 60-degree cone: FloWM-3D's
    # 177-cell mask at rows 7..25, columns 16..32, agent at (16,16).
    mask = (y.abs() - .4).clamp_min(0) <= (x + .4) / (3 ** .5)
    assert mask.sum() == 177
    background = torch.stack([.3 + .1 * torch.sin(.4*x + .2*y + k) + .1 * torch.cos(.3*y - .2*x + k)
                              for k in range(8)])[None]
    blob = torch.stack([torch.exp(-((x-10)**2+(y+1)**2)/.75) *
                       (1 + .3 * torch.cos(.9*x + k) * torch.sin(.7*y + k)) for k in range(8)])[None]
    return background, blob, mask


@pytest.mark.parametrize("c4", [False, True])
def test_legacy_measurement_bit_identical(c4):
    path = Path(__file__).parent / "reference/lock_nodes_pre_k3d.py"
    spec = importlib.util.spec_from_file_location("kwm._lock_pre_k3d", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.manual_seed(24)
    previous = torch.randn(2, 4, 12, 12)
    current = spectral_shift(previous, torch.tensor([[.2, -.1], [-.4, .3]]), c4=c4)
    nodes, action = named_nodes("v9"), torch.randn(2, 2) * .1
    velocities = nodes[None] + torch.randn(2, 9, 2) * .05
    for world in (12, 19):
        old = reference.lock_measurement(previous, current, velocities, nodes, action, world, c4=c4)
        new = lock_measurement(previous, current, velocities, nodes, action, world, c4=c4,
                               box=None, support_mask=None, sigma=2., radius=6, static_residual=False)
        assert all(torch.equal(a, b) for a, b in zip(old, new))


@pytest.mark.parametrize("sigma", [.5, 1., 2.])
def test_cone_translating_blob(sigma):
    background, blob, mask = cone_scene()
    velocity = torch.tensor([.3, -.6])
    previous = (background + blob) * mask
    moving = crop_box(spectral_shift(pad_world(blob, 33, (7, 16)), velocity), (19, 17), (7, 16))
    current = (background + moving) * mask
    nodes = named_nodes("v17")
    estimate, evidence, locked = measure_nodes(previous, current, nodes[None], nodes, torch.zeros(1, 2),
                                               world=33, box=(7, 16), support_mask=mask,
                                               sigma=sigma, radius=None, static_residual=True)
    i = (nodes - velocity).norm(dim=-1).argmin()
    assert ((velocity - nodes[i]).abs() < .75).all()
    assert evidence[0, i] > 0
    assert (estimate[0, i] - velocity).norm() < .1
    assert evidence[0, 0] == 0 and locked[0, 0] == 0
    assert torch.isfinite(estimate).all()


@pytest.mark.parametrize("action", [[0., 0.], [1., 0.], [.3, -.6]])
def test_background_null_and_ego_lock(action):
    background, _, mask = cone_scene()
    nodes, action = named_nodes("v17"), torch.tensor([action])
    previous = background * mask
    # Independent transport implementation; +action is camera displacement,
    # so static image content moves by -action.
    current = crop_box(spectral_shift(pad_world(previous, 33, (7, 16)), -action), (19, 17), (7, 16)) * mask
    # With zero ego motion, also exercise direct identical observations (not
    # an FFT-generated current), exposing roundoff in the static subtraction.
    if torch.count_nonzero(action) == 0:
        current = previous.clone()
    args = (previous, current, nodes[None], nodes, action, 33)
    kw = dict(box=(7, 16), support_mask=mask, sigma=1., radius=None)
    delta, evidence, locked = lock_measurement(*args, **kw, static_residual=True)
    assert torch.count_nonzero(evidence) == torch.count_nonzero(locked) == torch.count_nonzero(delta) == 0
    delta, evidence, locked = lock_measurement(*args, **kw)
    assert evidence[0, 0] > 0 and locked[0, 0] == 1
    torch.testing.assert_close(delta[0, 0], torch.zeros(2), rtol=0, atol=1e-5)


@pytest.mark.parametrize("world,box", [((23, 31), (4, 13)), ((23, 31), (20, 28)), (33, (7, 16))])
def test_box_transport_and_mask(world, box):
    torch.manual_seed(17)
    shape = (9, 11)
    snapshot = torch.randn(2, 3, *shape)
    shifts = torch.tensor([[[1., -2.], [-3., 2.]], [[0., 1.], [2., 0.]]])
    full = pad_world(snapshot, world, box)
    assert torch.equal(crop_box(full, shape, box), snapshot)
    actual = shift_snapshot(snapshot, shifts, world, box=box)
    expected = spectral_shift(full[:, None].expand(-1, 2, -1, -1, -1), shifts)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    support = torch.rand(*shape) > .3
    actual_mask = shifted_window_mask(shifts, world, shape, box=box, support_mask=support)
    full_mask = pad_world(support.float(), world, box)
    expected_mask = torch.stack([torch.stack([crop_box(torch.roll(full_mask, (int(dy), int(dx)), (-2, -1)), shape, box)
                                              for dx, dy in batch]) for batch in shifts])
    torch.testing.assert_close(actual_mask, expected_mask, rtol=0, atol=4e-6)


def test_masked_values_cannot_affect_measurement_or_gradients():
    torch.manual_seed(18)
    background, blob, mask = cone_scene()
    nodes = named_nodes("v9")
    previous = (background + blob) * mask
    current = crop_box(spectral_shift(pad_world(previous, (33, 35), (7, 16)), torch.tensor([.1, -.1])),
                       (19, 17), (7, 16)) * mask
    outputs = []
    for garbage in (0., 100.):
        p = (previous + garbage * ~mask).requires_grad_()
        c = (current - garbage * ~mask).requires_grad_()
        result = lock_measurement(p, c, nodes[None], nodes, torch.zeros(1, 2), (33, 35),
                                   box=(7, 16), support_mask=mask, sigma=1., radius=None)
        dp, dc = torch.autograd.grad(result[0].sum(), (p, c))
        assert (dp[..., ~mask] == 0).all() and (dc[..., ~mask] == 0).all()
        assert dp.abs().sum() > 0 and dc.abs().sum() > 0
        assert not result[1].requires_grad and not result[2].requires_grad
        outputs.append((*result, dp, dc))
    assert all(torch.equal(a, b) for a, b in zip(*outputs))


def test_lk_mask_border_and_statistics():
    torch.manual_seed(22)
    support = torch.zeros(1, 1, 15, 17)
    support[..., 3:12, 4:14] = 1
    h, u = torch.randn(1, 1, 3, 15, 17), torch.randn(1, 3, 15, 17)
    kw = dict(return_trace=True, per_node_damping=True, support_mask=support)
    baseline = lk_displacement(h, u, **kw)
    modified = lk_displacement(h + 1000 * (1-support[:, :, None]), u - 1000 * (1-support), **kw)
    assert all(torch.equal(a, b) for a, b in zip(baseline, modified))
    # Enlarging unobserved padding cannot alter damping or valid derivatives.
    padded = lk_displacement(torch.nn.functional.pad(h, (4, 3, 2, 5)), torch.nn.functional.pad(u, (4, 3, 2, 5)),
                             return_trace=True, per_node_damping=True,
                             support_mask=torch.nn.functional.pad(support, (4, 3, 2, 5)))
    for a, b in zip(baseline, padded):
        torch.testing.assert_close(a, b[..., 2:17, 4:21], rtol=1e-6, atol=1e-6)
    assert lk_support(support).sum() == 7 * 8


def test_geometry_cache_and_gaussian_options():
    _, blob, mask = cone_scene()
    kwargs = dict(box=(7, 16), support_mask=mask, sigma=1., gaussian_radius=None)
    first = LocalSearch(blob, 33, **kwargs)
    assert first.mask is LocalSearch(blob, 33, **kwargs).mask
    for replacement in (dict(box=(6, 16)), dict(world=(33, 35)), dict(support_mask=~mask)):
        kw = dict(kwargs, **replacement)
        other = LocalSearch(blob, kw.pop("world", 33), **kw)
        assert first.mask is not other.mask
    mask[9, 5] = ~mask[9, 5]
    assert first.mask is not LocalSearch(blob, 33, **kwargs).mask
    for sigma in (.5, 1., 2.):
        radius = math.ceil(3*sigma)
        assert torch.equal(gaussian(blob, sigma, None), gaussian(blob, sigma, radius))
    assert torch.equal(gaussian(blob, 1., 0), blob)


@pytest.mark.parametrize("velocity,expected_lock", [(.1, True), (.5, False)])
def test_box_gradient_routing(velocity, expected_lock):
    y, x = torch.meshgrid(torch.arange(19).float(), torch.arange(17).float(), indexing="ij")
    previous = torch.exp(-((x-8)**2+(y-9)**2)/8)[None, None].requires_grad_()
    current = crop_box(spectral_shift(pad_world(previous.detach(), 33, (7, 16)), torch.tensor([velocity, 0.])),
                       (19, 17), (7, 16)).requires_grad_()
    nodes = torch.zeros(1, 1, 2)
    delta, evidence, locked = lock_measurement(previous, current, nodes, nodes[0], torch.zeros(1, 2), 33,
                                               box=(7, 16), sigma=1., radius=None)
    assert locked.item() == expected_lock and evidence.item() > 0
    gradients = torch.autograd.grad(delta.sum(), (previous, current))
    assert all(torch.isfinite(g).all() for g in gradients)
    assert all(bool(g.abs().sum() > 0) == expected_lock for g in gradients)


def test_empty_support_and_velocity_units():
    background, _, mask = cone_scene()
    nodes = named_nodes("v5")
    kw = dict(world=(33, 35), box=(7, 16), dt=.5, support_mask=torch.zeros_like(mask), static_residual=True)
    estimate, evidence, locked = measure_nodes(background, background, nodes[None], nodes, torch.ones(1, 2), **kw)
    assert torch.equal(estimate, nodes[None])
    assert torch.count_nonzero(evidence) == torch.count_nonzero(locked) == 0
    kw.update(support_mask=mask, static_residual=False)
    delta, evidence, locked = lock_measurement(background, background, nodes[None], nodes, torch.zeros(1, 2), **kw)
    expected = nodes[None] + delta/.5, evidence, locked
    actual = measure_nodes(background, background, nodes[None], nodes, torch.zeros(1, 2), **kw)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


def test_static_residual_shares_fft_and_preserves_lk_gradients(monkeypatch):
    background, blob, mask = cone_scene()
    previous = ((background + blob) * mask).requires_grad_()
    velocity = torch.tensor([.3, -.6])
    moving = crop_box(spectral_shift(pad_world(blob, 33, (7, 16)), velocity), (19, 17), (7, 16))
    current = ((background + moving) * mask).requires_grad_()
    lattice = named_nodes("v17")
    nu = lattice[None].clone().requires_grad_()
    calls, fft = [], torch.fft.rfft2

    def counted(x, *args, **kw):
        calls.append(tuple(x.shape))
        return fft(x, *args, **kw)

    monkeypatch.setattr(torch.fft, "rfft2", counted)
    delta, evidence, locked = lock_measurement(previous, current, nu, lattice, torch.zeros(1, 2), 33,
                                               box=(7, 16), support_mask=mask, static_residual=True,
                                               sigma=1., radius=None)
    assert calls == [(1, 8, 33, 33), (1, 8, 33, 33)]
    i = (lattice - velocity).norm(dim=-1).argmin()
    assert locked[0, i] == 1 and evidence[0, i] > 0
    gradients = torch.autograd.grad(delta[0, i].sum(), (previous, current, nu))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    # Transported support is a detached gate, including for differentiable nu.
    original = lock.lk_displacement

    def checked(*args, **kw):
        assert not kw["support_mask"].requires_grad
        return original(*args, **kw)

    monkeypatch.setattr("kwm.lock_nodes.lk_displacement", checked)
    lock_measurement(previous, current, nu, lattice, torch.zeros(1, 2), 33,
                     box=(7, 16), support_mask=mask, static_residual=True, sigma=1., radius=None)
