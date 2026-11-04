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


def residual_scene(background="smooth", two_blobs=False, action=(0., 0.)):
    """The review's cone scene, with deterministic texture and optional ego motion."""
    bg, blob, mask = cone_scene()
    if background == "smooth3":
        bg = .3 + 3 * (bg - .3)
    elif background == "noise01":
        bg = .3 + .1 * torch.randn(bg.shape, generator=torch.Generator().manual_seed(0))
    elif background != "smooth":
        raise ValueError(background)
    velocities = [torch.tensor([.3, -.6])]
    blobs = [blob]
    if two_blobs:
        velocities.append(torch.tensor([-.5, .2]))
        blobs.append(torch.roll(blob, (4, 1), (-2, -1)))
    action = torch.tensor([action])

    def move(x, v):
        return crop_box(spectral_shift(pad_world(x, 33, (7, 16)), v - action[0]), (19, 17), (7, 16))

    previous = (bg + sum(blobs)) * mask
    current = (move(bg, torch.zeros(2)) + sum(move(b, v) for b, v in zip(blobs, velocities))) * mask
    return previous, current, mask, torch.stack(velocities), action


def report_nodes(name, mode, nodes, estimate, evidence, locked, diagnostics):
    """Print every node, including unselected mass, for pytest -s / failure logs."""
    print(f"\n{name} {mode}: node, estimate, evidence_before_modes, evidence, locked")
    for i, node in enumerate(nodes):
        print(f"{i:2d} {node.tolist()} {estimate[0, i].tolist()} "
              f"{diagnostics['evidence_before_modes'][0, i].item():.9g} "
              f"{evidence[0, i].item():.9g} {int(locked[0, i])}")


@pytest.mark.parametrize("mode", ["subtract", "mask"])
@pytest.mark.parametrize("background", ["smooth", "noise01", "smooth3"])
@pytest.mark.parametrize("action", [(0., 0.), (1., 0.)])
def test_residual_all_nodes(background, mode, action):
    """Review regression; subtract is recorded without weakening the old test.

    Subtract, ego=0: (nearest error, evidence, lock) = smooth
    (0.071999, 1.374438, 1), noise01 (0.170003, 0.000192912, 0), smooth3
    (0.122300, 0.0847343, 0). At (-.75,-.75), smooth estimates
    (-1.135,-.542) with evidence .362; (-.25,-.75) gets .882.
    These are known failures, not accuracy guarantees for subtract mode.
    """
    previous, current, mask, truth, ego = residual_scene(background, action=action)
    nodes = named_nodes("v17")
    estimate, evidence, locked, diagnostics = measure_nodes(
        previous, current, nodes[None], nodes, ego, world=33, box=(7, 16),
        support_mask=mask, sigma=1., radius=None, static_residual=mode, return_diagnostics=True)
    report_nodes(f"{background} ego={action}", mode, nodes, estimate, evidence, locked, diagnostics)
    i = (nodes - truth[0]).norm(dim=-1).argmin()
    assert evidence[0, 0] == locked[0, 0] == diagnostics["evidence_before_modes"][0, 0] == 0
    assert all(torch.isfinite(x).all() for x in (estimate, evidence, *diagnostics.values()))
    if mode == "mask":
        far = (nodes - truth[0]).norm(dim=-1) > .5
        assert evidence[0, i] > 0 and locked[0, i] == 1
        assert (estimate[0, i] - truth[0]).norm() <= .1
        # Check before and after the per-node two-mode selection.
        for mass in (evidence, diagnostics["evidence_before_modes"]):
            assert mass[0].argmax() == i
            assert (mass[0, far] < .25 * mass[0, i]).all()


@pytest.mark.parametrize("mode", ["subtract", "mask"])
def test_residual_two_blobs(mode):
    """Equal-amplitude blobs separated by (1,4), velocities (.3,-.6), (-.5,.2).

    Subtract nearest errors are .147460 and .398040; evidence .931538 and
    .851451, both locked. Mask errors are .037016 and .056064, both evidenced.
    """
    previous, current, mask, truth, ego = residual_scene(two_blobs=True)
    nodes = named_nodes("v17")
    estimate, evidence, locked, diagnostics = measure_nodes(
        previous, current, nodes[None], nodes, ego, world=33, box=(7, 16),
        support_mask=mask, sigma=1., radius=None, static_residual=mode, return_diagnostics=True)
    report_nodes("two_blobs", mode, nodes, estimate, evidence, locked, diagnostics)
    indices = (nodes[None] - truth[:, None]).norm(dim=-1).argmin(-1)
    assert indices.unique().numel() == 2
    if mode == "mask":
        assert (evidence[0, indices] > 0).all()
        assert ((estimate[0, indices] - truth).norm(dim=-1) <= .1).all()


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
        diagnostic = lock_measurement(previous, current, velocities, nodes, action, world, c4=c4,
                                      return_diagnostics=True)
        assert all(torch.equal(a, b) for a, b in zip(old, diagnostic))
        assert set(diagnostic[3]) == {"evidence_before_modes", "estimate", "locked"}
        assert torch.equal(diagnostic[3]["estimate"], velocities + new[0])


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


@pytest.mark.parametrize("mode", [True, "subtract", "mask"])
@pytest.mark.parametrize("action", [[0., 0.], [1., 0.], [.3, -.6]])
def test_background_null_and_ego_lock(action, mode):
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
    delta, evidence, locked = lock_measurement(*args, **kw, static_residual=mode)
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


@pytest.mark.parametrize("mode", [True, "mask"])
def test_empty_support_and_velocity_units(mode):
    background, _, mask = cone_scene()
    nodes = named_nodes("v5")
    kw = dict(world=(33, 35), box=(7, 16), dt=.5, support_mask=torch.zeros_like(mask), static_residual=mode)
    estimate, evidence, locked = measure_nodes(background, background, nodes[None], nodes, torch.ones(1, 2), **kw)
    assert torch.equal(estimate, nodes[None])
    assert torch.count_nonzero(evidence) == torch.count_nonzero(locked) == 0
    kw.update(support_mask=mask, static_residual=False)
    delta, evidence, locked = lock_measurement(background, background, nodes[None], nodes, torch.zeros(1, 2), **kw)
    expected = nodes[None] + delta/.5, evidence, locked
    actual = measure_nodes(background, background, nodes[None], nodes, torch.zeros(1, 2), **kw)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.parametrize("mode", [True, "subtract", "mask"])
def test_static_residual_shares_fft_and_preserves_lk_gradients(monkeypatch, mode):
    background, blob, mask = cone_scene()
    previous = ((background + blob) * mask).requires_grad_()
    velocity = torch.tensor([.3, -.6])
    moving = crop_box(spectral_shift(pad_world(blob, 33, (7, 16)), velocity), (19, 17), (7, 16))
    current = ((background + moving) * mask).requires_grad_()
    lattice = named_nodes("v17")
    nu = lattice[None].clone().requires_grad_()
    calls, inverse_calls, fft, inverse = [], [], torch.fft.rfft2, torch.fft.irfft2

    def counted(x, *args, **kw):
        calls.append(tuple(x.shape))
        return fft(x, *args, **kw)

    monkeypatch.setattr(torch.fft, "rfft2", counted)

    def counted_inverse(x, *args, **kw):
        inverse_calls.append(tuple(x.shape))
        return inverse(x, *args, **kw)

    monkeypatch.setattr(torch.fft, "irfft2", counted_inverse)
    delta, evidence, locked, diagnostics = lock_measurement(
        previous, current, nu, lattice, torch.zeros(1, 2), 33, box=(7, 16), support_mask=mask,
        static_residual=mode, sigma=1., radius=None, return_diagnostics=True)
    assert calls == [(1, 8, 33, 33), (1, 8, 33, 33)]
    assert inverse_calls == [(1, 18, 8, 33, 17), (1, 25, 8, 33, 17)]
    assert not any(x.requires_grad for x in diagnostics.values())
    i = (lattice - velocity).norm(dim=-1).argmin()
    assert locked[0, i] == 1 and evidence[0, i] > 0
    gradients = torch.autograd.grad(delta[0, i].sum(), (previous, current, nu), retain_graph=True)
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    j = ((evidence[0] > 0) & (locked[0] == 0)).nonzero()[0, 0]
    search_gradients = torch.autograd.grad(delta[0, j].sum(), (previous, current, nu))
    assert all(torch.count_nonzero(g) == 0 for g in search_gradients)
    # Transported support is a detached gate, including for differentiable nu.
    original = lock.lk_displacement

    def checked(*args, **kw):
        assert not kw["support_mask"].requires_grad
        if "support_weight" in kw:
            assert not kw["support_weight"].requires_grad
        return original(*args, **kw)

    monkeypatch.setattr("kwm.lock_nodes.lk_displacement", checked)
    lock_measurement(previous, current, nu, lattice, torch.zeros(1, 2), 33,
                     box=(7, 16), support_mask=mask, static_residual=mode, sigma=1., radius=None)


@pytest.mark.parametrize("mode", [False, "subtract", "mask"])
def test_diagnostics_and_masked_gradient_isolation(monkeypatch, mode):
    previous, current, mask, _, action = residual_scene()
    nodes = named_nodes("v17")
    calls, original = [], lock.two_modes

    def capture(displacement, weight, *args):
        calls.append(weight.sum((-2, -1)))
        return original(displacement, weight, *args)

    monkeypatch.setattr(lock, "two_modes", capture)
    kw = dict(world=33, box=(7, 16), support_mask=mask, sigma=1., radius=None, dt=.5, static_residual=mode)
    outputs = []
    for garbage in (0., 100.):
        p = (previous + garbage * ~mask).requires_grad_()
        c = (current - garbage * ~mask).requires_grad_()
        result = lock_measurement(p, c, nodes[None], nodes, action, **kw, return_diagnostics=True)
        delta, evidence, locked, diag = result
        plain = lock_measurement(p, c, nodes[None], nodes, action, **kw)
        assert len(plain) == 3 and all(torch.equal(a, b) for a, b in zip(result, plain))
        assert torch.equal(diag["evidence_before_modes"], calls[-1])
        assert torch.equal(diag["estimate"], nodes[None] + delta / .5)
        assert torch.equal(diag["locked"], locked)
        assert (diag["evidence_before_modes"] >= evidence - 1e-5).all()
        assert ("residual_weight" in diag) == bool(mode)
        assert not any(x.requires_grad for x in diag.values())
        if mode:
            assert diag["residual_weight"].shape == (1, 19, 17)
            assert torch.count_nonzero(diag["residual_weight"][..., ~mask]) == 0
        dp, dc = torch.autograd.grad(delta.sum(), (p, c))
        assert (dp[..., ~mask] == 0).all() and (dc[..., ~mask] == 0).all()
        outputs.append((*result[:3], *diag.values(), dp, dc))
    assert all(torch.equal(a, b) for a, b in zip(*outputs))
    velocity_result = measure_nodes(previous, current, nodes[None], nodes, action, **kw, return_diagnostics=True)
    assert torch.equal(velocity_result[0], velocity_result[3]["estimate"])
    if mode == "subtract":
        alias = lock_measurement(previous, current, nodes[None], nodes, action, **dict(kw, static_residual=True))
        assert all(torch.equal(a, b) for a, b in zip(alias, outputs[0]))


@pytest.mark.parametrize("shape", [(19, 17), (18, 20)])
def test_periodic_gradients_match_fourier_derivative(shape):
    """Independent Fourier reference for the spatial derivative's sign/Nyquist handling."""
    x = torch.randn(2, 3, *shape, generator=torch.Generator().manual_seed(29), requires_grad=True)
    spectrum = torch.fft.fft2(x)
    fy = torch.fft.fftfreq(shape[0])[:, None]
    fx = torch.fft.fftfreq(shape[1])[None]
    expected = [torch.fft.ifft2(spectrum * (2j * math.pi * f)).real for f in (fx, fy)]
    actual = lock._periodic_gradients(x)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    gradients = torch.autograd.grad(actual[0].square().sum(), x)[0]
    assert torch.isfinite(gradients).all() and gradients.abs().sum() > 0


def test_residual_threshold_normalization_and_search_mass():
    """Relative threshold is channel-summed, masked, scale invariant, and soft."""
    p = torch.zeros(1, 2, 9, 11)
    c = torch.arange(99).reshape(1, 1, 9, 11).float().expand(1, 2, -1, -1) / 99
    mask = torch.ones(9, 11, dtype=torch.bool)
    mask[:, -2:] = False
    nodes = named_nodes("v5")
    maps = []
    for scale in (1., 10.):
        _, _, _, diag = lock_measurement(
            p, c * scale + 100 * ~mask, nodes[None], nodes, torch.zeros(1, 2), 15,
            box=(2, 3), support_mask=mask, static_residual="mask", residual_rel=.1,
            return_diagnostics=True)
        maps.append(diag["residual_weight"])
    energy = c.square().sum(1)
    expected = (energy / energy[:, mask].mean()).clamp(0, 1)
    expected *= mask & (energy > .1 * energy[:, mask].max())
    torch.testing.assert_close(maps[0], expected)
    torch.testing.assert_close(maps[1], expected)
    assert ((maps[0] > 0) & (maps[0] < 1)).any()
    _, evidence, locked, diag = lock_measurement(
        p, c, nodes[None], nodes, torch.zeros(1, 2), 15, box=(2, 3), support_mask=mask,
        static_residual="mask", residual_rel=1., return_diagnostics=True)
    assert torch.count_nonzero(evidence) == torch.count_nonzero(locked) == torch.count_nonzero(diag["residual_weight"]) == 0
    search = LocalSearch(c, 15, box=(2, 3), support_mask=mask, sigma=1., gaussian_radius=None)
    support = torch.ones(1, 1, 9, 11)
    _, usable = search(c[:, None], support)
    assert usable.any()
    _, weighted_usable = search(c[:, None], support, support_weight=.4 * support)
    assert not weighted_usable.any()  # all candidate validity masses are below .5


@pytest.mark.parametrize("kw", [dict(static_residual="unknown"), dict(residual_rel=-.1), dict(residual_rel=1.1)])
def test_residual_options_reject_invalid_values(kw):
    x = torch.zeros(1, 1, 7, 7)
    nodes = named_nodes("v5")
    with pytest.raises(ValueError):
        lock_measurement(x, x, nodes[None], nodes, torch.zeros(1, 2), 9, **kw)
