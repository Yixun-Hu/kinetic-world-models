"""CPU acceptance tests for the opt-in snapshot lock estimator.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1. The two dataset tests
intentionally exercise full cells on 20/100 sequences (several minutes).
"""
import importlib.util
from pathlib import Path

import pytest
import torch

from kwm.cells import AdaptiveNodeCell
from kwm.collision import lattice_nodes, reflection_perms, known_wall_field
from kwm.env import EnvConfig, make_batch
from kwm.equiv import boost_transform_nodes, equiv_error, interior_mask
from kwm.lock_nodes import LockState, LocalSearch, lock_measurement, two_modes
from kwm.model import build_model
from kwm.train import parse_args
from kwm.transport import spectral_shift
from test_galilean import _blob, _node


def _cell(**kwargs):
    torch.manual_seed(0)
    return AdaptiveNodeCell(C=kwargs.pop('C', 8), world=kwargs.pop('world', 24),
                            window=kwargs.pop('window', 16), node_update='lock', **kwargs)


def test_lk_bitwise_prechange():
    path = Path(__file__).parent / 'reference' / 'adaptive_cell_pre_lock.py'
    spec = importlib.util.spec_from_file_location('adaptive_pre_lock', path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    torch.manual_seed(14)
    old = ref.AdaptiveNodeCell(C=8, world=24, window=16)
    new = AdaptiveNodeCell(C=8, world=24, window=16, node_update='lk')
    new.load_state_dict(old.state_dict())
    frames, actions = torch.rand(2, 7, 1, 16, 16), torch.randn(2, 7, 2)
    ho = hn = torch.zeros(2, 25, 19, 24, 24)
    nodes = lattice_nodes()
    for t in range(7):
        old.update_nodes = new.update_nodes = t < 5
        ho = old(frames[:, t], ho, actions[:, t], nodes)
        hn = new(frames[:, t], hn, actions[:, t], nodes)
        assert torch.equal(ho, hn)
        assert torch.equal(old.last_nodes, new.last_nodes)
    ho.square().mean().backward()
    hn.square().mean().backward()
    for p, q in zip(old.parameters(), new.parameters()):
        assert torch.equal(p.grad, q.grad)
    assert parse_args([]).node_update == 'lk'
    assert build_model(vars(parse_args(['--node_update', 'lock']))).cell.node_update == 'lock'


@torch.no_grad()
def _dataset_run(seed, size, n, compare=False):
    from scripts.estimator_gate import Estimator, SPECS, metrics
    batch = make_batch(size, 30, 0, EnvConfig(real_velocity=True, real_camera=True),
                       gen=torch.Generator().manual_seed(seed), split='test')
    cell = _cell(C=32, world=50, window=32)
    nodes = lattice_nodes()
    trajectories = []
    all_max, final_max = 0., 0.
    for start in range(0, n, 2):
        frames = batch['frames'][start:start + 2]
        actions = batch['actions'][start:start + 2]
        state = cell.initial_state(frames[:, 0])
        if compare:
            ref = Estimator(next(s for s in SPECS if s.name == 'D_lock_LK_2mode_grid1_heavier'), len(frames), 'cpu')
        trajectory = []
        for t in range(30):
            state = cell(frames[:, t], state, actions[:, t], nodes)
            nu = nodes[None] + cell.offsets(state)
            trajectory.append(nu)
            if compare:
                expected = ref.step(cell.imprint(cell.conv_u(frames[:, t])), actions[:, t])
                error = (nu - expected).abs().max().item()
                all_max = max(all_max, error)
                if t == 29:
                    final_max = max(final_max, error)
        trajectories.append(torch.stack(trajectory, 1))
    trajectory = torch.cat(trajectories)
    result, _ = metrics(trajectory, batch['digit_states']['vel'][:n], batch['digit_states']['visible'][:n])
    return result, all_max, final_max


def test_gate_parity_20():
    # Seed-2024 gate data are the prefix of the original 300-sequence draw.
    result, all_max, final_max = _dataset_run(2024, 300, 20, compare=True)
    print(f'gate parity: all-step max={all_max:.9g}, final max={final_max:.9g}; {result}')
    assert final_max <= 1e-4
    assert all_max <= 1e-4


def test_accuracy_100():
    result, _, _ = _dataset_run(777, 100, 100)
    print(f'seed-777 accuracy: {result}')
    assert abs(result['matched_median'] / 0.0241 - 1) <= 0.1
    assert abs(result['matched_gt_0_1'] / 0.20 - 1) <= 0.1


@pytest.mark.parametrize('prediction', ['on', 'off'])
def test_checkpoint_functional(prediction):
    torch.manual_seed(4)
    obs, ain, aout = torch.rand(2, 4, 1, 8, 8), torch.randn(2, 4, 2), torch.randn(2, 3, 2)
    results = []
    for checkpoint in (False, True):
        torch.manual_seed(5)
        model = build_model(dict(model='adaptive', C=4, world=16, window=8,
                                node_update='lock', node_update_pred=prediction, grad_ckpt=checkpoint)).train()
        pred, aux = model(obs, ain, aout, record_nodes=True)
        loss = (pred - .3).square().mean()
        loss.backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters()}
        assert all(torch.isfinite(g).all() for g in grads.values())
        assert model.cell.last_nodes is None  # lock recurrence does not write diagnostics into the cell
        assert model.cell.n_s.grad.abs() > 0
        results.append((loss.detach(), grads, aux['node_traj']))
    assert torch.allclose(results[0][0], results[1][0], atol=1e-5, rtol=0)
    assert torch.equal(results[0][2], results[1][2])
    for n in results[0][1]:
        torch.testing.assert_close(results[0][1][n], results[1][1][n], atol=1e-6, rtol=1e-5)
    pred_traj = results[0][2][:, 4:]
    if prediction == 'off':
        assert torch.equal(pred_traj, results[0][2][:, 3:4].expand_as(pred_traj))
    else:
        assert (pred_traj - results[0][2][:, 3:4]).abs().max() > 1e-5


@torch.no_grad()
def test_state_replay_and_prediction_freeze():
    cell = _cell()
    nodes = lattice_nodes()
    frame, action = torch.rand(2, 1, 16, 16), torch.randn(2, 2)
    initial = cell.initial_state(frame)
    state = cell(frame, initial, action, nodes)
    saved = LockState(*(x.clone() for x in state))
    expected = cell(frame, state, action, nodes)
    # Interleave an unrelated sequence through the same module.
    cell(frame * 2, initial, -action, nodes)
    actual = cell(frame, saved, action, nodes)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    assert all(torch.equal(a, b) for a, b in zip(saved, state))
    frozen = cell(frame, state, action, nodes, update_nodes=False)
    assert torch.equal(frozen.nodes, state.nodes)
    assert torch.equal(frozen.snapshot, state.snapshot)
    assert torch.equal(frozen.camera, state.camera)
    assert state.snapshot.shape == (2, 8, 16, 16)
    assert state.snapshot.numel() == 2 * 8 * 16 * 16
    assert not any(isinstance(v, (torch.Tensor, LockState)) for k, v in vars(cell).items() if not k.startswith('_'))


@pytest.mark.parametrize('boost', [(1., 0.), (-1., 1.)])
@torch.no_grad()
def test_on_lattice_boost(boost):
    # Same H, T, blob centres, velocities and boosts as test_galilean's
    # synthetic transport test; exercise observation-driven node updates too.
    H, T = 48, 10
    nodes, u = lattice_nodes(), torch.tensor([boost])
    errors = {}
    for mode in ('lk', 'lock'):
        torch.manual_seed(0)
        cell = AdaptiveNodeCell(C=2, world=H, window=H, node_update=mode)
        state = cell.initial_state(torch.zeros(1, 1, H, H))
        boosted = cell.initial_state(torch.zeros(1, 1, H, H))
        for t in range(T):
            frame = (_blob(H, 20 + t, 20) + _blob(H, 30, 12 - t))[None, None]
            state = cell(frame, state, torch.zeros(1, 2), nodes)
            boosted = cell(frame, boosted, u if t else torch.zeros_like(u), nodes)
        transformed, mask = boost_transform_nodes(state, nodes, 1., 2, u, .75)
        errors[mode] = equiv_error(transformed, boosted, 2, mask & interior_mask(nodes, 1.)).item()
    print(f'on-lattice feature equivariance {boost}: {errors}')
    assert errors['lock'] <= errors['lk']


@torch.no_grad()
def test_off_lattice_locked_velocities():
    H, T = 32, 12
    velocity, boost = torch.tensor([.7, -1.3]), torch.tensor([.27, -.41])
    nodes = lattice_nodes()
    estimates = []
    for v in (velocity, velocity + boost):
        cell = _cell(world=H, window=H)
        state = cell.initial_state(torch.zeros(1, 1, H, H))
        for t in range(T):
            f = _blob(H, 10 + v[0] * t, 20 + v[1] * t)[None, None]
            state = cell(f, state, torch.zeros(1, 2), nodes)
        i = _node(nodes, v.round().tolist())
        assert state.nodes[0, i, 3] == 1
        estimates.append(nodes[i] + state.nodes[0, i, :2])
    error = (estimates[1] - estimates[0] - boost).abs().max()
    print(f'off-lattice locked velocity boost max error: {error.item():.9g}')
    assert error <= .05


def test_bf16_state_and_search():
    cell = _cell(world=16, window=8)
    frame, action, nodes = torch.rand(1, 1, 8, 8), torch.zeros(1, 2), lattice_nodes()
    state = cell.initial_state(frame)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        for _ in range(3):
            state = cell(frame, state, action, nodes)
            assert state.nodes.dtype == state.snapshot.dtype == torch.float32
            assert state.camera.dtype == torch.float32
        search = LocalSearch(state.snapshot.requires_grad_(), 16)
        d, _ = search(state.snapshot[:, None].expand(-1, 25, -1, -1, -1), torch.ones(1, 25, 8, 8))
        assert d.dtype == torch.float32 and not d.requires_grad
        assert search.shifted.dtype == torch.float32 and not search.shifted.requires_grad
    assert state.features.dtype == torch.bfloat16
    state.features.float().square().mean().backward()
    assert cell.n_s.grad is not None and torch.isfinite(cell.n_s.grad)


@pytest.mark.parametrize('velocity, expected_lock', [(.2, 1.), (.5, 0.)])
def test_displacement_gradient_and_scale(velocity, expected_lock):
    axis = torch.arange(32).float()
    previous = torch.exp(-((axis[None] - 16).square() + (axis[:, None] - 16).square()) / 12)
    previous = previous[None, None].expand(1, 8, 32, 32).clone()
    current = spectral_shift(previous, torch.tensor([[velocity, 0.]])).detach().requires_grad_()
    nu = torch.zeros(1, 1, 2)
    delta, evidence, locked = lock_measurement(previous, current, nu, nu[0], torch.zeros(1, 2), 32)
    assert locked.item() == expected_lock and evidence.item() > 0
    gradient = torch.autograd.grad(delta.sum(), current, retain_graph=True)[0]
    if expected_lock:
        assert gradient.abs().sum() > 0
    else:
        assert torch.count_nonzero(gradient) == 0
    # The running mean and scale remain differentiable for both routes.
    s = torch.tensor(.2, requires_grad=True)
    step = (s * evidence[..., None] / (evidence[..., None] + 1) * delta).clamp(-.3, .3)
    step.sum().backward()
    assert s.grad.abs() > 0


def test_heavier_mode_and_boundary_rejection():
    d = torch.zeros(1, 1, 2, 8, 8)
    d[:, :, 0, :4], d[:, :, 0, 4:] = -.2, .4
    weight = torch.ones(1, 1, 8, 8)
    weight[:, :, 4:] = 2
    primary, mass, selected = two_modes(d.requires_grad_(), weight, torch.zeros(1, 1, 2),
                                        torch.zeros(1, 2), 1.)
    torch.testing.assert_close(primary[0, 0], torch.tensor([.4, 0.]))
    assert mass.item() == 64 and not primary.requires_grad and not selected.requires_grad
    previous = _blob(32, 15, 15)[None, None]
    current = spectral_shift(previous, torch.tensor([[2., 0.]]))
    displacement, usable = LocalSearch(current, 32)(previous[:, None], torch.ones(1, 1, 32, 32))
    edge = (displacement.abs() >= 1).any(2)
    assert edge.any() and not (edge & usable).any()


@torch.no_grad()
def test_permutation_and_no_host_sync():
    cell = _cell(world=16, window=8)
    frame, action, nodes = torch.rand(2, 1, 8, 8), torch.randn(2, 2), lattice_nodes()
    state = cell.initial_state(frame)
    for _ in range(3):
        state = cell(frame, state, action, nodes)
    perm = torch.randperm(25)
    permuted = LockState(state.features[:, perm], state.nodes[:, perm], state.snapshot, state.camera)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        out = cell(frame, state, action, nodes)
    forbidden = {'aten::item', 'aten::_local_scalar_dense'}
    assert not forbidden.intersection(e.key for e in prof.key_averages())
    other = cell(frame, permuted, action, nodes[perm])
    torch.testing.assert_close(out.features[:, perm], other.features, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out.nodes[:, perm], other.nodes, rtol=1e-5, atol=1e-5)
    assert torch.equal(out.snapshot, other.snapshot)


@torch.no_grad()
def test_bounce_and_boost_shared_snapshot():
    cell = _cell(C=2, world=24, window=8)
    nodes = lattice_nodes()
    px, py = reflection_perms(nodes)
    state = cell.initial_state(torch.zeros(1, 1, 8, 8))
    i, j = _node(nodes, (-1., 0.)), _node(nodes, (1., 0.))
    state.features[0, i, 0, 10:14, 1:5] = 1
    state.nodes[0, i] = torch.tensor([-.3, .2, 5., 1.])
    out = cell.bounce(state, nodes, px, py, known_wall_field(24, band=2)[None])
    assert out.features[0, j, 0].sum() == 16
    torch.testing.assert_close(out.nodes[0, j], torch.tensor([.3, .2, 5., 1.]))
    assert out.snapshot is state.snapshot
    boosted, mask = boost_transform_nodes(state, nodes, 1., 2, torch.tensor([[1., 0.]]), .75)
    assert boosted.snapshot is state.snapshot
    assert boosted.nodes.dtype == torch.float32
    assert mask.any()
