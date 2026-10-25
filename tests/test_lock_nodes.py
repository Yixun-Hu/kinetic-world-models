"""CPU acceptance tests for the opt-in snapshot lock estimator.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1. The two dataset cases
use 12-sequence prefixes and require -m slow or KWM_SLOW=1.
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


def _blob(size, cx, cy, s=2.5):
    axis = torch.arange(size).float()
    dx = (axis[None] - cx + size / 2) % size - size / 2
    dy = (axis[:, None] - cy + size / 2) % size - size / 2
    return torch.exp(-(dx.square() + dy.square()) / (2 * s**2))


def _node(nodes, velocity):
    return int(((nodes - torch.tensor(velocity)).abs().sum(1) < 1e-6).nonzero())


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


@pytest.mark.parametrize('update', [False, True])
@torch.no_grad()
def test_lk_update_nodes_override(update):
    torch.manual_seed(14)
    cell = AdaptiveNodeCell(C=2, world=16, window=8)
    frame, action, nodes = torch.rand(1, 1, 8, 8), torch.randn(1, 2), lattice_nodes()
    state = cell(frame, cell.initial_state(frame), action, nodes)
    cell.update_nodes = update
    expected = cell(frame, state, action, nodes)
    cell.update_nodes = not update
    actual = cell(frame, state, action, nodes, update_nodes=update)
    assert torch.equal(actual, expected)
    assert cell.update_nodes is not update


@torch.no_grad()
def test_shared_fft_node_parity(monkeypatch):
    import kwm.lock_nodes as lock

    def repeated_fft(snapshot, shifts, world):
        # Pre-optimization transport: repeat the snapshot before its FFT.
        full = lock.pad_world(snapshot, world)[:, None].expand(-1, shifts.shape[1], -1, -1, -1)
        return spectral_shift(full, shifts)

    cell = _cell(C=8, world=50, window=32)
    frames, actions = torch.rand(2, 6, 1, 32, 32), torch.randn(2, 6, 2)
    # Eight channels / world 50 also exercise complex-multiply layout rounding.
    snapshot = torch.rand(2, 8, 32, 32)
    shifts = torch.randn(2, 25, 2)
    torch.testing.assert_close(lock.shift_snapshot(snapshot, shifts, 50),
                               repeated_fft(snapshot, shifts, 50), rtol=0, atol=1e-7)
    nodes, trajectories = lattice_nodes(), []
    for transport in (lock.shift_snapshot, repeated_fft):
        monkeypatch.setattr(lock, 'shift_snapshot', transport)
        state, trajectory = cell.initial_state(frames[:, 0]), []
        for t in range(frames.shape[1]):
            state = cell(frames[:, t], state, actions[:, t], nodes)
            trajectory.append(nodes[None] + state.nodes[..., :2])
        trajectories.append(torch.stack(trajectory))
    torch.testing.assert_close(trajectories[0], trajectories[1], rtol=0, atol=1e-6)
    first, second = LocalSearch(state.snapshot, 50), LocalSearch(state.snapshot[:1], 50)
    assert first.mask is second.mask
    assert torch.equal(first.mask, lock.shifted_window_mask(-first.shifts, 50, 32)[None, None])


@pytest.mark.parametrize('mode', ['lk', 'lock'])
@pytest.mark.parametrize('prediction', ['on', 'off'])
@torch.no_grad()
def test_panorama_rollout(mode, prediction):
    from scripts.panorama_figs import run_model
    torch.manual_seed(4)
    model = build_model(dict(model='adaptive', C=2, world=16, window=8,
                             node_update=mode, node_update_pred=prediction)).eval()
    frames, actions = torch.rand(1, 5, 1, 8, 8), torch.randn(1, 5, 2)
    actual, memory, nodes = run_model(model, frames, actions, 3, 2)
    expected, aux = model(frames[:, :3], actions[:, :3], actions[:, 3:], record_nodes=True)
    torch.testing.assert_close(actual, expected.clamp(0, 1), rtol=0, atol=1e-6)
    torch.testing.assert_close(nodes, aux['node_traj'][:, :3], rtol=0, atol=1e-6)
    assert memory.shape == (1, 5, 16, 16)
    assert model.state_extra == (11 if mode == 'lk' else 0)


@pytest.mark.parametrize('saved_mode', [None, 'lock'])
def test_eval_node_update_override(tmp_path, saved_mode):
    from kwm.eval import load_model
    args = dict(model='adaptive', C=2, world=16, window=8)
    if saved_mode:
        args['node_update'] = saved_mode
    model = build_model(args)
    torch.save(dict(args=args, model=model.state_dict(), step=7), tmp_path / 'best.pt')
    for override in (None, 'lk', 'lock'):
        loaded, loaded_args, step = load_model(tmp_path, 'best', 'cpu', node_update=override)
        assert loaded.node_update == (override or saved_mode or 'lk')
        assert loaded_args.get('node_update', 'lk') == loaded.node_update
        assert step == 7
        for name, value in model.state_dict().items():
            assert torch.equal(value, loaded.state_dict()[name])


def _assignment(cost):
    """Small rectangular Hungarian assignment, with rows <= columns.

    CPU-only metric helper: successive shortest augmenting paths with row and
    column potentials. Assign all digits before filtering visibility, exactly
    as in the gate, without importing the gate or its optional dependencies.
    """
    costs = cost.tolist()
    n, m = cost.shape
    u, v = [0.] * (n + 1), [0.] * (m + 1)
    matched, previous = [0] * (m + 1), [0] * (m + 1)
    for row in range(1, n + 1):
        matched[0] = row
        column = 0
        distance, used = [float("inf")] * (m + 1), [False] * (m + 1)
        while True:
            used[column] = True
            current_row = matched[column]
            delta, next_column = float("inf"), 0
            for candidate in range(1, m + 1):
                if not used[candidate]:
                    reduced = costs[current_row - 1][candidate - 1] - u[current_row] - v[candidate]
                    if reduced < distance[candidate]:
                        distance[candidate], previous[candidate] = reduced, column
                    if distance[candidate] < delta:
                        delta, next_column = distance[candidate], candidate
            for candidate in range(m + 1):
                if used[candidate]:
                    u[matched[candidate]] += delta
                    v[candidate] -= delta
                else:
                    distance[candidate] -= delta
            column = next_column
            if matched[column] == 0:
                break
        while column:
            parent = previous[column]
            matched[column] = matched[parent]
            column = parent
    assignment = [0] * n
    for column in range(1, m + 1):
        if matched[column]:
            assignment[matched[column] - 1] = column - 1
    return assignment


def test_assignment_minimizes_total_cost():
    # Greedy matching would take (0,0) and force row 1 onto cost 100.
    assert _assignment(torch.tensor([[1., 2., 50.], [1.1, 100., 100.]])) == [1, 0]


@torch.no_grad()
def _dataset_run(seed, size, n):
    batch = make_batch(size, 30, 0, EnvConfig(real_velocity=True, real_camera=True),
                       gen=torch.Generator().manual_seed(seed), split='test')
    cell = _cell(C=32, world=50, window=32)
    nodes = lattice_nodes()
    final_nodes = []
    for start in range(0, n, 2):
        frames = batch['frames'][start:start + 2]
        actions = batch['actions'][start:start + 2]
        state = cell.initial_state(frames[:, 0])
        for t in range(30):
            state = cell(frames[:, t], state, actions[:, t], nodes)
        final_nodes.append(nodes[None] + cell.offsets(state))
    distance = (batch['digit_states']['vel'][:n, -1, :, None] - torch.cat(final_nodes)[:, None]).norm(dim=-1)
    matched = torch.stack([cost[torch.arange(cost.shape[0]), _assignment(cost)] for cost in distance])
    visible = batch['digit_states']['visible'][:n, -10:].any(1)
    errors = matched[visible]
    return dict(matched_median=errors.median().item(), matched_gt_0_1=(errors > .1).float().mean().item())


@pytest.mark.slow
@pytest.mark.parametrize('seed,size,median,miss', [
    (2024, 300, 0.015656080096960068, 0.15000000596046448),
    (777, 100, 0.023351069539785385, 0.18333333730697632),
])
def test_accuracy_final_rule(seed, size, median, miss):
    # Frozen final-rule gate values for the first 12 sequences of each draw;
    # full 100-sequence results and the prefix protocol are documented.
    result = _dataset_run(seed, size, 12)
    print(f'seed-{seed} accuracy: {result}')
    assert result['matched_median'] <= 1.1 * median
    assert result['matched_gt_0_1'] <= 1.1 * miss


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
    assert errors['lock'] < 1e-6
    assert errors['lk'] < 1e-6


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
    # Exercise the actual cell update (including gain/clamps), on both routes.
    cell = AdaptiveNodeCell(C=1, imprint_k=1, num_v=1, world=32, window=32,
                            node_update='lock', s_init=.2)
    with torch.no_grad():
        cell.conv_u.weight.zero_()
        cell.conv_u.weight[0, 0, 1, 1] = 1
        cell.P.fill_(1)
    state = cell.initial_state(current[:, :1])
    state = state._replace(snapshot=previous[:, :1])
    state.nodes[..., 2] = 1
    out = cell(current[:, :1].detach(), state, torch.zeros(1, 2), nu[0])
    assert out.nodes[0, 0, 3].item() == expected_lock
    out.nodes[..., :2].sum().backward()
    assert cell.n_s.grad is not None and cell.n_s.grad.abs() > 0


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


def test_permutation_and_no_host_sync():
    cell = _cell(world=16, window=8)
    frame, action, nodes = torch.rand(2, 1, 8, 8), torch.randn(2, 2), lattice_nodes()
    state = cell.initial_state(frame)
    for _ in range(3):
        state = cell(frame, state, action, nodes)
    perm = torch.randperm(25)
    permuted = LockState(state.features[:, perm], state.nodes[:, perm], state.snapshot)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        out = cell(frame, state, action, nodes)
        (out.features.square().mean() + out.nodes[..., :2].square().mean()).backward()
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
    state.nodes[0, i, :4] = torch.tensor([-.3, .2, 5., 1.])
    out = cell.bounce(state, nodes, px, py, known_wall_field(24, band=2)[None])
    assert out.features[0, j, 0].sum() == 16
    torch.testing.assert_close(out.nodes[0, j, :4], torch.tensor([.3, .2, 5., 1.]))
    assert out.snapshot is state.snapshot
    boosted, mask = boost_transform_nodes(state, nodes, 1., 2, torch.tensor([[1., 0.]]), .75)
    assert boosted.snapshot is state.snapshot
    assert boosted.nodes.dtype == torch.float32
    assert mask.any()
