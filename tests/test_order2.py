"""PR2 CPU acceptance: frozen PR1 arithmetic before acceleration diagnostics."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from kwm.cells import AdaptiveNodeCell
from kwm.collision import lattice_nodes
from kwm import env
from kwm.lock_nodes import (NODE_DIM, SW, MT, CTT, MZ, CTZ, SW2, CZZ, MODE, TIME, CORR,
                            acceleration_fit, accumulate_acceleration, transport_velocity, fit_correction,
                            corrected_offsets, clamp_mask)
from kwm.model import build_model
from kwm.train import parse_args


def reference(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / 'reference' / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


reference('lock_nodes_pre_order2', 'lock_nodes_pre_order2.py')
old_cells = reference('cells_pre_order2', 'cells_pre_order2.py')
old_env = reference('env_pre_accel', 'env_pre_accel.py')
old_model = reference('model_pre_order2', 'model_pre_order2.py')


@pytest.mark.parametrize('mode', ['lk', 'lock'])
@pytest.mark.parametrize('dt', [1., .5])
def test_order1_bitwise_forward_state_and_gradients(mode, dt):
    torch.manual_seed(14)
    kw = dict(C=2, world=16, window=8, imprint_k=2, node_update=mode)
    old, new = old_cells.AdaptiveNodeCell(**kw), AdaptiveNodeCell(**kw, nu_forget=1., pred_shrink='none')
    new.load_state_dict(old.state_dict())
    frames = torch.rand(1, 5, 1, 8, 8)
    old_input = frames.clone().requires_grad_()
    new_input = frames.clone().requires_grad_()
    actions, nodes = torch.randn(1, 5, 2), lattice_nodes()
    ho, hn = old.initial_state(frames[:, 0]), new.initial_state(frames[:, 0])
    for t in range(5):
        ho = old(old_input[:, t], ho, actions[:, t], nodes, dt, update_nodes=t < 4)
        hn = new(new_input[:, t], hn, actions[:, t], nodes, dt, update_nodes=t < 4, prediction=t >= 3)
        if mode == 'lock':
            assert hn.nodes.shape[-1] == NODE_DIM
            assert torch.equal(ho.features, hn.features)
            assert torch.equal(ho.nodes, hn.nodes[..., :4])
            assert torch.count_nonzero(hn.nodes[..., 4:]) == 0
            assert torch.equal(ho.snapshot, hn.snapshot)
        else:
            assert torch.equal(ho, hn)
    (ho.features if mode == 'lock' else ho).square().mean().backward()
    (hn.features if mode == 'lock' else hn).square().mean().backward()
    assert torch.equal(old_input.grad, new_input.grad)
    for (name, p), (_, q) in zip(old.named_parameters(), new.named_parameters()):
        assert torch.equal(p.grad, q.grad), name


@pytest.mark.parametrize('prediction', ['on', 'off'])
@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('mode', ['lk', 'lock'])
def test_order1_bitwise_frozen_model(prediction, checkpoint, mode):
    # Frozen model/cells/lock implementation at 0c1198a; imports only redirected.
    torch.manual_seed(42)
    frames = torch.rand(1, 5, 1, 8, 8)
    ain, aout = torch.randn(1, 5, 2), torch.randn(1, 3, 2)
    kw = dict(model='adaptive', C=2, world=16, window=8, imprint_k=2,
              node_update=mode, node_update_pred=prediction, grad_ckpt=checkpoint)
    old, new = old_model.build_model(kw), build_model(kw)
    new.load_state_dict(old.state_dict())
    x, y = frames.clone().requires_grad_(), frames.clone().requires_grad_()
    p, a = old(x, ain, aout, dt=.5, record_nodes=True, return_states=[4])
    q, b = new(y, ain, aout, dt=.5, record_nodes=True, return_states=[4])
    assert torch.equal(p, q) and torch.equal(a['node_traj'], b['node_traj'])
    if mode == 'lock':
        ho, hn = a['states'][4], b['states'][4]
        assert torch.equal(ho.features, hn.features) and torch.equal(ho.snapshot, hn.snapshot)
        assert torch.equal(ho.nodes, hn.nodes[..., :4])
        assert not hn.nodes[..., 4:].any()
        assert (hn.nodes[..., 2] > 0).any() and (hn.nodes[..., 3] > 0).any()
    else:
        assert torch.equal(a['states'][4], b['states'][4])
    p.square().mean().backward()
    q.square().mean().backward()
    assert torch.equal(x.grad, y.grad)
    for (name, pp), (_, qq) in zip(old.named_parameters(), new.named_parameters()):
        assert torch.equal(pp.grad, qq.grad), name


@pytest.mark.parametrize('prediction', ['on', 'off'])
@pytest.mark.parametrize('checkpoint', [False, True])
def test_infinite_gate_bitwise_model(prediction, checkpoint):
    torch.manual_seed(42)
    obs, ain, aout = torch.rand(1, 5, 1, 8, 8), torch.randn(1, 5, 2), torch.randn(1, 3, 2)
    kw = dict(model='adaptive', C=2, world=16, window=8, imprint_k=2, node_update='lock',
              node_update_pred=prediction, grad_ckpt=checkpoint)
    one, two = build_model(kw), build_model(dict(kw, order=2, accel_kappa=float('inf')))
    two.load_state_dict(one.state_dict())
    p, aux = one(obs, ain, aout, record_nodes=True, return_states=[4])
    q, aux2 = two(obs, ain, aout, record_nodes=True, return_states=[4])
    assert torch.equal(p, q)
    assert torch.equal(aux['node_traj'], aux2['node_traj'])
    for key in ('features', 'snapshot'):
        assert torch.equal(getattr(aux['states'][4], key), getattr(aux2['states'][4], key))
    assert torch.equal(aux['states'][4].nodes, aux2['states'][4].nodes)
    p.square().mean().backward(); q.square().mean().backward()
    for (name, pp), (_, qq) in zip(one.named_parameters(), two.named_parameters()):
        assert torch.equal(pp.grad, qq.grad), name


@pytest.mark.parametrize('settings', [{}, dict(real_velocity=True, real_camera=True),
    dict(real_velocity=True, real_camera=True, walls=True, dt=.5)])
def test_env_disabled_bitwise(settings, monkeypatch):
    # Read the existing bank; never download into the reference directory.
    monkeypatch.setattr(old_env, 'load_mnist', env.load_mnist)
    old = old_env.make_batch(2, 4, 2, old_env.EnvConfig(**settings), gen=torch.Generator().manual_seed(5))
    new = env.make_batch(2, 4, 2, env.EnvConfig(**settings), gen=torch.Generator().manual_seed(5))
    for key in ('frames', 'actions', 'actions_noisy', 'bounce', 'bounce_events', 'scene_boost'):
        assert torch.equal(old[key], new[key]), key
    for key in old['digit_states']:
        assert torch.equal(old['digit_states'][key], new['digit_states'][key]), key
    assert not new['digit_states']['acc'].any()
    assert old_env.EnvConfig(**settings).tag() == env.EnvConfig(**settings).tag()


def test_accel_tags_validation_and_cli():
    real = env.EnvConfig(real_velocity=True, real_camera=True)
    accel = env.EnvConfig(real_velocity=True, real_camera=True, accel=.02)
    assert accel.tag() == real.tag() + '_acc0.02'
    assert 'vel1' in env.EnvConfig(accel=.02).tag()
    assert env.EnvConfig(accel_zero_p=.1).tag() == env.EnvConfig().tag()
    assert 'accel_zero_p0.1' in env.EnvConfig(accel=.02, accel_zero_p=.1).tag()
    with pytest.raises(ValueError, match='walls'):
        env.EnvConfig(accel=.02, walls=True)
    with pytest.raises(ValueError, match='walls'):
        parse_args(['--accel', '.02', '--walls'])
    with pytest.raises(ValueError, match='order 2'):
        parse_args(['--order', '2'])
    with pytest.raises(ValueError, match='order 2'):
        AdaptiveNodeCell(order=2)
    args = parse_args(['--node_update', 'lock', '--order', '2', '--accel_kappa', 'inf'])
    assert args.accel_kappa == float('inf') and build_model(vars(args)).order == 2
    assert parse_args(['--node_update', 'lock', '--order', '2']).accel_kappa == 4
    with pytest.raises(SystemExit):
        parse_args(['--accel_var_inflation', '4'])


@pytest.mark.parametrize('dt', [1., .5])
def test_accel_bounds_closed_form_and_velocity(dt, monkeypatch):
    cfg = env.EnvConfig(real_velocity=True, real_camera=True, accel=.02, dt=dt)
    v, a, stats = env.sample_acceleration(torch.Generator().manual_seed(7), (10000,), cfg, return_stats=True)
    for time in (0, 20, 50, 150, 180):
        assert (v + a * time).abs().max() <= cfg.vmax
    assert .71 < stats['acceptance'] < .74
    sampled = []
    sampler = env.sample_acceleration
    def capture(*args, **kwargs):
        result = sampler(*args, **kwargs)
        sampled.append(result[0].clone())
        return result
    monkeypatch.setattr(env, 'sample_acceleration', capture)
    batch = env.make_batch(2, 180, 0, cfg, gen=torch.Generator().manual_seed(5))
    ds = batch['digit_states']; a = ds['acc'].double()
    v0 = sampled[0].double()
    p = ds['cent'][:, 0].double()
    p0 = p.clone()
    for t in range(180):
        exact = p0 + v0 * (t * dt) + .5 * a * (t * dt)**2
        if t:
            p = p + (v0 + a * ((t - .5) * dt)) * dt
            exact = exact.remainder(cfg.world)
        torch.testing.assert_close(p.remainder(cfg.world) if t else p, exact, atol=1e-5, rtol=0)
        torch.testing.assert_close(ds['cent'][:, t].double(), exact, atol=1e-5, rtol=0)
        torch.testing.assert_close(ds['vel'][:, t].double(), v0 + a * ((t + .5) * dt), atol=2e-7, rtol=0)
    if dt == .5:
        sub = env.subsample(batch, 2)['digit_states']
        torch.testing.assert_close(sub['vel'], ds['vel'][:, ::2] + ds['acc'][:, None] * .25)


def fit_linear(dt=1., a=(.004, -.002), velocity=(.2, .1), steps=30):
    state = torch.zeros(1, 1, NODE_DIM)
    a, v = torch.tensor(a), torch.tensor(velocity)
    for t in range(steps):
        time = torch.full((1, 1), t * dt)
        state = accumulate_acceleration(state, (v + a * time[..., None]), torch.ones(1, 1), time,
                                        torch.ones(1, 1), torch.zeros(1, 1))
        state[..., 3] = 1
    return state


def test_weighted_empirical_variance_and_detachment():
    gen = torch.Generator().manual_seed(7)
    times = torch.arange(30).float() * .5
    weight = torch.rand(30, generator=gen) + .5
    measured = torch.tensor([.3, -.2]) + times[:, None] * torch.tensor([.004, .002]) + .01 * torch.randn(30, 2, generator=gen)
    measured.requires_grad_()
    state = torch.zeros(1, 1, NODE_DIM)
    for t, w, z in zip(times, weight, measured):
        state = accumulate_acceleration(state, z[None, None], w[None, None], t,
                                        torch.ones(1, 1), torch.zeros(1, 1))
    slope, gate, var, neff = acceleration_fit(state)
    tw = (weight * times).sum() / weight.sum()
    zw = (weight[:, None] * measured).sum(0) / weight.sum()
    denom = (weight * (times - tw).square()).sum()
    expected = (weight[:, None] * (times - tw)[:, None] * (measured - zw)).sum(0) / denom
    residual = measured - zw - (times - tw)[:, None] * expected
    effective = weight.sum().square() / weight.square().sum()
    variance = ((weight * residual.square().sum(-1)).sum() / (weight.sum() * (1 - 2 / effective))
                / denom * weight.square().sum() / weight.sum())
    torch.testing.assert_close(slope[0, 0], expected, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(var[0, 0], variance, atol=1e-9, rtol=1e-4)
    torch.testing.assert_close(neff[0, 0], effective)
    assert not any(x.requires_grad for x in (state, slope, gate, var))


def test_weight_scale_invariance():
    gen = torch.Generator().manual_seed(18)
    z = .02 * torch.randn(30, 32, 2, generator=gen)
    z[:, 16:] += torch.arange(30)[:, None, None] * .004
    weights = torch.exp(torch.rand(30, 32, generator=gen) * 7 - 4)
    fitted = []
    for scale in (1., 1000.):
        state = torch.zeros(32, NODE_DIM)
        for t in range(30):
            state = accumulate_acceleration(state, z[t], weights[t] * scale, float(t),
                                            torch.ones(32), torch.zeros(32))
        fitted.append(acceleration_fit(state))
    for actual, expected in zip(fitted[1], fitted[0]):
        torch.testing.assert_close(actual, expected, rtol=5e-5, atol=2e-6)
    assert (fitted[0][1] > 0).any() and (fitted[0][1] == 0).any()


def test_heterogeneous_weight_null_calibration():
    # Independent empirical coverage, not a second implementation of the fit.
    gen = torch.Generator().manual_seed(83)
    n = 4096
    state = torch.zeros(n, NODE_DIM)
    scale = torch.tensor([.1, 1., 10., 40.]).repeat(n // 4)
    for t in range(60):
        w = torch.exp(torch.rand(n, generator=gen) * 7 - 4) * scale
        sigma = .01 + .02 * torch.rand(n, 1, generator=gen)
        z = torch.tensor([.2, -.3]) + sigma * torch.randn(n, 2, generator=gen)
        state = accumulate_acceleration(state, z, w, float(t), torch.ones(n), torch.zeros(n))
    slope, gate, variance, neff = acceleration_fit(state, kappa=9)
    assert (neff >= 4).float().mean() > .95
    assert (gate > 0).float().mean() < .02
    empirical_trace = slope.var(0, unbiased=True).sum()
    assert .65 < variance.mean() / empirical_trace < 1.5


@pytest.mark.parametrize('dt', [1., .5])
def test_constant_acceleration_composes_exactly(dt):
    # Binary-exact motion isolates recurrence composition from FFT rounding.
    a = torch.tensor([1 / 256, -1 / 512])
    state = fit_linear(dt=dt, a=a.tolist(), velocity=(0., 0.))
    state[..., :2] = torch.tensor([.0625, -.03125])
    state[..., TIME] = 30 * dt
    state[..., CORR] = fit_correction(state, state[..., :2], 29 * dt, kappa=0.)
    start = corrected_offsets(state).clone()
    cell = AdaptiveNodeCell(C=1, imprint_k=1, num_v=1, node_update='lock', order=2, accel_kappa=0.)
    lattice, dummy = torch.zeros(1, 2), torch.zeros(1, 1, 32, 32)
    position = torch.zeros(1, 1, 2)
    slope0 = acceleration_fit(state, 0.)[0]
    for _ in range(100):
        position += transport_velocity(state, lattice, dt, kappa=0.) * dt
        state = cell.lock_update(dummy, dummy, state, torch.zeros(1, 2), lattice, dt,
                                 update_nodes=False, prediction=True)
        assert torch.equal(acceleration_fit(state, 0.)[0], slope0)
    tau = 100 * dt
    torch.testing.assert_close(position, start * tau + .5 * a * tau**2, atol=1e-6, rtol=0)


def test_reset_on_measured_jump_and_unlock_not_mode_labels():
    state = fit_linear()
    state[..., MODE], state[..., 3] = 1, 1
    predicted = state[..., MZ] + acceleration_fit(state)[0] * (30 - state[..., MT])[..., None]
    for lock, mode in ((1., 1.), (0., 1.)):
        z = torch.tensor([[[.8, .1]]]) if lock else predicted
        out = accumulate_acceleration(state, z, torch.ones(1, 1),
                                      30., torch.full((1, 1), lock), torch.full((1, 1), mode))
        assert out[..., SW] == 1
        assert out[..., CTT] == 0
        slope, gate, _, n = acceleration_fit(out)
        assert n == 1 and not slope.any() and not gate.any()
    same = accumulate_acceleration(state, predicted, torch.ones(1, 1), 30.,
                                   torch.ones(1, 1), torch.full((1, 1), 2.))
    assert same[..., SW] == 31
    young = fit_linear(steps=2)
    out = accumulate_acceleration(young, torch.tensor([[[.8, .1]]]), torch.ones(1, 1), 2.,
                                  torch.ones(1, 1), torch.zeros(1, 1))
    assert out[..., SW] == 3 and acceleration_fit(out)[1] == 0
    assert acceleration_fit(fit_linear(steps=3))[1] == 0
    assert acceleration_fit(fit_linear(steps=4))[1] > .99


def test_accel_half_dt_trajectory_same_law():
    one = env.make_batch(1, 30, 0, env.EnvConfig(real_velocity=True, real_camera=True, accel=.02),
                         gen=torch.Generator().manual_seed(12))
    half = env.make_batch(1, 59, 0, env.EnvConfig(real_velocity=True, real_camera=True, accel=.02, dt=.5),
                          gen=torch.Generator().manual_seed(12))
    assert torch.equal(one['digit_states']['acc'], half['digit_states']['acc'])
    assert torch.equal(one['frames'], half['frames'][:, ::2])
    torch.testing.assert_close(one['digit_states']['vel'], env.subsample(half, 2)['digit_states']['vel'], atol=2e-7, rtol=0)


def test_boost_preserves_acceleration_and_shifts_mean():
    from kwm.equiv import boost_transform_nodes
    from kwm.lock_nodes import LockState
    nodes = lattice_nodes()
    state = fit_linear().expand(1, 25, -1).clone()
    state[..., :2] = .1
    features = torch.ones(1, 25, 1, 8, 8)
    snapshot = torch.zeros(1, 1, 8, 8)
    boost = torch.tensor([[1., 0.]])
    actual, mask = boost_transform_nodes(LockState(features, state, snapshot), nodes, 1., 1, boost, .75)
    slope, gate, _, _ = acceleration_fit(actual.nodes)
    before, g0, _, _ = acceleration_fit(state)
    torch.testing.assert_close(slope[mask], before[mask], atol=1e-7, rtol=0)
    torch.testing.assert_close(gate[mask], g0[mask], atol=1e-6, rtol=0)
    torch.testing.assert_close(actual.nodes[..., MZ][mask], (state[..., MZ] + boost[:, None])[mask], atol=2e-7, rtol=0)
    assert actual.snapshot is snapshot


def test_bounce_resets_order2_fit():
    from kwm.collision import reflection_perms, known_wall_field
    from kwm.lock_nodes import LockState
    cell = AdaptiveNodeCell(C=1, imprint_k=1, world=24, window=8, node_update='lock', order=2)
    nodes = lattice_nodes(); px, py = reflection_perms(nodes)
    state = cell.initial_state(torch.zeros(1, 1, 8, 8))
    i = int(((nodes - torch.tensor([-1., 0.])).abs().sum(-1) == 0).nonzero())
    j = int(px[i])
    state.features[0, i, 0, 10:14, 1:5] = 1
    state.nodes[0, i] = fit_linear()[0, 0]
    state.nodes[0, i, :4] = torch.tensor([-.3, .2, 5., 1.])
    out = cell.bounce(state, nodes, px, py, known_wall_field(24, band=2)[None])
    torch.testing.assert_close(out.nodes[0, j, :4], torch.tensor([.3, .2, 5., 1.]))
    assert not out.nodes[0, j, 4:14].any() and not out.nodes[0, j, CORR].any()
    assert out.snapshot is state.snapshot


@pytest.mark.parametrize('prediction', ['on', 'off'])
def test_checkpoint_with_active_gate_and_shrink(prediction):
    torch.manual_seed(7)
    obs, ain, aout = torch.rand(1, 6, 1, 8, 8), torch.randn(1, 6, 2), torch.randn(1, 3, 2)
    kw = dict(model='adaptive', C=2, world=16, window=8, imprint_k=2, node_update='lock', order=2,
              accel_kappa=0., node_update_pred=prediction, pred_shrink='learned')
    normal, checkpoint = build_model(kw), build_model(dict(kw, grad_ckpt=True))
    checkpoint.load_state_dict(normal.state_dict())
    p, a = normal(obs, ain, aout, record_nodes=True)
    q, b = checkpoint(obs, ain, aout, record_nodes=True)
    assert torch.equal(p, q) and torch.equal(a['node_traj'], b['node_traj'])
    p.square().mean().backward(); q.square().mean().backward()
    for (name, x), (_, y) in zip(normal.named_parameters(), checkpoint.named_parameters()):
        assert torch.equal(x.grad, y.grad), name
    assert normal.cell.shrink_logit.grad.abs() > 0
    assert abs(float(normal.cell.shrink_logit.detach().sigmoid()) - .999) < 1e-7
    assert 0 <= a['clamp_rate_obs'] <= 1 and 0 <= a['clamp_rate_pred'] <= 1


def test_old_described_cache_loads_with_same_name(tmp_path, monkeypatch):
    from kwm import train
    from kwm.provenance import env_fingerprint
    monkeypatch.setattr(train, 'ROOT', str(tmp_path))
    cfg = env.EnvConfig(real_velocity=True, real_camera=True)
    cache = train.val_cache(cfg, 1, 2, 1)
    old_cfg = {k: v for k, v in cache['cfg'].items() if not k.startswith('accel')}
    cache['cfg'] = old_cfg
    cache['env_sha256'] = env_fingerprint(old_cfg)['sha256']
    path = tmp_path / 'results' / f'val_cache_{cfg.tag()}_n1_T2+1.pt'
    torch.save(cache, path)
    before = path.read_bytes()
    loaded = train.val_cache(cfg, 1, 2, 1)
    assert torch.equal(loaded['frames'], cache['frames']) and before == path.read_bytes()


def test_eval_order2_overrides(tmp_path):
    from kwm.eval import load_model
    args = dict(model='adaptive', C=2, world=16, window=8, node_update='lock')
    model = build_model(args)
    torch.save(dict(args=args, model=model.state_dict(), step=7), tmp_path / 'last.pt')
    loaded, flags, _ = load_model(tmp_path, 'last', 'cpu', order=2, accel_kappa=float('inf'), nu_forget=.9)
    assert loaded.order == flags['order'] == 2
    assert loaded.cell.accel_kappa == float('inf') and loaded.cell.nu_forget == .9


def test_synthetic_ma1_null_gate_below_two_percent():
    # Constant velocity + shared-frame position errors: displacement noise is
    # MA(1). This checks the variance gate, separately from MNIST association.
    gen = torch.Generator().manual_seed(777)
    state = torch.zeros(100, 25, NODE_DIM)
    velocity = torch.rand(100, 25, 2, generator=gen)
    noise = .02 * torch.randn(100, 31, 25, 2, generator=gen)
    opens = 0
    for t in range(30):
        z = velocity + noise[:, t + 1] - noise[:, t]
        state = accumulate_acceleration(state, z, torch.ones(100, 25), float(t),
                                        torch.ones(100, 25), torch.zeros(100, 25))
        opens += int((acceleration_fit(state)[1] > 0).sum())
    assert opens / (100 * 25 * 30) < .02


def test_synthetic_seed_ensemble_detection_vs_null():
    import json
    from scripts.order2_gate import synthetic_ensemble
    result = synthetic_ensemble(range(777, 797))  # 20 paired seeds, C2/K2, world32/window24
    published = json.loads((Path(__file__).resolve().parents[1] / 'results/order2_gate/synthetic.json').read_text())
    for row in result['rows']:
        reference_row = next(r for r in published['rows'] if r['accel'] == row['accel'] and r['kappa'] == row['kappa'])
        expected = torch.tensor(reference_row['gates'][:20])
        torch.testing.assert_close(torch.tensor(row['gates']), expected, atol=2e-5, rtol=0)
    # Full 50-seed gate: kappa9 null 0%, detection 94%. Allow 5% null and
    # 85% detection for this smaller 20-seed regression subset.
    null, detected = [r for r in result['rows'] if r['kappa'] == 9]
    assert null['open_rate'] <= .05
    assert detected['open_rate'] >= .85
    assert detected['open_rate'] - null['open_rate'] >= .8


def test_on_lattice_comoving_boost_order2_recurrence():
    from kwm.equiv import boost_transform_nodes, equiv_error, interior_mask
    H = 32
    nodes, boost = lattice_nodes(), torch.tensor([[1., 0.]])
    torch.manual_seed(0)
    cell = AdaptiveNodeCell(C=2, imprint_k=2, world=H, window=H, node_update='lock', order=2)
    h = cell.initial_state(torch.zeros(1, 1, H, H)); hb = cell.initial_state(torch.zeros(1, 1, H, H))
    axis = torch.arange(H).float()
    with torch.no_grad():
        for t in range(12):
            dx = (axis[None] - (12 + .1 * t + .002 * t*t) + H/2) % H - H/2
            dy = (axis[:, None] - 16 + H/2) % H - H/2
            frame = torch.exp(-(dx.square() + dy.square()) / 12)[None, None]
            h = cell(frame, h, torch.zeros(1, 2), nodes)
            hb = cell(frame, hb, boost if t else torch.zeros_like(boost), nodes)
        transformed, mask = boost_transform_nodes(h, nodes, 1., 2, boost, .75)
        mask &= interior_mask(nodes, 1.)
        assert equiv_error(transformed, hb, 2, mask).item() < 1e-6
        torch.testing.assert_close(transformed.nodes[..., :2][mask], hb.nodes[..., :2][mask], atol=1e-6, rtol=0)
        torch.testing.assert_close(acceleration_fit(transformed.nodes)[0][mask], acceleration_fit(hb.nodes)[0][mask], atol=1e-6, rtol=0)


def test_no_n_s_in_acceleration_fit(monkeypatch):
    import kwm.cells as cells
    # Exact measurements isolate the slope from the learned PR1 step scale.
    action, lattice = torch.zeros(1, 2), torch.zeros(1, 2)
    dummy = torch.zeros(1, 1, 8, 8)
    final = []
    for scale in (.5, 1.5):
        cell = AdaptiveNodeCell(C=1, imprint_k=1, world=16, window=8, num_v=1,
                                node_update='lock', order=2, s_init=scale)
        state = cell.initial_state(dummy).nodes
        for step in range(1, 12):
            z = torch.tensor([[[.1 + .004*(step-.5), 0.]]])
            def measurement(previous, current, nu, *args, **kw):
                return z - nu, torch.ones(1, 1), torch.ones(1, 1), torch.zeros(1, 1)
            monkeypatch.setattr(cells, 'lock_measurement', measurement)
            state = cell.lock_update(dummy, dummy, state, action, lattice)
        final.append(acceleration_fit(state)[0])
    torch.testing.assert_close(final[0], final[1], atol=1e-7, rtol=0)
    torch.testing.assert_close(final[0][0, 0], torch.tensor([.004, 0.]), atol=1e-7, rtol=0)


@pytest.mark.slow
def test_accel_dataset_regression_from_gate():
    import json
    from scripts.order2_gate import encoder, run, corrected_final, matches
    result = json.loads((Path(__file__).resolve().parents[1] / 'results/order2_gate/detection.json').read_text())
    batch = env.make_batch(100, 30, 0, env.EnvConfig(real_velocity=True, real_camera=True, accel=.02),
                           gen=torch.Generator().manual_seed(777), split='test')
    # Eight full-size sequences keep this file below 60s even with slow enabled.
    count = 8
    subset = dict(frames=batch['frames'][:count], actions=batch['actions'][:count])
    states, _ = run(subset, encoder(order=2))
    nu, _, _, _, _ = corrected_final(states[:, -1], kappa=4.)
    saved = result['metrics']['order2_k4']
    metrics, _ = matches(nu, batch['digit_states']['vel'][:count, -1], batch['digit_states']['visible'][:count, -10:].any(1),
                          states[:, -1, :, 2], torch.tensor(saved['assignments'][:count]))
    expected = torch.tensor(saved['errors'][:count])
    torch.testing.assert_close(torch.tensor(metrics['errors']), expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize('order', [1, 2])
def test_clamp_diagnostics_count_both_phases(monkeypatch, order):
    model = build_model(dict(model='adaptive', C=1, imprint_k=1, world=8, window=8,
                              node_update='lock', order=order, node_update_pred='off'))
    frame = torch.zeros(1, 1, 8, 8)
    initial = model.initial_state(frame)
    st = fit_linear().expand(1, 25, -1).clone()
    st[..., :2], st[..., 2], st[..., 3], st[..., TIME] = .75, 1., 0., 200.
    st[:, 1:] = 0  # empty nodes must not dilute the denominator
    initial = initial._replace(nodes=st)
    monkeypatch.setattr(model, 'initial_state', lambda observation: initial)
    _, aux = model(frame[:, None], torch.zeros(1, 1, 2), torch.zeros(1, 2, 2),
                   return_aux=True, return_states=[0])
    assert aux['clamp_rate_obs'] == 1 and aux['clamp_rate_pred'] == 1
    assert aux['clamp_node_steps_obs'] == 1 and aux['clamp_node_steps_pred'] == 2
    assert model.cell.offsets(aux['states'][0]).abs().max() <= .75


def test_clamp_includes_half_interval_acceleration():
    state = fit_linear()
    state[..., :2] = torch.tensor([.749, 0.])
    assert clamp_mask(state).all()
    assert not clamp_mask(state, order=1).any()


@pytest.mark.parametrize('update', [False, True])
def test_prediction_blends_frozen_intercept_without_double_count(monkeypatch, update):
    import kwm.cells as cells
    state = fit_linear(a=(.004, 0.), velocity=(.2, .1))
    # Deliberately disagree with the fit intercept, as a stale all-time mean does.
    state[..., :2], state[..., 2], state[..., TIME] = -.1, 30., 30.
    state[..., CORR] = fit_correction(state, state[..., :2], 29.)
    dummy, lattice = torch.zeros(1, 1, 8, 8), torch.zeros(1, 2)
    cell = AdaptiveNodeCell(C=1, imprint_k=1, num_v=1, node_update='lock', order=2)
    frozen = state[..., 4:14].clone()
    for step in range(30, 35):
        def measurement(previous, current, nu, *args, **kwargs):
            z = torch.tensor([[[.2 + .004 * (step - .5), .1]]])
            return z - nu, torch.ones(1, 1), torch.ones(1, 1)
        monkeypatch.setattr(cells, 'lock_measurement', measurement)
        state = cell.lock_update(dummy, dummy, state, torch.zeros(1, 2), lattice,
                                 update_nodes=update, prediction=True)
        assert torch.equal(state[..., 4:14], frozen)
        torch.testing.assert_close(transport_velocity(state, lattice, 1.),
                                   torch.tensor([[[.2 + .004 * (step + .5), .1]]]), atol=1e-6, rtol=0)
    if update:
        assert (state[..., :2] != -.1).all()


def test_zero_gate_selects_order1_offsets_exactly():
    state = fit_linear(a=(0., 0.))
    state[..., :2] = torch.tensor([.1234567, -.2345678])
    state[..., CORR] = fit_correction(state, state[..., :2], 29.)
    assert torch.equal(corrected_offsets(state), state[..., :2])


def test_partial_gate_uses_fit_intercept_and_shrink_scales_slope():
    state = fit_linear()
    state[..., :2] = -.1
    state[..., TIME] = 30.
    # Set residual power so the gate is fractional, not just an endpoint.
    slope = acceleration_fit(state)[0]
    power = slope.square().sum(-1)
    state[..., CZZ] += .5 * power / 4 * state[..., CTT] * (state[..., SW] - 2)
    slope, gate, _, _ = acceleration_fit(state)
    assert .45 < gate < .55
    state[..., CORR] = fit_correction(state, state[..., :2], 29.)
    expected = (1 - gate[..., None]) * state[..., :2] + gate[..., None] * (
        state[..., MZ] + slope * (29 - state[..., MT])[..., None])
    torch.testing.assert_close(corrected_offsets(state), expected, atol=1e-7, rtol=0)
    cell = AdaptiveNodeCell(C=1, imprint_k=1, num_v=1, node_update='lock', order=2, pred_shrink='learned')
    shrunk = cell.prediction_state(state, torch.zeros(1, 2))
    factor = cell.shrink_logit.sigmoid().detach()
    a, g, _, _ = acceleration_fit(shrunk)
    torch.testing.assert_close(a, slope * factor, atol=1e-8, rtol=0)
    torch.testing.assert_close(g, gate, atol=1e-6, rtol=0)


def test_gate_assignment_excludes_empty_nodes_and_proxy_uses_interval_mean():
    from scripts.order2_gate import matches, motion_metrics
    velocity = torch.tensor([[[.15, 0.], [.1, 0.]]])
    truth = torch.tensor([[[.15, 0.]]])
    evidence, visible = torch.tensor([[0., 1.]]), torch.ones(1, 1, dtype=torch.bool)
    _, assignment = matches(velocity, truth, visible, evidence)
    assert assignment.item() == 1  # the perfect-velocity node has no evidence
    metrics = motion_metrics(velocity, torch.zeros_like(velocity), truth, torch.tensor([[[.01, 0.]]]),
                              visible, evidence, assignment)
    assert metrics['displacement']['20']['mean'] == pytest.approx(2.9)
    assert metrics['displacement']['50']['mean'] == pytest.approx(14.75)


def test_boost_merge_keeps_modes_categorical():
    from kwm.equiv import boost_transform_nodes
    from kwm.lock_nodes import LockState
    nodes = lattice_nodes()
    st = torch.zeros(1, 25, NODE_DIM)
    i = int((nodes == torch.tensor([0., 0.])).all(-1).nonzero()[0])
    j = int((nodes == torch.tensor([1., 0.])).all(-1).nonzero()[0])
    st[0, i, 0], st[0, j, 0] = .7, -.3
    st[0, i, MODE], st[0, j, MODE] = 1, 2
    features = torch.zeros(1, 25, 1, 8, 8)
    features[:, i], features[:, j] = 1, 2
    out, _ = boost_transform_nodes(LockState(features, st, torch.zeros(1, 1, 8, 8)),
                                    nodes, 1., 1, torch.tensor([[.2, 0.]]), .75)
    assert out.nodes[0, j, MODE] == 2
    assert torch.equal(out.nodes[..., MODE], out.nodes[..., MODE].round())
