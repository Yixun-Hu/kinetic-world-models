"""Fast CPU regressions for prediction policies, transport, and ghosting metrics."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from kwm.diagnostics import (node_concentration, concentration_summary, effective_node_count,
                             effective_count_summary)
from kwm import eval as evaluate
from kwm.equiv import boost_transform, equiv_error, interior_mask, target_mask
from kwm.eval import configure_prediction, load_model, prediction_metadata, rollout
from kwm.lock_nodes import LockState, features
from kwm.model import KineticWM, build_model
from kwm.train import parse_args
from kwm.transport import stream, spectral_shift


KINDS = [("flowm", "lk"), ("spectral", "lk"), ("adaptive", "lk"), ("adaptive", "lock")]


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def model_for(kind, rule, **kwargs):
    torch.manual_seed(71)
    mode = kwargs.pop("dec_input", "flowm")
    m = KineticWM(kind, C=2, world=12, window=8, imprint_k=2, node_update=rule, **kwargs)
    configure_prediction(m, {}, mode, m.frozen_k)
    return m


def inputs():
    g = torch.Generator().manual_seed(72)
    return (torch.rand(1, 5, 1, 8, 8, generator=g),
            torch.randn(1, 5, 2, generator=g), torch.randn(1, 3, 2, generator=g))


def old_forward(model, obs, ain, aout):
    """Loop-level reference sharing current step plumbing; golden test covers base code."""
    def original_update(u, hs, return_prelift=False):
        assert not return_prelift
        return model.cell.act(model.cell.lift(u, model.num_v) + model.cell.mix(hs))
    model.cell.feature_update = original_update
    h = model.initial_state(obs[:, 0])
    for t in range(obs.shape[1]):
        h, _ = model.step(obs[:, t], h, ain[:, t], 1., None)
    prev, out = obs[:, -1], []
    if model.kind == "adaptive":
        model.cell.update_nodes = model.node_update_pred
    for t in range(aout.shape[1]):
        frame = torch.zeros_like(prev) if model.dec_input == "zero" else prev
        h, _ = model.step(frame, h, aout[:, t], 1., None)
        prev = model.decode(h)
        out.append(prev)
    return torch.stack(out, 1)


@pytest.mark.parametrize("kind,rule", KINDS)
@pytest.mark.parametrize("mode", ["flowm", "zero"])
def test_default_eval_bit_identical(kind, rule, mode):
    m = model_for(kind, rule, dec_input=mode).eval()
    ref = copy.deepcopy(m)
    obs, ain, aout = inputs()
    with torch.no_grad():
        expected = old_forward(ref, obs, ain, aout)
        assert configure_prediction(m, {"dec_input": mode}) == ""
        fr = torch.cat([obs, obs[:, :3]], 1)
        an = torch.cat([ain, aout], 1)
        actual, _ = rollout(m, fr, an, torch.zeros(1, 2), 5, 3, 1., 3, False, "cpu", aux=True)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("kind,rule", KINDS)
@pytest.mark.parametrize("mode", ["flowm", "zero"])
def test_pre_pr3_checkpoint_no_override_matches_base_golden(kind, rule, mode, tmp_path):
    # These checkpoint bytes and predictions were produced by git base 0c1198a,
    # including its original cells, _step, build_model, load_model and rollout.
    # Regenerate with tests/reference/generate_pred_input_golden.py.
    path = Path(__file__).parent / "reference/pre_pr3_predictions.npz"
    with np.load(path, allow_pickle=False) as golden:
        key = f"{kind}_{rule}_{mode}"
        assert str(golden["base"]).startswith("0c1198a")
        (tmp_path / "last.pt").write_bytes(golden[key + "_checkpoint"].tobytes())
        model, saved, step = load_model(tmp_path, "last", "cpu")
        assert "frozen_k" not in saved and model.frozen_k == 0 and step == 17
        assert configure_prediction(model, saved) == ""
        with torch.no_grad():
            actual, _ = rollout(model, torch.from_numpy(golden["frames"]),
                                torch.from_numpy(golden["actions"]), torch.zeros(1, 2),
                                5, 3, 1., 3, False, "cpu", aux=True)
        assert torch.equal(actual, torch.from_numpy(golden[key + "_predictions"]))


@pytest.mark.parametrize("kind,rule", KINDS)
def test_zero_override_matches_saved_zero(kind, rule, tmp_path):
    m = model_for(kind, rule).eval()
    args = dict(model=kind, C=2, world=12, window=8, imprint_k=2, node_update=rule, dec_input="flowm")
    torch.save(dict(args=args, model=m.state_dict(), step=1), tmp_path / "last.pt")
    override, saved, _ = load_model(tmp_path, "last", "cpu", dec_input="zero")
    trained = build_model(dict(args, dec_input="zero")).eval()
    trained.load_state_dict(m.state_dict())
    assert saved["dec_input"] == "flowm"
    with torch.no_grad():
        assert torch.equal(override(*inputs()), trained(*inputs()))


def identity_mix(m):
    with torch.no_grad():
        m.cell.conv_h.weight.zero_()
        for c in range(m.C):
            m.cell.conv_h.weight[c, c, 1, 1] = 1
        m.cell.conv_u.weight.fill_(.05)


@pytest.mark.parametrize("kind,rule", KINDS)
def test_frozen_first_step_matches_zero_identity_mix(kind, rule):
    m = model_for(kind, rule, readout="max").eval()
    identity_mix(m)
    obs, ain, aout = inputs()
    # On-lattice shifts commute with ReLU; FloWM observation/action rounding is exact here.
    ain, aout = ain.round(), aout[:, :1].round()
    m.node_update_pred = False
    if kind == "adaptive":
        m.cell.n_s.data.zero_()
    with torch.no_grad():
        m.dec_input = "zero"
        zero = m(obs, ain, aout)
        m.dec_input = "frozen"
        frozen = m(obs, ain, aout)
    assert (frozen - zero).abs().max() <= 1e-5


@pytest.mark.parametrize("kind,rule", KINDS)
def test_prelift_k0_removes_only_last_lift(kind, rule, monkeypatch):
    m = model_for(kind, rule, readout="max").eval()
    identity_mix(m)
    obs, ain, aout = inputs()
    ain.zero_(); aout.zero_()
    if kind == "adaptive":
        m.cell.n_s.data.zero_()
    captured = []
    transport = m.transport_frozen
    def capture(h, a, elapsed, activate=False):
        captured.append(h)
        return transport(h, a, elapsed, activate)
    monkeypatch.setattr(m, "transport_frozen", capture)
    with torch.no_grad():
        m.dec_input = "frozen"
        m(obs, ain, aout[:, :1])
        m.dec_input, m.frozen_k = "frozen_prelift", 0
        m(obs, ain, aout[:, :1])
        post, pre = [features(h)[:, :, :m.C] for h in captured]
        lift = m.cell.lift(obs[:, -1], m.num_v)
    assert (post - pre - lift).abs().max() <= 1e-5
    assert lift.abs().max() > .01
    if kind == "adaptive":
        assert torch.equal(m.cell.offsets(captured[0]), m.cell.offsets(captured[1]))


@pytest.mark.parametrize("kind,rule", KINDS)
@pytest.mark.parametrize("k", [0, 1, 2, 4])
def test_frozen_observations_unchanged_and_prediction_updates_ignored(kind, rule, k):
    m = model_for(kind, rule).eval()
    obs, ain, aout = inputs()
    with torch.no_grad():
        _, baseline = m(obs, ain, aout, return_states=range(5))
        m.dec_input, m.frozen_k = "frozen_prelift", k
        p, aux = m(obs, ain, aout, return_states=range(5), record_nodes=True)
        m.node_update_pred = False
        q = m(obs, ain, aout)
    assert torch.equal(p, q)
    for t in range(5):
        assert torch.equal(features(baseline["states"][t]), features(aux["states"][t]))
    if kind == "adaptive":
        nodes = aux["node_traj"]
        assert nodes.shape[1] == 8
        cutoff = 4 if k == 0 else 4 - k
        # LK packs constant offsets into spatial planes; offsets() averages them
        # back, with fp32 reduction roundoff. Prediction steps themselves are fixed.
        torch.testing.assert_close(nodes[:, 5:], nodes[:, cutoff:cutoff + 1].expand(-1, 3, -1, -1),
                                   rtol=0, atol=5e-7)
        assert torch.equal(nodes[:, 5:], nodes[:, 5:6].expand(-1, 3, -1, -1))
    assert prediction_metadata(m)["node_update_pred_ignored"]
    assert prediction_metadata(m)["node_update_pred"] is False


@pytest.mark.parametrize("rule", ["lk", "lock"])
def test_frozen_uses_carried_velocity_and_single_real_shift(rule):
    m = model_for("adaptive", rule)
    h = m.initial_state(torch.zeros(1, 1, 8, 8))
    feat = torch.rand(1, m.num_v, m.C, 12, 12)
    if isinstance(h, LockState):
        h.nodes[..., :2] = torch.tensor([.37, -.29])
    else:
        h[:, :, m.C:m.C + 2] = torch.tensor([.37, -.29])[None, None, :, None, None]
    h = m.with_features(h, feat)
    cumulative_action = torch.tensor([[.13, -.79]])
    expected = stream(feat, m.nodes()[None] + m.cell.offsets(h), cumulative_action, 2.5)
    actual = features(m.transport_frozen(h, cumulative_action, 2.5))[:, :, :m.C]
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("kind,rule", KINDS)
@pytest.mark.parametrize("mode,k", [("frozen", 1), ("frozen_prelift", 0), ("frozen_prelift", 2)])
def test_frozen_comoving_boost_predictions(kind, rule, mode, k, monkeypatch):
    m = model_for(kind, rule, dec_input=mode, frozen_k=k, readout="max").eval()
    # Comoving boost: viewport content stays identical; only camera actions move.
    # Fixed carried offsets isolate analytic-transport equivariance from estimator errors.
    if kind == "adaptive":
        m.cell.n_s.data.zero_()
    obs, ain, aout = inputs()
    ain, aout = ain.round(), aout.round()
    boost = torch.tensor([[1., 0.]])
    nodes = m.nodes()
    target = target_mask(nodes, 1., boost) & interior_mask(nodes, 1.)[None]
    source = ((nodes[:, None] + boost[:, None] - nodes[None, :]).abs().sum(-1) < 1e-5)
    source = (source & target).any(-1)[None]
    captured = []
    decode = m.decode
    def masked_decode(h, aux=None):
        captured.append(features(h)[:, :, :m.C].clone())
        mask = source if len(captured) <= 3 else target
        return decode(m.with_features(h, features(h)[:, :, :m.C] * mask[:, :, None, None, None]), aux)
    monkeypatch.setattr(m, "decode", masked_decode)
    boosted_ain = ain.clone()
    boosted_ain[:, 1:] += boost[:, None]
    with torch.no_grad():
        p = m(obs, ain, aout)
        q = m(obs, boosted_ain, aout + boost[:, None])
    assert (p - q).abs().max() < 1e-5  # co-moving viewport is invariant
    for t in range(3):
        transformed = boost_transform(captured[t], nodes, 1., m.C, boost, 0)
        assert equiv_error(transformed, captured[t + 3], m.C, target).max() < 1e-5
        # In world coordinates the same predictions are displaced by boost * time.
        camera = ain.sum(1) + aout[:, :t + 1].sum(1)
        time = obs.shape[1] + t
        world_p = spectral_shift(spectral_shift(p[:, t], camera), boost * time)
        world_q = spectral_shift(q[:, t], camera + boost * time)
        assert (world_p - world_q).abs().max() < 1e-5


@pytest.mark.parametrize("mode", ["frozen", "frozen_prelift"])
def test_frozen_modes_are_eval_only(mode):
    with pytest.raises(SystemExit):
        parse_args(["--dec_input", mode])
    with pytest.raises(ValueError, match="eval overrides"):
        build_model(dict(model="spectral", dec_input=mode))
    m = model_for("spectral", "lk", dec_input=mode).train()
    with pytest.raises(ValueError, match="evaluation-only"):
        m(*inputs())


def test_training_args_have_no_frozen_knob():
    assert parse_args([]).dec_input == "flowm"
    assert parse_args(["--dec_input", "zero"]).dec_input == "zero"
    assert not hasattr(parse_args([]), "frozen_k")
    with pytest.raises(SystemExit):
        parse_args(["--frozen_k", "0"])


def test_effective_nodes_entropy_before_spatial_averaging():
    feat = torch.ones(1, 4, 2, 2, 2)
    assert effective_node_count(feat).item() == pytest.approx(4.)
    feat.zero_()
    # Each pixel belongs to a different node: per-pixel count is 1, whereas
    # entropy after spatial pooling would incorrectly report 4.
    for node in range(4):
        feat[:, node, :, node // 2, node % 2] = 1
    assert effective_node_count(feat).item() == pytest.approx(1.)
    assert effective_node_count(feat * 0).item() == 0
    values = torch.tensor([[4., 1., 3.], [2., 3., 1.]])
    assert effective_count_summary(values) == {"0": 3., "1": 2., "mean_prediction": 2.}


def test_concentration_oracle_and_edge_cases():
    m = KineticWM("oracle", C=1, world=8, window=8, n_digits=1, readout="max")
    nodes = torch.tensor([[[0., 0.], [1., .25]]])
    feat = torch.zeros(1, 2, 1, 8, 8)
    feat[:, 1, :, 2:5, 2:5] = 1
    masks = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    masks[:, :, 2:5, 2:5] = True
    v, visible = nodes[:, 1:], torch.ones(1, 1, dtype=torch.bool)
    assert m.num_v == feat.shape[1]
    score = node_concentration(feat, nodes, v, masks, visible)
    assert score.item() == pytest.approx(1.)
    feat[:, 0] = feat[:, 1]
    assert node_concentration(feat, nodes, v, masks, visible).item() == pytest.approx(.5)
    assert node_concentration(feat * 0, nodes, v, masks, visible).item() == 0
    assert node_concentration(feat, nodes, v, masks, ~visible).isnan().all()
    assert concentration_summary(score[:, None].expand(-1, 20, -1)) == {"1": 1., "10": 1., "20": 1.}


def test_oracle_one_digit_rollout_concentration(monkeypatch):
    m = KineticWM("oracle", C=1, world=8, window=8, n_digits=1, readout="max").eval()
    configure_prediction(m, {}, "frozen")
    nodes = torch.tensor([[[0., 0.], [1., 0.]]])
    h = torch.zeros(1, 2, 1, 8, 8)
    h[:, 1, :, 2:5, 2:5] = 1
    identity_mix(m)
    m.cell.conv_u.weight.data.zero_()
    monkeypatch.setattr(m, "initial_state", lambda frame: h)
    action = torch.tensor([[[1., 0.]]])
    truth = dict(cent=torch.tensor([[[[5., 3.]]]]), half=torch.ones(1, 1, 2),
                 cam=torch.tensor([[[2., 0.]]]), vel=nodes[:, 1:, None],
                 visible=torch.ones(1, 1, 1, dtype=torch.bool))
    with torch.no_grad():
        _, aux = m(torch.zeros(1, 1, 1, 8, 8), action, action, nodes=nodes, digit_truth=truth)
    assert aux["node_concentration"].item() == pytest.approx(1., abs=1e-6)


@pytest.mark.parametrize("kind,rule", KINDS)
def test_default_training_gradients_bit_identical(kind, rule):
    m = model_for(kind, rule).train()
    ref = copy.deepcopy(m)
    expected = old_forward(ref, *inputs())
    actual = m(*inputs())
    assert torch.equal(actual, expected)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for p, q in zip(m.parameters(), ref.parameters()):
        assert (p.grad is None and q.grad is None) or torch.equal(p.grad, q.grad)


@pytest.mark.parametrize("mode", ["flowm", "zero", "frozen", "frozen_prelift"])
def test_eval_aux_concentration_all_modes(mode):
    m = model_for("spectral", "lk", dec_input=mode).eval()
    obs, ain, aout = inputs()
    truth = dict(cent=torch.full((1, 3, 1, 2), 3.), half=torch.ones(1, 1, 2),
                 cam=torch.zeros(1, 3, 2), vel=torch.zeros(1, 3, 1, 2),
                 visible=torch.ones(1, 3, 1, dtype=torch.bool))
    with torch.no_grad():
        expected = m(obs, ain, aout)
        actual, aux = m(obs, ain, aout, digit_truth=truth)
    assert torch.equal(actual, expected)
    values = aux["node_concentration"]
    assert values.shape == (1, 3, 1) and ((values >= 0) & (values <= 1)).all()
    counts = aux["effective_node_count"]
    assert counts.shape == (1, 4) and ((counts >= 0) & (counts <= m.num_v + 1e-5)).all()


@pytest.mark.parametrize("saved", ["flowm", "zero"])
def test_filename_suffix(saved):
    m = model_for("spectral", "lk").eval()
    assert configure_prediction(m, {"dec_input": saved}) == ""
    assert configure_prediction(m, {"dec_input": saved}, saved) == ""
    for mode in ("flowm", "zero", "frozen", "frozen_prelift"):
        assert configure_prediction(m, {"dec_input": saved}, mode) == ("" if mode == saved else f"_dec{mode}")
    assert configure_prediction(m, {"dec_input": "flowm"}, "frozen_prelift", 0) == "_decfrozen_prelift"
    assert m.frozen_k == 0
    assert configure_prediction(m, {"dec_input": "flowm", "frozen_k": 1}, "frozen_prelift") == "_decfrozen_prelift"
    assert m.frozen_k == 0  # transient PR3 training arguments cannot silently select K>0
    assert configure_prediction(m, {"dec_input": "flowm"}, "frozen_prelift", 4) == "_decfrozen_prelift_fk4"
    with pytest.raises(ValueError, match="nonnegative"):
        configure_prediction(m, {}, "frozen_prelift", -1)
    m.frozen_k = 5
    with pytest.raises(ValueError, match="observation length"):
        m(*inputs())


def test_eval_cli_metadata_and_actual_filenames(tmp_path, monkeypatch, capsys):
    args = vars(parse_args(["--model", "spectral", "--C", "2", "--world", "12", "--window", "12",
                           "--node_spacing", "4", "--t_obs", "3", "--n_val", "1", "--device", "cpu"]))
    m = build_model(args).eval()
    run = tmp_path / "run"
    run.mkdir()
    torch.save(dict(args=args, model=m.state_dict(), step=1), run / "last.pt")
    cache = dict(frames=torch.rand(1, 153, 1, 12, 12), actions_noisy=torch.zeros(1, 153, 2),
                 cent=torch.full((1, 153, 1, 2), 5.), half=torch.full((1, 1, 2), 2.),
                 cam=torch.zeros(1, 153, 2), vel=torch.zeros(1, 153, 1, 2),
                 visible=torch.ones(1, 153, 1, dtype=torch.bool),
                 bounce_events=torch.empty(0, 4, dtype=torch.long))
    monkeypatch.setattr(evaluate, "val_cache", lambda *a, **kw: cache)
    out = tmp_path / "out"
    flags = ["--run", str(run), "--ckpt", "last", "--out", str(out), "--device", "cpu",
             "--skip_ft", "--skip_tp"]
    evaluate.main(flags)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    assert set(before) == {"eval.json", "samples.png", "mse_curve.png"}
    evaluate.main(flags + ["--dec_input", "flowm"])
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}
    # Frozen overrides also work with the default throughput benchmark enabled:
    # inference is measured, while unsupported frozen training is skipped.
    evaluate.main([f for f in flags if f != "--skip_tp"] +
                  ["--dec_input", "frozen_prelift", "--frozen_k", "0", "--node_update_pred", "on"])
    suffix = "_decfrozen_prelift"
    assert {f"eval{suffix}.json", f"samples{suffix}.png", f"mse_curve{suffix}.png"} <= {p.name for p in out.iterdir()}
    result = json.loads((out / f"eval{suffix}.json").read_text())
    assert result["dec_input"] == "frozen_prelift" and result["frozen_k"] == 0
    assert result["node_update_pred_ignored"] and result["node_update_pred"] is False
    assert set(result["node_concentration"]) == {"1", "10", "20"}
    assert all(0 <= v <= 1 for v in result["node_concentration"].values())
    assert set(result["effective_node_count"]) == {"0", "1", "10", "20", "mean_prediction"}
    assert result["throughput"]["train_step_ms"] is None
    assert result["throughput"]["infer_ms_per_seq"] > 0
    assert result["throughput"]["training_skipped"] == "frozen modes are evaluation-only"
    capsys.readouterr()
