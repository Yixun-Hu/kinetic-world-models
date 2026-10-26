"""The cheap gate must retain the public evaluation dynamics and cache identity."""
import json

import pytest
import torch

from kwm.diagnostics import effective_count_at
from kwm.eval import configure_prediction
from kwm.lock_nodes import features
from kwm.model import KineticWM
from scripts.pred_input_gate import (FrozenTransport, Measurements, frozen, observation_pass,
                                     recurrent, reusable_baseline, zero_then_frozen)


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("rule", ["lk", "lock"])
def test_shared_observation_and_cached_transport_match_eval(rule):
    torch.manual_seed(810)
    m = KineticWM("adaptive", C=2, world=12, window=8, imprint_k=2, node_update=rule).eval()
    obs = torch.rand(1, 5, 1, 8, 8)
    ain, aout = torch.randn(1, 5, 2), torch.randn(1, 3, 2)
    target = torch.rand(1, 3, 1, 8, 8)
    truth = dict(cent=torch.full((1, 3, 1, 2), 3.), half=torch.ones(1, 1, 2),
                 cam=torch.zeros(1, 3, 2), vel=torch.zeros(1, 3, 1, 2),
                 visible=torch.ones(1, 3, 1, dtype=torch.bool))
    with torch.no_grad():
        h, pre, anchors = observation_pass(m, obs, ain, [0, 4])
        for mode, k in [("flowm", 0), ("zero", 0), ("frozen", 0), ("frozen_prelift", 0), ("frozen_prelift", 4)]:
            configure_prediction(m, {}, mode, k)
            expected, aux = m(obs, ain, aout, digit_truth=truth)
            metric = Measurements(m, truth, target, effective_count_at(m, h))
            if mode in ("flowm", "zero"):
                recurrent(m, h, obs[:, -1], aout, metric, mode, 3)
            else:
                anchor = h
                if mode == "frozen_prelift":
                    anchor = m.with_features(h, pre) if k == 0 else anchors[k]
                    for t in range(5 - k, 5):
                        anchor, _ = m._step(torch.zeros_like(obs[:, -1]), anchor, ain[:, t], 1., None, update_nodes=False)
                transport = FrozenTransport(m, anchor, mode == "frozen_prelift")
                for elapsed in (1., 37.5, 150.):
                    actual = transport(aout.sum(1), elapsed)
                    reference = m.transport_frozen(anchor, aout.sum(1), elapsed, mode == "frozen_prelift")
                    assert torch.equal(features(actual)[:, :, :m.C], features(reference)[:, :, :m.C])
                    assert torch.equal(m.node_velocity(actual), m.node_velocity(reference))
                    assert torch.equal(m.decode(actual), m.decode(reference))
                frozen(m, anchor, aout, metric, activate=mode == "frozen_prelift")
            expected_mse = (expected.clamp(0, 1) - target).square().mean((2, 3, 4))
            assert torch.equal(torch.stack(metric.mse, 1), expected_mse)
            assert torch.equal(torch.stack(metric.counts, 1), aux["effective_node_count"])


def test_baseline_cache_rejects_different_checkpoint_or_sequences(tmp_path):
    common = dict(checkpoint_sha256="weights-a", cache_sha256="cache-a", n=16,
                  t_obs=30, t_pred=150, seed=12345, threads=4,
                  provenance=dict(files={"kwm/model.py": "source-a"}))
    row = dict(common, dec_input="zero", node_update_pred=True, effective_node_count={"0": 23.})
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(row))
    assert reusable_baseline(path, common, "zero") == row
    assert reusable_baseline(path, common, "flowm") is None
    for key, value in [("checkpoint_sha256", "weights-b"), ("cache_sha256", "cache-b"),
                       ("n", 32), ("threads", 1), ("provenance", dict(files={"kwm/model.py": "source-b"})),
                       ("provenance", dict(files=common["provenance"]["files"], software={"torch": "different"}))]:
        assert reusable_baseline(path, dict(common, **{key: value}), "zero") is None


@pytest.mark.parametrize("rule", ["lk", "lock"])
@pytest.mark.parametrize("update", [False, True])
def test_gate_hybrid_matches_legacy_variant_and_eval(rule, update):
    torch.manual_seed(812)
    m = KineticWM("adaptive", C=2, world=12, window=8, imprint_k=2,
                  node_update=rule, node_update_pred=update, readout_rel=True).eval()
    frames = torch.rand(1, 8, 1, 8, 8)
    actions = torch.randn(1, 8, 2)
    truth = dict(cent=torch.full((1, 3, 1, 2), 3.), half=torch.ones(1, 1, 2),
                 cam=torch.randn(1, 3, 2), vel=torch.randn(1, 3, 1, 2),
                 visible=torch.ones(1, 3, 1, dtype=torch.bool))
    with torch.no_grad():
        h, _, _ = observation_pass(m, frames[:, :5], actions[:, :5], [])
        old = Measurements(m, truth, frames[:, 5:], effective_count_at(m, h))
        predictions = []
        add = old.add
        def record(t, state, pred):
            predictions.append(pred)
            add(t, state, pred)
        old.add = record
        m.cell.update_nodes = update
        anchor, _ = recurrent(m, h, frames[:, 4], actions[:, 5:], old, "zero", 1)
        frozen(m, anchor, actions[:, 5:], old, start=1)
        m.cell.update_nodes = True
        pred, aux = zero_then_frozen(m, {}, frames, actions, torch.zeros(1, 2), 5, 1, truth)
        torch.testing.assert_close(pred, torch.stack(predictions, 1), rtol=0, atol=1e-6)
        direct, direct_aux = m(frames[:, :5], actions[:, :5], actions[:, 5:], digit_truth=truth)
        assert torch.equal(pred, direct)
        assert all(torch.equal(aux[k], direct_aux[k]) for k in aux)
        actual = Measurements(m, truth, frames[:, 5:], effective_count_at(m, h))
        actual.record_rollout(pred, aux)
        torch.testing.assert_close(torch.stack(actual.counts), torch.stack(old.counts), rtol=0, atol=1e-6)
        torch.testing.assert_close(torch.stack(actual.mse), torch.stack(old.mse), rtol=0, atol=1e-6)
        assert actual.concentration == pytest.approx(old.concentration, abs=1e-6)


@pytest.mark.parametrize("threads", [1, 4])
def test_cached_zero_prefix_keeps_full_horizon_reduction_order(threads):
    torch.set_num_threads(threads)
    g = torch.Generator().manual_seed(91)
    metric = Measurements(None, None, torch.zeros(16, 150, 1, 1, 1), torch.ones(16))
    metric.mse = list(torch.rand(16, 150, generator=g).unbind(1))
    metric.counts.extend([torch.ones(16)] * 150)
    full, prefix = metric.result(), metric.prefix(20).result()
    assert prefix["mse20"] == full["mse20"]
    assert prefix["per_seq_mse20"] == full["per_seq_mse20"]
    assert prefix["mse_curve"] == full["mse_curve"][:20]
