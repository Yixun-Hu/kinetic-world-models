"""The cheap gate must retain the public evaluation dynamics and cache identity."""
import json

import pytest
import torch

from kwm.diagnostics import effective_count_at
from kwm.eval import configure_prediction
from kwm.lock_nodes import features
from kwm.model import KineticWM
from scripts.pred_input_gate import (FrozenTransport, Measurements, frozen, observation_pass,
                                     recurrent, reusable_baseline)


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
