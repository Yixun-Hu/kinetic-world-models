"""Diagnostic aggregation, exact ulp inputs, budget and backend restoration."""
import pytest
import torch

from kwm.eval import rotation_check
from kwm.model import oracle_nodes_from


def _rotate_vector(v, k):
    for _ in range(k):
        v = torch.stack((-v[..., 1], v[..., 0]), -1)
    return v


class DiagnosticModel:
    collision_kind = "none"
    oracle = True
    window = 2

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    @staticmethod
    def predict(obs, ain, aout, nodes):
        # Deliberately anisotropic, sequence- and time-dependent outputs.
        ramp = obs.new_tensor([[1., 2.], [3., 4.]])
        shift = aout[..., 0] + ain[..., 1].sum(1)[:, None] + nodes[..., 0].sum(1)[:, None]
        return obs[:, -1:] * ramp + shift[:, :, None, None, None]

    def __call__(self, obs, ain, aout, nodes):
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        assert not torch.is_autocast_enabled("cpu")
        assert obs.dtype == ain.dtype == aout.dtype == torch.float32
        self.calls.append((obs.clone(), ain.clone(), aout.clone(), nodes.clone()))
        if self.fail and len(self.calls) == 2:
            raise RuntimeError("injected failure")
        return self.predict(obs, ain, aout, nodes)


def _cache(n=35, horizon=23):
    g = torch.Generator().manual_seed(98)
    return dict(frames=torch.rand(n, 2 + horizon, 1, 2, 2, generator=g),
                actions_noisy=torch.randn(n, 2 + horizon, 2, generator=g),
                vel=torch.randn(n, 2 + horizon, 2, 2, generator=g))


def test_rotation_check_aggregation_inputs_and_budget():
    cache = _cache()
    model = DiagnosticModel()
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = rotation_check(model, cache, 2, "cpu", n=35, bs=7)
            assert torch.is_autocast_enabled("cpu")
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old
    assert len(model.calls) == 25  # five forwards per batch, including the partial batch
    assert sum(len(c[0]) for c in model.calls[::5]) == 32
    assert {len(c[0]) for c in model.calls} == {7, 4}
    for i in range(0, len(model.calls), 5):
        obs, ain, aout, nodes = model.calls[i]
        changed, cai, cao, cn = model.calls[i + 1]
        assert torch.equal(changed, torch.nextafter(obs, torch.full_like(obs, float("inf"))))
        assert torch.equal(cai, ain) and torch.equal(cao, aout) and torch.equal(cn, nodes)
        for k in (1, 2, 3):
            ro, ri, ra, rn = model.calls[i + 1 + k]
            assert torch.equal(ro, torch.rot90(obs, -k, (-2, -1)))
            assert torch.equal(ri, _rotate_vector(ain, k))
            assert torch.equal(ra, _rotate_vector(aout, k))
            assert torch.equal(rn, _rotate_vector(nodes, k))

    obs = cache["frames"][:32, :2]
    ain, aout = cache["actions_noisy"][:32, :2], cache["actions_noisy"][:32, 2:]
    nodes = oracle_nodes_from(cache["vel"][:32, 0])
    pred = model.predict(obs, ain, aout, nodes)
    pairs = [(result["ulp_baseline"], pred,
              model.predict(torch.nextafter(obs, torch.full_like(obs, float("inf"))), ain, aout, nodes))]
    for k in (1, 2, 3):
        pairs.append((result["rot_equiv_error"][k], torch.rot90(pred, -k, (-2, -1)),
                      model.predict(torch.rot90(obs, -k, (-2, -1)), _rotate_vector(ain, k),
                                    _rotate_vector(aout, k), _rotate_vector(nodes, k))))
    for stats, expected, actual in pairs:
        assert set(stats) == {1, 4, 20, 23}
        for h in stats:
            error = ((expected[:, :h] - actual[:, :h]).flatten(1).norm(dim=1)
                     / expected[:, :h].flatten(1).norm(dim=1))
            assert stats[h]["mean"] == pytest.approx(error.mean().item())
            assert stats[h]["max"] == pytest.approx(error.max().item())


@pytest.mark.parametrize("matmul,cudnn", [(False, True), (True, False)])
def test_rotation_check_restores_tf32_after_error(matmul, cudnn):
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn
        with pytest.raises(RuntimeError, match="injected failure"):
            rotation_check(DiagnosticModel(fail=True), _cache(1), 2, "cpu")
        assert torch.backends.cuda.matmul.allow_tf32 == matmul
        assert torch.backends.cudnn.allow_tf32 == cudnn
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old
