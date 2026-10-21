"""Per-sample node velocities and the oracle-node diagnostic."""
import torch
import pytest

from kwm.model import build_model, oracle_nodes_from
from kwm.transport import stream


def test_stream_per_sample_nodes():
    torch.manual_seed(2)
    h = torch.randn(3, 6, 2, 16, 16)
    nodes = torch.randn(3, 6, 2)
    a = torch.randn(3, 2)
    out = stream(h, nodes, a, 0.5)
    for b in range(3):
        ref = stream(h[b:b + 1], nodes[b], a[b:b + 1], 0.5)
        assert (out[b] - ref[0]).abs().max() < 1e-5


def test_oracle_nodes_model():
    torch.manual_seed(3)
    m = build_model(dict(model="oracle", C=8, world=20, window=12)).train()
    assert m.num_v == 6
    vel0 = torch.rand(2, 5, 2) * 4 - 2
    nodes = oracle_nodes_from(vel0)
    assert nodes.shape == (2, 6, 2) and (nodes[:, 0] == 0).all()
    obs = torch.rand(2, 4, 1, 12, 12)
    p, aux = m(obs, torch.randn(2, 4, 2), torch.randn(2, 3, 2), nodes=nodes, return_aux=True)
    assert p.shape == (2, 3, 1, 12, 12) and aux["mean_velocity"].shape == (2, 3, 2, 12, 12)
    ((p - 0.3) ** 2).mean().backward()
    with pytest.raises(ValueError):
        m(obs, torch.randn(2, 4, 2), torch.randn(2, 3, 2))
