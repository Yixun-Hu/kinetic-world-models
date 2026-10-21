import math
import torch
from kwm.transport import spectral_shift, stream, phase_correlation


def blob(H, W, cx, cy, s=2.5):
    y = torch.arange(H).float()[:, None]
    x = torch.arange(W).float()[None, :]
    dx = (x - cx + W / 2) % W - W / 2
    dy = (y - cy + H / 2) % H - H / 2
    return torch.exp(-(dx ** 2 + dy ** 2) / (2 * s ** 2))


def test_integer_shift_equals_roll():
    torch.manual_seed(0)
    h = torch.randn(3, 4, 5, 10, 13)                      # [B,V,C,H,W], odd and even sizes
    s = torch.randint(-7, 8, (3, 4, 2)).float()           # per (B,V)
    out = spectral_shift(h, s)
    ref = torch.stack([torch.stack([torch.roll(h[b, v], shifts=(int(s[b, v, 1]), int(s[b, v, 0])), dims=(-2, -1))
                                    for v in range(4)]) for b in range(3)])
    assert (out - ref).abs().max() < 1e-5
    sb = torch.randint(-7, 8, (3, 2)).float()             # per batch, broadcast over V, C
    out = spectral_shift(h, sb)
    ref = torch.stack([torch.roll(h[b], shifts=(int(sb[b, 1]), int(sb[b, 0])), dims=(-2, -1)) for b in range(3)])
    assert (out - ref).abs().max() < 1e-5


def test_phase_composition_no_accumulation():
    H = W = 50
    v = torch.tensor([0.7, -1.3])
    h = blob(H, W, 10.0, 20.0)
    T = 150
    x = h.clone()
    for _ in range(T):
        x = spectral_shift(x, v)
    once = spectral_shift(h, v * T)
    gt = blob(H, W, 10.0 + 0.7 * T, 20.0 - 1.3 * T)
    assert (x - once).abs().max() < 1e-4
    assert (x - gt).abs().max() < 1e-4


def test_stream_matches_manual():
    torch.manual_seed(1)
    h = torch.randn(2, 3, 4, 16, 16)
    nodes = torch.tensor([[0.5, 0.0], [-1.25, 0.75], [2.0, -2.0]])
    a = torch.tensor([[3.3, -1.1], [-0.4, 6.0]])
    dt = 0.5
    out = stream(h, nodes, a, dt)
    for b in range(2):
        for i in range(3):
            ref = spectral_shift(h[b, i], nodes[i] * dt - a[b])
            assert (out[b, i] - ref).abs().max() < 1e-5


def test_shift_differentiable_wrt_shift():
    h = blob(32, 32, 10.0, 12.0)
    s = torch.tensor([0.3, -0.2], requires_grad=True)
    y = spectral_shift(h, s)
    target = blob(32, 32, 10.8, 12.5)
    loss = ((y - target) ** 2).sum()
    loss.backward()
    assert s.grad is not None and s.grad.abs().sum() > 0
    assert s.grad[0] < 0 and s.grad[1] < 0       # gradient points towards increasing the shift


def test_phase_correlation_subpixel():
    g = blob(50, 50, 20, 25, 2.0) + 0.5 * blob(50, 50, 30, 15, 3.0)
    shifts = torch.tensor([[0.3, -1.7], [5.5, 2.25], [-7.8, 0.1], [12.0, -9.0]])
    est = phase_correlation(g.expand(4, 50, 50), spectral_shift(g.expand(4, 50, 50), shifts))
    assert (est - shifts).abs().max() < 1e-2
