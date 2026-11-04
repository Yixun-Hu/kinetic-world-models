"""Three-parameter C4-symmetric convolution on scalar map channels."""
import pytest
import torch
import torch.nn.functional as F

from kwm.gconv import C4SymmetricDepthwise


def test_identity_and_parameter_count():
    layer = C4SymmetricDepthwise(256)
    x = torch.randn(1, 256, 33, 33)
    assert torch.equal(layer(x), x)
    assert sum(p.numel() for p in layer.parameters()) == 3 * 256
    assert set(dict(layer.named_parameters())) == {"weight"}


@pytest.mark.parametrize("k", [1, 2, 3])
def test_symmetric_kernel_equivariance_and_free_projection(k):
    torch.manual_seed(27)
    free = torch.randn(7, 3, 3) * .2
    layer = C4SymmetricDepthwise.from_free_kernel(free)
    projected = torch.stack([torch.rot90(free, r, (-2, -1)) for r in range(4)]).mean(0)
    torch.testing.assert_close(layer.kernel()[:, 0], projected, rtol=0, atol=3e-8)
    assert torch.equal(layer.kernel(), torch.rot90(layer.kernel(), 1, (-2, -1)))
    x = torch.randn(2, 7, 33, 33)
    expected = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="circular"), projected[:, None], groups=7)
    torch.testing.assert_close(layer(x), expected, rtol=0, atol=1e-6)
    torch.testing.assert_close(layer(torch.rot90(x, k, (-2, -1))), torch.rot90(layer(x), k, (-2, -1)),
                               rtol=0, atol=1e-6)
    layer(x).square().mean().backward()
    assert torch.isfinite(layer.weight.grad).all() and (layer.weight.grad != 0).all()
