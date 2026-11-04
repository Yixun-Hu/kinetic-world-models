"""Weight-tied p4 convolutions, with channels ordered field * 4 + orientation."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class C4SymmetricDepthwise(nn.Module):
    """Circular scalar-field convolution with centre/edge/corner weights.

    The three parameters per channel expand to a C4-invariant 3x3 kernel;
    channels are independent scalar fields, not orientation channels.
    """
    def __init__(self, channels, init="identity"):
        super().__init__()
        if channels <= 0 or init != "identity":
            raise ValueError("channels must be positive and init must be 'identity'")
        self.channels = channels
        self.weight = nn.Parameter(torch.zeros(channels, 3))
        with torch.no_grad():
            self.weight[:, 0] = 1

    def kernel(self):
        center, edge, corner = self.weight.unbind(1)
        return torch.stack((corner, edge, corner, edge, center, edge, corner, edge, corner), -1).reshape(-1, 1, 3, 3)

    def forward(self, x):
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="circular"), self.kernel(), groups=self.channels)

    @classmethod
    def from_free_kernel(cls, weight):
        """Project [C,3,3] onto C4 invariants by averaging each spatial orbit."""
        if weight.ndim != 3 or weight.shape[-2:] != (3, 3):
            raise ValueError("weight must have shape [C,3,3]")
        layer = cls(weight.shape[0]).to(device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            layer.weight[:, 0] = weight[:, 1, 1]
            layer.weight[:, 1] = weight[:, (0, 1, 2, 1), (1, 2, 1, 0)].mean(-1)
            layer.weight[:, 2] = weight[:, (0, 0, 2, 2), (0, 2, 2, 0)].mean(-1)
        return layer


class _GConv(nn.Module):
    def convolve(self, x, weight):
        pad = weight.shape[-1] // 2
        if pad:
            x = F.pad(x, (pad,) * 4, mode="circular")
        return F.conv2d(x, weight)


class GConvLift(_GConv):
    def __init__(self, in_ch, F, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(F, in_ch, k, k))
        nn.init.uniform_(self.weight, -1 / math.sqrt(in_ch * k * k), 1 / math.sqrt(in_ch * k * k))

    def forward(self, x):
        w = torch.stack([torch.rot90(self.weight, -r, (-2, -1)) for r in range(4)], 1)
        return self.convolve(x, w.flatten(0, 1))


class GConvRegular(_GConv):
    def __init__(self, F_in, F_out, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(F_out, F_in, 4, k, k))
        bound = 1 / math.sqrt(F_in * 4 * k * k)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x):
        w = torch.stack([torch.rot90(self.weight.roll(r, 2), -r, (-2, -1)) for r in range(4)], 1)
        return self.convolve(x, w.flatten(0, 1).flatten(1, 2))


class GConvInvariantOut(_GConv):
    def __init__(self, F_in, out_ch, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_ch, F_in, k, k))
        bound = 1 / math.sqrt(F_in * 4 * k * k)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x):
        w = torch.stack([torch.rot90(self.weight, -r, (-2, -1)) for r in range(4)], 2)
        return self.convolve(x, w.flatten(1, 2))


class MomentProjection(nn.Module):
    """[regular mass | vector mean | invariant spread | regular max] -> regular."""
    def __init__(self, fields):
        super().__init__()
        self.C = fields * 4
        self.regular = GConvRegular(2 * fields + 1, fields, 1)
        self.scalar = GConvLift(1, fields, 1)
        # Both branches belong to one projection: initialize using its total
        # fan-in, including the four vector projections and invariant spread.
        with torch.no_grad():
            self.regular.weight.mul_(math.sqrt((8 * fields + 4) / (8 * fields + 5)))
            self.scalar.weight.div_(math.sqrt(8 * fields + 5))

    def forward(self, x):
        mass, mean, spread, mx = x.split([self.C, 2, 1, self.C], 1)
        vx, vy = mean.unbind(1)
        directions = torch.stack([vx, vy, -vx, -vy], 1)
        return self.regular(torch.cat([mass, directions, mx], 1)) + self.scalar(spread)
