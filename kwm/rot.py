"""C4 action in (x=column, y=row) coordinates: R(x,y)=(-y,x).

Positive R is torch.rot90(..., -1), since displayed rows increase downwards.
The pivot is the observed window centre, even on the larger world torus.
"""
from functools import lru_cache

import torch
from torch import nn

from .lock_nodes import LockState


def rot_vec(v, k=1):
    x, y = v.unbind(-1)
    return torch.stack(((x, y), (-y, x), (-x, -y), (y, -x))[k % 4], -1)


rot_nodes = rot_vec


class TorusRotation(nn.Module):
    """Precomputed gather permutations; no interpolation or floating arithmetic."""
    def __init__(self, size, window=32, device=None):
        super().__init__()
        y, x = torch.meshgrid(torch.arange(size, device=device),
                              torch.arange(size, device=device), indexing="ij")
        # Inverse action at each destination. Twice the pivot is window - 1.
        indices = []
        for _ in range(4):
            indices.append((y.remainder(size) * size + x.remainder(size)).flatten())
            x, y = y, window - 1 - x
        self.register_buffer("indices", torch.stack(indices), persistent=False)

    def forward(self, x, k=1):
        return x.flatten(-2).index_select(-1, self.indices[k % 4]).reshape_as(x)


@lru_cache(maxsize=32)
def _rotation(size, window, device):
    return TorusRotation(size, window, device)


def rot_map(x, k=1, window=32):
    """Rotate square [...,H,W] maps about ((window-1)/2, (window-1)/2)."""
    if x.shape[-2] != x.shape[-1]:
        raise ValueError("C4 requires a square torus")
    return _rotation(x.shape[-1], window, x.device)(x, k)


def rot_regular(x, k=1, window=32):
    """Spatial action and c=f*4+r -> f*4+(r+k)%4, channel axis -3."""
    if x.shape[-3] % 4:
        raise ValueError("regular channels must be divisible by four")
    z = rot_map(x, k, window)
    return z.unflatten(-3, (-1, 4)).roll(k, -3).reshape_as(z)


def rotation_perms(nodes, tol=1e-6):
    """perm[k,i] is the destination of i; nodes[perm[k,i]] == R**k nodes[i]."""
    perms = []
    for k in range(4):
        d = (rot_nodes(nodes, k)[:, None] - nodes[None]).abs().amax(-1)
        values, index = d.min(-1)
        assert (values <= tol).all(), "node set not closed under C4"
        assert index.unique().numel() == len(nodes), "node rotation is not a permutation"
        perms.append(index)
    return torch.stack(perms)


def rot_state(h, k, lattice, C, window=32, perm_rot=None):
    """Rotate features, node labels/offsets, evidence, and lock/LK imprints."""
    perms = rotation_perms(lattice) if perm_rot is None else perm_rot
    source = perms[(-k) % 4]
    if isinstance(h, LockState):
        metadata = h.nodes[:, source]
        metadata = torch.cat([rot_vec(metadata[..., :2], k), metadata[..., 2:]], -1)
        return LockState(rot_regular(h.features[:, source], k, window), metadata,
                         rot_regular(h.snapshot, k, window))
    h = h[:, source]
    feat = rot_regular(h[:, :, :C], k, window)
    if h.shape[2] == C:
        return feat
    rho = rot_map(h[:, :, C:C + 2], k, window).movedim(2, -1)
    rho = rot_vec(rho, k).movedim(-1, 2)
    return torch.cat([feat, rho, rot_map(h[:, :, C + 2:C + 3], k, window),
                      rot_regular(h[:, :, C + 3:], k, window)], 2)


def rot_episode(obs, actions_in, actions_out, digit_state=None, k=1, *, world=50):
    """Rotate ego-frame pixels, actions, and ego-frame digit centres/velocities.

    Digit positions must already be relative to the camera; world-frame dataset
    positions need camera subtraction before this helper. Other fields carry.
    """
    window = obs.shape[-1]
    state = None if digit_state is None else dict(digit_state)
    if state is not None:
        for key in ("cent", "pos"):
            if key in state:
                state[key] = (rot_vec(state[key] - (window - 1) / 2, k) + (window - 1) / 2).remainder(world)
        if "vel" in state:
            state["vel"] = rot_vec(state["vel"], k)
    return rot_map(obs, k, window), rot_vec(actions_in, k), rot_vec(actions_out, k), state


def normalized_error(expected, actual):
    """Per-sample L2 discrepancy relative to the expected tensor's norm."""
    a, b = expected.float().flatten(1), actual.float().flatten(1)
    return (a - b).norm(dim=1) / a.norm(dim=1).clamp_min(1e-12)
