"""Functional fp32 lock/search estimator, ported from estimator_gate D2 heavier.

No estimator history lives here: the cell owns an explicit LockState value.
"""
from typing import NamedTuple

import torch
import torch.nn.functional as F

from .transport import spectral_shift, lk_displacement


class LockState(NamedTuple):
    features: torch.Tensor  # [B,V,C,H,W], feature/autocast dtype
    nodes: torch.Tensor     # [B,V,4], fp32: rho_x, rho_y, evidence, locked
    snapshot: torch.Tensor  # [B,K,win,win], fp32, shared by all nodes
    camera: torch.Tensor    # [B,2], fp32 cumulative action at the snapshot


def features(h):
    return h.features if isinstance(h, LockState) else h


def pad_world(x, world):
    return F.pad(x, (0, world - x.shape[-1], 0, world - x.shape[-2]))


def gaussian(x, sigma=2.0, radius=6):
    """Separable Gaussian; preserves arbitrary leading dimensions."""
    shape = x.shape
    # Depthwise channels avoid the large internal channel padding of a batch
    # of single-channel convolutions on CPU. Algebra and border policy match.
    z = x.reshape(1, -1, *shape[-2:])
    channels = z.shape[1]
    axis = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (axis / sigma).square())
    k = k / k.sum()
    z = F.conv2d(F.pad(z, (radius, radius, 0, 0), mode="replicate"),
                 k[None, None, None].expand(channels, 1, 1, -1), groups=channels)
    z = F.conv2d(F.pad(z, (0, 0, radius, radius), mode="replicate"),
                 k[None, None, :, None].expand(channels, 1, -1, 1), groups=channels)
    return z.reshape(shape)


def coherence(m, current, trace, displacement, support=None, lag=1, tr_rel=0.1, gate=0.6, coh_rel=0.5):
    """KWM's mass, texture, residual-size and *unwarped* agreement gates."""
    weight = m.norm(dim=2)
    weight *= trace > tr_rel * trace.mean((1, 2, 3), keepdim=True)
    weight *= displacement.abs().amax(2) / lag <= gate
    sigma = coh_rel * current.norm(dim=1).mean((1, 2)) + 1e-6
    mismatch = (m - current[:, None]).square().sum(2)
    weight *= torch.exp(-mismatch / sigma[:, None, None, None].square())
    if support is not None:
        weight *= support
    return weight


def pool(displacement, weight):
    evidence = weight.sum((-2, -1))
    delta = (weight[:, :, None] * displacement).sum((-2, -1)) / (evidence[..., None] + 1e-6)
    return delta, evidence


@torch.no_grad()
def two_modes(displacement, weight, nu, lattice, lag, bound=0.75):
    """Weighted k=2 along the principal axis, within a node's reachable cell.

    Up to two modes: require >=10% mass in each and separation >=0.1 px/frame.
    Select the heavier mode (left wins ties); return its detached pixel weights.
    """
    dv = (displacement / lag).flatten(-2).transpose(-1, -2)  # B,V,pixel,xy
    w = weight.flatten(-2)
    inside = ((nu[:, :, None] + dv - lattice[None, :, None]).abs() <= bound).all(-1)
    w = w * inside
    total = w.sum(-1)
    mean = (w[..., None] * dv).sum(-2) / total[..., None].clamp_min(1e-6)
    centered = dv - mean[:, :, None]
    cov = torch.einsum("bvp,bvpi,bvpj->bvij", w, centered, centered)
    angle = 0.5 * torch.atan2(2 * cov[..., 0, 1], cov[..., 0, 0] - cov[..., 1, 1])
    axis = torch.stack([angle.cos(), angle.sin()], -1)
    proj = (centered * axis[:, :, None]).sum(-1)
    std = ((w * proj.square()).sum(-1) / total.clamp_min(1e-6)).sqrt()
    lo, hi = -std, std
    for _ in range(5):
        left = (proj - lo[..., None]).abs() <= (proj - hi[..., None]).abs()
        wl, wh = w * left, w * ~left
        sl, sh = wl.sum(-1), wh.sum(-1)
        lo = (wl * proj).sum(-1) / sl.clamp_min(1e-6)
        hi = (wh * proj).sum(-1) / sh.clamp_min(1e-6)
    vl = (wl[..., None] * dv).sum(-2) / sl[..., None].clamp_min(1e-6)
    vh = (wh[..., None] * dv).sum(-2) / sh[..., None].clamp_min(1e-6)
    split = (torch.minimum(sl, sh) >= 0.1 * total) & (total > 1e-6)
    split &= (vl - vh).norm(dim=-1) >= 0.1
    take_left = sl >= sh
    primary = torch.where(take_left[..., None], vl, vh)
    wp = torch.where(take_left, sl, sh)
    primary = torch.where(split[..., None], primary, mean)
    wp = torch.where(split, wp, total)
    selected = torch.where(take_left[..., None], wl, wh)
    selected = torch.where(split[..., None], selected, w).reshape_as(weight)
    return primary * lag, wp, selected



def shifted_window_mask(shifts, world, win):
    """Bilinear support on the world torus, without Fourier mask ringing."""
    y, x = torch.meshgrid(torch.arange(win, device=shifts.device),
                          torch.arange(win, device=shifts.device), indexing="ij")
    coord = torch.stack([x, y], -1).float() - shifts[..., None, None, :]
    coord = coord.remainder(world) + 1
    grid = (2 * coord / (world + 1) - 1).reshape(-1, win, win, 2)
    mask = pad_world(torch.ones(1, 1, win, win, device=shifts.device), world)
    mask = F.pad(mask, (1, 1, 1, 1), mode="circular").expand(grid.shape[0], -1, -1, -1)
    return F.grid_sample(mask, grid, align_corners=True)[:, 0].reshape(*shifts.shape[:-1], win, win)


class LocalSearch:
    """Dense residual SSD volume, sigma-2 aggregation, parabolic refinement.

    Compare m(x) to current(x+d). Shifts of the current snapshot are shared
    across nodes/lags. Candidate chunks limit transient memory (no image bank).
    """
    @torch.no_grad()
    def __init__(self, current, world, candidate_chunk=7, radius=1.0):
        win = current.shape[-1]
        self.current = current.float()
        self.chunk = candidate_chunk
        if radius not in (1.0, 1.5):
            raise ValueError("Search radius must be 1.0 or 1.5 pixels")
        self.radius = radius
        self.side = int(4 * radius + 1)
        self.count = self.side**2
        with torch.autocast(device_type=current.device.type, enabled=False):
            axis = torch.arange(-radius, radius + 0.01, 0.5, device=current.device)
            yy, xx = torch.meshgrid(axis, axis, indexing="ij")
            self.shifts = torch.stack([xx, yy], -1).reshape(self.count, 2)
            full = pad_world(self.current, world)[:, None].expand(-1, self.count, -1, -1, -1)
            self.shifted = spectral_shift(full, -self.shifts[None])[..., :win, :win]
            self.shifted_norm = self.shifted.square().sum(2)
            self.mask = shifted_window_mask(-self.shifts, world, win)[None, None]

    @torch.no_grad()
    def __call__(self, transported, support):
        with torch.autocast(device_type=transported.device.type, enabled=False):
            return self.search(transported.float(), support.float())

    def search(self, transported, support):
        costs = []
        transported_norm = transported.square().sum(2)
        for start in range(0, self.count, self.chunk):
            stop = min(start + self.chunk, self.count)
            valid = support[:, :, None] * self.mask[:, :, start:stop]
            shifted = self.shifted[:, start:stop]
            # Contract channels directly: never allocate B,V,candidate,C,H,W.
            dot = torch.einsum("bvchw,bkchw->bvkhw", transported, shifted)
            ssd = (transported_norm[:, :, None]
                   + self.shifted_norm[:, None, start:stop] - 2 * dot).clamp_min(0)
            mass = gaussian(valid)
            cost = gaussian(ssd * valid) / mass.clamp_min(1e-6)
            costs.append(cost.masked_fill(mass < 0.5, 1e6))
        costs = torch.cat(costs, 2)
        index = costs.argmin(2)
        iy, ix = index // self.side, index % self.side

        def at(i):
            return costs.gather(2, i[:, :, None])[:, :, 0]

        center = at(index)
        refined = []
        for coordinate, stride in ((ix, 1), (iy, self.side)):
            left = at((index - stride).clamp_min(0))
            right = at((index + stride).clamp_max(self.count - 1))
            curvature = left + right - 2 * center
            offset = (0.5 * (left - right) / curvature.clamp_min(1e-8)).clamp(-0.5, 0.5)
            interior = (coordinate > 0) & (coordinate < self.side - 1) & (curvature > 1e-8)
            refined.append(-self.radius + 0.5 * (coordinate + torch.where(interior, offset, 0)))
        displacement = torch.stack(refined, 2)
        usable = (center < 1e5) & (support > 0.99)
        # Edge minima have no two-sided refinement. At long lags their clamped
        # displacement can pass the velocity-size gate and corrupt the fit.
        usable &= (ix > 0) & (ix < self.side - 1) & (iy > 0) & (iy < self.side - 1)
        return displacement, usable


def lock_measurement(previous, current, nu, lattice, action, world, dt=1.0,
                     lam_rel=1e-2, tr_rel=0.1, gate=0.6, coh_rel=0.5, bound=0.75):
    """Return residual, evidence, and the freshly evaluated lock bit.

    LK retains its displacement derivative at locked nodes. Search, all gates,
    clustering and mode selection are detached; the final weighted LK sum is
    the only estimator gradient. The caller applies the learned scale s.
    """
    with torch.autocast(device_type=current.device.type, enabled=False):
        current, previous, nu = current.float(), previous.float(), nu.float()
        shift = nu * dt - action.float()[:, None]
        m = spectral_shift(pad_world(previous, world)[:, None].expand(-1, nu.shape[1], -1, -1, -1),
                           shift)[..., :current.shape[-2], :current.shape[-1]]
        lk, trace = lk_displacement(m, current, lam_rel, return_trace=True)
        with torch.no_grad():
            support = shifted_window_mask(shift, world, current.shape[-1])
            lw = coherence(m, current, trace, lk, support, dt, tr_rel, gate, coh_rel)
            ld, le = pool(lk, lw)
            textured = m.norm(dim=2) * (trace > tr_rel * trace.mean((1, 2, 3), keepdim=True)) * support
            agreement = le / textured.sum((-2, -1)).clamp_min(1e-6)
            locked = (agreement >= 0.1) & ((ld / dt).norm(dim=-1) <= 0.3)
            d, usable = LocalSearch(current, world)(m, support)
            d = torch.where(locked[..., None, None, None], lk, d)
            usable = torch.where(locked[..., None, None], True, usable)
            weight = coherence(m, current, trace, d, support * usable, dt, tr_rel, gate, coh_rel)
            delta, evidence, selected = two_modes(d, weight, nu, lattice, dt, bound)
        # Keep the forward value exactly equal to the detached mode estimator.
        # Only the selected, locked LK field receives a displacement gradient.
        differentiable = (selected[:, :, None] * lk).sum((-2, -1)) / evidence[..., None].clamp_min(1e-6)
        differentiable = torch.where(locked[..., None], differentiable, 0)
        delta = delta + (differentiable - differentiable.detach())
        return delta, evidence, locked.float()
