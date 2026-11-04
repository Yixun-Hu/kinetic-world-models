"""Functional fp32 lock-then-refine estimator with per-node texture and damping.

No estimator history lives here: the cell owns an explicit LockState value.
See docs/PR1_lock_nodes.md for thresholds, state layout and gradient routing.
"""
from functools import lru_cache
import math
from typing import NamedTuple

import torch
import torch.nn.functional as F

from .transport import _phase, lk_displacement, lk_support, masked_mean


class LockState(NamedTuple):
    features: torch.Tensor  # [B,V,C,H,W], feature/autocast dtype
    nodes: torch.Tensor     # [B,V,4], fp32: rho_x, rho_y, evidence, locked
    snapshot: torch.Tensor  # [B,K,Hb,Wb], fp32, shared by all nodes


def features(h):
    return h.features if isinstance(h, LockState) else h


def _world_shape(world):
    return (world, world) if isinstance(world, int) else tuple(world)


def pad_world(x, world, box=None):
    """Embed an observed rectangle at box=(row,col), wrapping on the torus."""
    H, W = _world_shape(world)
    if x.shape[-2] > H or x.shape[-1] > W:
        raise ValueError("snapshot cannot be larger than the world")
    padded = F.pad(x, (0, W - x.shape[-1], 0, H - x.shape[-2]))
    return padded if box is None else torch.roll(padded, tuple(box), (-2, -1))


def crop_box(x, shape, box=None):
    """Gather an observed rectangle, including boxes crossing either seam."""
    h, w = shape
    if box is None:
        return x[..., :h, :w]
    rows = (torch.arange(h, device=x.device) + box[0]) % x.shape[-2]
    cols = (torch.arange(w, device=x.device) + box[1]) % x.shape[-1]
    return x.index_select(-2, rows).index_select(-1, cols)


def shift_snapshot(snapshot, shifts, world, *, c4=False, box=None):
    """One forward FFT per snapshot, broadcast over node/candidate phases.

    snapshot: [B,K,Hb,Wb]; shifts: [B,V,2] (or [1,V,2]). Returns full world maps. The
    inverse FFT still produces each transported view; gradients reach both
    the snapshot and shifts. Phases modify only the materialized node spectra.
    """
    H, W = _world_shape(world)
    spectrum = torch.fft.rfft2(pad_world(snapshot.float(), world, box))
    phase = _phase(H, W, shifts[:, :, None], snapshot.device, c4=c4)
    # Match the expanded-snapshot FFT's B,K,V,H,freq storage order. Merely
    # broadcasting spectrum during multiplication changes CPU complex-multiply
    # rounding, which discrete mode decisions can amplify across steps.
    spectrum = spectrum[:, None].expand(-1, shifts.shape[1], -1, -1, -1)
    spectrum = spectrum.transpose(1, 2).contiguous().transpose(1, 2)
    if phase.requires_grad:
        spectrum = spectrum * phase
    else:
        spectrum.mul_(phase)
    return torch.fft.irfft2(spectrum, s=(H, W))


def gaussian(x, sigma=2.0, radius=6):
    """Separable Gaussian; preserves arbitrary leading dimensions."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = math.ceil(3 * sigma) if radius is None else radius
    if not isinstance(radius, int) or radius < 0:
        raise ValueError("Gaussian radius must be a nonnegative integer or None")
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


def coherence(m, current, trace, displacement, support=None, lag=1, tr_rel=0.1, gate=0.6, coh_rel=0.5,
              *, mean_support=None):
    """KWM's mass, texture, residual-size and *unwarped* agreement gates."""
    weight = m.norm(dim=2)
    reference = trace.mean((2, 3), keepdim=True) if mean_support is None else masked_mean(trace, mean_support)
    weight *= trace > tr_rel * reference
    weight *= displacement.abs().amax(2) / lag <= gate
    if mean_support is None:
        sigma = (coh_rel * current.norm(dim=1).mean((1, 2)) + 1e-6)[:, None, None, None]
    else:
        magnitude = current.norm(dim=1)[:, None].expand_as(trace)
        sigma = coh_rel * masked_mean(magnitude, mean_support) + 1e-6
    mismatch = (m - current[:, None]).square().sum(2)
    weight *= torch.exp(-mismatch / sigma.square())
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



def shifted_window_mask(shifts, world, win, *, box=None, support_mask=None):
    """Bilinear support on the world torus, without Fourier mask ringing."""
    if box is not None or support_mask is not None or not isinstance(world, int) or not isinstance(win, int):
        H, W = _world_shape(world)
        h, w = (win, win) if isinstance(win, int) else tuple(win)
        row, col = (0, 0) if box is None else box
        y, x = torch.meshgrid(torch.arange(h, device=shifts.device) + row,
                              torch.arange(w, device=shifts.device) + col, indexing="ij")
        coord = torch.stack([x, y], -1).float() - shifts[..., None, None, :]
        size = shifts.new_tensor((W, H))
        coord = coord.remainder(size) + 1
        grid = (2 * coord / (size + 1) - 1).reshape(-1, h, w, 2)
        mask = (torch.ones(h, w, device=shifts.device) if support_mask is None
                else support_mask.to(device=shifts.device, dtype=torch.float32))
        mask = pad_world(mask[None, None], world, box)
        mask = F.pad(mask, (1, 1, 1, 1), mode="circular").expand(grid.shape[0], -1, -1, -1)
        return F.grid_sample(mask, grid, align_corners=True)[:, 0].reshape(*shifts.shape[:-1], h, w)
    y, x = torch.meshgrid(torch.arange(win, device=shifts.device),
                          torch.arange(win, device=shifts.device), indexing="ij")
    coord = torch.stack([x, y], -1).float() - shifts[..., None, None, :]
    coord = coord.remainder(world) + 1
    grid = (2 * coord / (world + 1) - 1).reshape(-1, win, win, 2)
    mask = pad_world(torch.ones(1, 1, win, win, device=shifts.device), world)
    mask = F.pad(mask, (1, 1, 1, 1), mode="circular").expand(grid.shape[0], -1, -1, -1)
    return F.grid_sample(mask, grid, align_corners=True)[:, 0].reshape(*shifts.shape[:-1], win, win)


@lru_cache(maxsize=16)
@torch.no_grad()
def _candidate_geometry(world, win, radius, device, box=None, mask_key=None):
    """Cache only fixed geometry, never observations or recurrent state."""
    axis = torch.arange(-radius, radius + 0.01, 0.5, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    shifts = torch.stack([xx, yy], -1).reshape(-1, 2)
    if box is None and mask_key is None:
        mask = shifted_window_mask(-shifts, world, win)[None, None]
    else:
        support_mask = None if mask_key is None else torch.tensor(mask_key, device=device, dtype=torch.bool).reshape(win)
        mask = shifted_window_mask(-shifts, world, win, box=box, support_mask=support_mask)[None, None]
    return shifts, mask


class LocalSearch:
    """Dense residual SSD volume, Gaussian aggregation, parabolic refinement.

    Compare m(x) to current(x+d). Shifts of the current snapshot are shared
    across nodes/lags. Candidate chunks limit transient memory (no image bank).
    """
    @torch.no_grad()
    def __init__(self, current, world, candidate_chunk=7, radius=1.0, *, c4=False,
                 box=None, support_mask=None, sigma=2.0, gaussian_radius=6):
        win = current.shape[-1]
        self.current = current.float()
        self.chunk = candidate_chunk
        self.c4 = c4
        self.sigma, self.gaussian_radius = sigma, gaussian_radius
        if radius not in (1.0, 1.5):
            raise ValueError("Search radius must be 1.0 or 1.5 pixels")
        self.radius = radius
        self.side = int(4 * radius + 1)
        self.count = self.side**2
        with torch.autocast(device_type=current.device.type, enabled=False):
            shift_args = {"c4": True} if c4 else {}
            if box is None and support_mask is None and isinstance(world, int) and current.shape[-2] == win:
                self.shifts, self.mask = _candidate_geometry(world, win, radius, current.device)
                self.shifted = shift_snapshot(self.current, -self.shifts[None], world, **shift_args)[..., :win, :win]
            else:
                shape = tuple(current.shape[-2:])
                box = (0, 0) if box is None else tuple(box)
                # The mask is fixed geometry. A value key avoids stale cached
                # masks after an in-place edit, or aliases of equal-shaped cones.
                mask_key = None if support_mask is None else tuple(support_mask.detach().cpu().flatten().tolist())
                self.shifts, self.mask = _candidate_geometry(_world_shape(world), shape, radius,
                                                              current.device, box, mask_key)
                if support_mask is not None:
                    self.current = self.current * support_mask.to(self.current)
                self.shifted = crop_box(shift_snapshot(self.current, -self.shifts[None], world,
                                                       box=box, **shift_args), shape, box)
            self.shifted_norm = self.shifted.square().sum(2)

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
            if self.c4:
                # Near matching patches, norm + norm - 2*dot cancels most
                # significant digits. Direct differences keep parabolic search
                # refinement stable under fp32 rotation/reduction roundoff.
                ssd = (transported[:, :, None] - shifted[:, None]).square().sum(3)
            else:
                dot = torch.einsum("bvchw,bkchw->bvkhw", transported, shifted)
                ssd = (transported_norm[:, :, None]
                       + self.shifted_norm[:, None, start:stop] - 2 * dot).clamp_min(0)
            mass = gaussian(valid, self.sigma, self.gaussian_radius)
            cost = gaussian(ssd * valid, self.sigma, self.gaussian_radius) / mass.clamp_min(1e-6)
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
                     lam_rel=1e-2, tr_rel=0.1, gate=0.6, coh_rel=0.5, bound=0.75, *, c4=False,
                     box=None, support_mask=None, sigma=2.0, radius=6, static_residual=False):
    """Return residual, evidence, and the freshly evaluated lock bit.

    LK retains its displacement derivative at locked nodes. Search, all gates,
    clustering and mode selection are detached; the final weighted LK sum is
    the only estimator gradient. The caller applies the learned scale s.

    box=(row0,col0) embeds [B,K,Hb,Wb] snapshots on an int or (H,W)
    torus. Optional boolean support_mask excludes unobserved pixels; masked
    LK erodes transported overlap by one pixel before taking differences.
    sigma/radius control SSD aggregation, not the fixed +/-1 candidate grid.
    static_residual subtracts the ego-only transported previous snapshot from
    both images and excludes node zero (which must be the static lattice node).
    """
    if box is not None or support_mask is not None or static_residual or not isinstance(world, int):
        return _lock_box(previous, current, nu, lattice, action, world, dt, lam_rel, tr_rel, gate, coh_rel,
                         bound, c4=c4, box=box, support_mask=support_mask, sigma=sigma, radius=radius,
                         static_residual=static_residual)
    if previous.shape[-2] != previous.shape[-1] or previous.shape != current.shape:
        raise ValueError("matching square snapshots required without box")
    with torch.autocast(device_type=current.device.type, enabled=False):
        current, previous, nu = current.float(), previous.float(), nu.float()
        shift = nu * dt - action.float()[:, None]
        shift_args = {"c4": True} if c4 else {}
        m = shift_snapshot(previous, shift, world, **shift_args)[..., :current.shape[-2], :current.shape[-1]]
        lk, trace = lk_displacement(m, current, lam_rel, return_trace=True, per_node_damping=True)
        with torch.no_grad():
            support = shifted_window_mask(shift, world, current.shape[-1])
            lw = coherence(m, current, trace, lk, support, dt, tr_rel, gate, coh_rel)
            ld, le = pool(lk, lw)
            textured = m.norm(dim=2) * (trace > tr_rel * trace.mean((2, 3), keepdim=True)) * support
            agreement = le / textured.sum((-2, -1)).clamp_min(1e-6)
            locked = (agreement >= 0.1) & ((ld / dt).norm(dim=-1) <= 0.3)
            d, usable = LocalSearch(current, world, c4=c4, sigma=sigma, gaussian_radius=radius)(m, support)
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


def _lock_box(previous, current, nu, lattice, action, world, dt, lam_rel, tr_rel, gate, coh_rel, bound,
              *, c4, box, support_mask, sigma, radius, static_residual):
    """Opt-in geometry/masking path; the legacy estimator above is unchanged."""
    if previous.shape != current.shape or previous.ndim != 4:
        raise ValueError("snapshots must have matching [B,K,Hb,Wb] shapes")
    shape = tuple(current.shape[-2:])
    box = (0, 0) if box is None else tuple(box)
    if support_mask is None:
        support_mask = torch.ones(shape, dtype=torch.bool, device=current.device)
    if support_mask.shape != shape or support_mask.dtype != torch.bool:
        raise ValueError("support_mask must be bool [Hb,Wb]")
    support_mask = support_mask.to(current.device)
    if static_residual and not torch.equal(lattice[0], torch.zeros_like(lattice[0])):
        raise ValueError("static_residual requires lattice[0] == (0,0)")
    with torch.autocast(device_type=current.device.type, enabled=False):
        current = current.float() * support_mask
        previous = previous.float() * support_mask
        nu = nu.float()
        shift = nu * dt - action.float()[:, None]
        # Append the ego-only phase to the same FFT bank, independently of rho_0.
        shifts = torch.cat([shift, -action.float()[:, None]], 1) if static_residual else shift
        transported = crop_box(shift_snapshot(previous, shifts, world, c4=c4, box=box), shape, box)
        with torch.no_grad():
            support = shifted_window_mask(shifts, world, shape, box=box, support_mask=support_mask) * support_mask
        if static_residual:
            m0 = transported[:, -1]
            m, current = transported[:, :-1] - m0[:, None], current - m0
            support = support[:, :-1] * support[:, -1:]
        else:
            m = transported
        lk, trace = lk_displacement(m, current, lam_rel, return_trace=True, per_node_damping=True,
                                    support_mask=support)
        with torch.no_grad():
            support = lk_support(support)
            if static_residual:
                # A perfectly static target is zero up to FFT roundoff. Require
                # actual current residual signal, so numerical dust cannot turn
                # a background-only pair into moving evidence.
                floor = 32 * torch.finfo(torch.float32).eps * m0.norm(dim=1).amax((-2, -1), keepdim=True)
                support = support * (current.norm(dim=1) > floor)[:, None]
                support[:, 0] = 0
            lw = coherence(m, current, trace, lk, support, dt, tr_rel, gate, coh_rel, mean_support=support)
            ld, le = pool(lk, lw)
            textured = m.norm(dim=2) * (trace > tr_rel * masked_mean(trace, support)) * support
            agreement = le / textured.sum((-2, -1)).clamp_min(1e-6)
            locked = (agreement >= 0.1) & ((ld / dt).norm(dim=-1) <= 0.3)
            d, usable = LocalSearch(current, world, c4=c4, box=box, support_mask=support_mask,
                                     sigma=sigma, gaussian_radius=radius)(m, support)
            d = torch.where(locked[..., None, None, None], lk, d)
            usable = torch.where(locked[..., None, None], True, usable)
            valid = support * usable
            weight = coherence(m, current, trace, d, valid, dt, tr_rel, gate, coh_rel, mean_support=valid)
            delta, evidence, selected = two_modes(d, weight, nu, lattice, dt, bound)
        differentiable = (selected[:, :, None] * lk).sum((-2, -1)) / evidence[..., None].clamp_min(1e-6)
        differentiable = torch.where(locked[..., None], differentiable, 0)
        delta = delta + (differentiable - differentiable.detach())
        return delta, evidence, locked.float()


def measure_nodes(previous, current, nu, lattice, action, **kw):
    """Thin velocity view of lock_measurement: (nu + delta/dt, evidence, lock)."""
    delta, evidence, locked = lock_measurement(previous, current, nu, lattice, action, **kw)
    return nu + delta / kw.get("dt", 1.0), evidence, locked
