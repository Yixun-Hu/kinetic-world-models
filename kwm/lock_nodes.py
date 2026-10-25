"""Functional fp32 lock-then-refine estimator with per-node texture and damping.

No estimator history lives here: the cell owns an explicit LockState value.
See docs/PR1_lock_nodes.md for thresholds, state layout and gradient routing.
"""
from functools import lru_cache
from typing import NamedTuple

import torch
import torch.nn.functional as F

from .transport import _phase, lk_displacement


class LockState(NamedTuple):
    features: torch.Tensor  # [B,V,C,H,W], feature/autocast dtype
    nodes: torch.Tensor     # [B,V,NODE_DIM], fp32; layout below (PR1 prefix unchanged)
    snapshot: torch.Tensor  # [B,K,win,win], fp32, shared by all nodes


def features(h):
    return h.features if isinstance(h, LockState) else h


def pad_world(x, world):
    return F.pad(x, (0, world - x.shape[-1], 0, world - x.shape[-2]))


def shift_snapshot(snapshot, shifts, world):
    """One forward FFT per snapshot, broadcast over node/candidate phases.

    snapshot: [B,K,win,win]; shifts: [B,V,2] (or [1,V,2]). The
    inverse FFT still produces each transported view; gradients reach both
    the snapshot and shifts. Phases modify only the materialized node spectra.
    """
    spectrum = torch.fft.rfft2(pad_world(snapshot.float(), world))
    phase = _phase(world, world, shifts[:, :, None], snapshot.device)
    # Match the expanded-snapshot FFT's B,K,V,H,freq storage order. Merely
    # broadcasting spectrum during multiplication changes CPU complex-multiply
    # rounding, which discrete mode decisions can amplify across steps.
    spectrum = spectrum[:, None].expand(-1, shifts.shape[1], -1, -1, -1)
    spectrum = spectrum.transpose(1, 2).contiguous().transpose(1, 2)
    if phase.requires_grad:
        spectrum = spectrum * phase
    else:
        spectrum.mul_(phase)
    return torch.fft.irfft2(spectrum, s=(world, world))


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
    weight *= trace > tr_rel * trace.mean((2, 3), keepdim=True)
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
def two_modes(displacement, weight, nu, lattice, lag, bound=0.75, return_mode=False):
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
    result = (primary * lag, wp, selected)
    mode = torch.where(split, torch.where(take_left, 1., 2.), 0.)
    return (*result, mode) if return_mode else result



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


@lru_cache(maxsize=16)
@torch.no_grad()
def _candidate_geometry(world, win, radius, device):
    """Cache only fixed geometry, never observations or recurrent state."""
    axis = torch.arange(-radius, radius + 0.01, 0.5, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    shifts = torch.stack([xx, yy], -1).reshape(-1, 2)
    mask = shifted_window_mask(-shifts, world, win)[None, None]
    return shifts, mask


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
            self.shifts, self.mask = _candidate_geometry(world, win, radius, current.device)
            self.shifted = shift_snapshot(self.current, -self.shifts[None], world)[..., :win, :win]
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
                     lam_rel=1e-2, tr_rel=0.1, gate=0.6, coh_rel=0.5, bound=0.75, return_mode=False):
    """Return residual, evidence, and the freshly evaluated lock bit.

    LK retains its displacement derivative at locked nodes. Search, all gates,
    clustering and mode selection are detached; the final weighted LK sum is
    the only estimator gradient. The caller applies the learned scale s.
    """
    with torch.autocast(device_type=current.device.type, enabled=False):
        current, previous, nu = current.float(), previous.float(), nu.float()
        shift = nu * dt - action.float()[:, None]
        m = shift_snapshot(previous, shift, world)[..., :current.shape[-2], :current.shape[-1]]
        lk, trace = lk_displacement(m, current, lam_rel, return_trace=True, per_node_damping=True)
        with torch.no_grad():
            support = shifted_window_mask(shift, world, current.shape[-1])
            lw = coherence(m, current, trace, lk, support, dt, tr_rel, gate, coh_rel)
            ld, le = pool(lk, lw)
            textured = m.norm(dim=2) * (trace > tr_rel * trace.mean((2, 3), keepdim=True)) * support
            agreement = le / textured.sum((-2, -1)).clamp_min(1e-6)
            locked = (agreement >= 0.1) & ((ld / dt).norm(dim=-1) <= 0.3)
            d, usable = LocalSearch(current, world)(m, support)
            d = torch.where(locked[..., None, None, None], lk, d)
            usable = torch.where(locked[..., None, None], True, usable)
            weight = coherence(m, current, trace, d, support * usable, dt, tr_rel, gate, coh_rel)
            modes = two_modes(d, weight, nu, lattice, dt, bound, return_mode=return_mode)
            delta, evidence, selected = modes[:3]
        # Keep the forward value exactly equal to the detached mode estimator.
        # Only the selected, locked LK field receives a displacement gradient.
        differentiable = (selected[:, :, None] * lk).sum((-2, -1)) / evidence[..., None].clamp_min(1e-6)
        differentiable = torch.where(locked[..., None], differentiable, 0)
        delta = delta + (differentiable - differentiable.detach())
        result = (delta, evidence, locked.float())
        return (*result, modes[3]) if return_mode else result


# Compact layout shared by both orders. Welford's centred moments avoid
# subtracting large, nearly equal raw second moments in fp32.
# 0:4 PR1; 4 W_fit, 5 mean_t, 6 Ctt, 7:9 mean_z, 9:11 Ctz,
# 11 sum(w²), 12 Czz (trace); 13 selected split mode; 14 next observation time;
# 15:17 additive instantaneous-velocity correction (carried during prediction).
SW, MT, CTT, MZ, CTZ, SW2, CZZ, MODE, TIME, CORR = 4, 5, 6, slice(7, 9), slice(9, 11), 11, 12, 13, 14, slice(15, 17)
NODE_DIM = 17


@torch.no_grad()
def acceleration_fit(state, kappa=9., inflation=4.):
    """Return detached slope, scalar gate, trace Var(slope), n_eff.

    n_eff = (sum w)^2 / sum(w²); residual scatter has denominator
    W - 2 W/n_eff. The trace of the 2D slope covariance is scatter/Ctt,
    inflated for shared-frame displacement errors. No model scale n_s enters.
    """
    w, ctt = state[..., SW], state[..., CTT]
    neff = w.square() / state[..., SW2].clamp_min(1e-12)
    valid = (neff >= 3) & (ctt >= 1e-6)
    slope = state[..., CTZ] / ctt[..., None].clamp_min(1e-6)
    slope = torch.where(valid[..., None], slope, 0)
    rss = (state[..., CZZ] - (state[..., CTZ] * slope).sum(-1)).clamp_min(0)
    dof = w * (1 - 2 / neff.clamp_min(1))
    variance = inflation * rss / dof.clamp_min(1e-6) / ctt.clamp_min(1e-6)
    if kappa == float('inf'):
        gate = torch.zeros_like(w)
    else:
        power = slope.square().sum(-1)
        gate = (1 - kappa * variance / power.clamp_min(1e-20)).clamp(0, 1)
        gate = torch.where(valid & (power > 1e-20), gate, 0)
    return slope, gate, variance, neff


@torch.no_grad()
def accumulate_acceleration(state, z, weight, time, locked, mode):
    """Detached weighted centred sums; reset on a split-mode or unlock jump.

    mode 0 = unsplit, 1/2 = canonical principal-axis left/right. Compare
    consecutive split identities; disappearing/reappearing splits alone do
    not constitute a switch. Reset before consuming the new measurement.
    """
    out = state.detach().float().clone()
    switched = (state[..., MODE] > 0) & (mode > 0) & (state[..., MODE] != mode)
    reset = switched | ((state[..., 3] > .5) & (locked < .5))
    out[..., 4:13] = torch.where(reset[..., None], 0, out[..., 4:13])
    w0 = out[..., SW].clone()
    w = weight.detach().float()
    total = w0 + w
    gain = w / total.clamp_min(1e-12)
    dt = time - out[..., MT]
    dz = z.detach().float() - out[..., MZ]
    cross = w0 * gain
    out[..., CTT] += cross * dt.square()
    out[..., CTZ] += (cross * dt)[..., None] * dz
    out[..., CZZ] += cross * dz.square().sum(-1)
    out[..., MT] += gain * dt
    out[..., MZ] += gain[..., None] * dz
    out[..., SW] = total
    out[..., SW2] += w.square()
    out[..., MODE] = mode
    return out


def corrected_offsets(state, bound=.75):
    """Instantaneous offset, with an exact PR1 branch when correction is zero."""
    rho, correction = state[..., :2], state[..., CORR]
    return torch.where(correction != 0, (rho + correction).clamp(-bound, bound), rho)


def transport_velocity(state, lattice, dt, kappa=9., inflation=4., bound=.75):
    """Outgoing interval-mean velocity: v(t) + .5 gated_a dt.

    The stream operator multiplies this by dt. Disabled gates preserve every
    PR1 operation on velocity, including the lattice + rho addition.
    """
    slope, gate, _, _ = acceleration_fit(state, kappa, inflation)
    nu = lattice[None].float() + corrected_offsets(state, bound)
    half = .5 * (gate[..., None] * slope) * dt
    return torch.where(half != 0, nu + half, nu)
