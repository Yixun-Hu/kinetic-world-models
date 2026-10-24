#!/usr/bin/env python3
"""No-learning, online velocity-estimator gate; see docs/GATE_estimator.md.

Example (the script itself restricts CUDA to physical GPU 1):
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/estimator_gate.py \
      --n 300 --device cpu --estimators A,B,C,D,E --out results/gate_estimator.json

Only observations and exact camera actions reach the estimators. Labels enter
metrics and the explicitly labelled oracle. No training, autocast, or TF32.
Persistent image history is per sample, never a per-node lag bank. To reproduce
the original LK recurrence under that constraint, A replays its imprint from
snapshots and past node velocities; this extra work is included in its timing.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

# Set these before importing torch/numpy. Never touch physical GPU 0.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

from kwm.cells import AdaptiveNodeCell
from kwm.collision import lattice_nodes
from kwm.env import EnvConfig, make_batch
from kwm.transport import lk_displacement, spectral_shift

WORLD, WIN, NODES, CHANNELS = 50, 32, 25, 8
BOUND, THRESHOLD = 0.75, 0.1


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def pad_world(x):
    return F.pad(x, (0, WORLD - x.shape[-1], 0, WORLD - x.shape[-2]))


def wrap(x):
    return (x + WORLD / 2) % WORLD - WORLD / 2


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


def texture_trace(m):
    gx = (torch.roll(m, -1, -1) - torch.roll(m, 1, -1)) / 2
    gy = (torch.roll(m, -1, -2) - torch.roll(m, 1, -2)) / 2
    # The same trace as lk_displacement (5x5, sigma=1).
    return gaussian((gx.square() + gy.square()).sum(2) / 2, 1.0, 2)


def coherence(m, current, trace, displacement, support=None, lag=1):
    """KWM's mass, texture, residual-size and *unwarped* agreement gates."""
    weight = m.norm(dim=2)
    weight *= trace > 0.1 * trace.mean((1, 2, 3), keepdim=True)
    weight *= displacement.abs().amax(2) / lag <= 0.6
    sigma = 0.5 * current.norm(dim=1).mean((1, 2)) + 1e-6
    mismatch = (m - current[:, None]).square().sum(2)
    weight *= torch.exp(-mismatch / sigma[:, None, None, None].square())
    if support is not None:
        weight *= support
    return weight


def pool(displacement, weight):
    evidence = weight.sum((-2, -1))
    delta = (weight[:, :, None] * displacement).sum((-2, -1)) / (evidence[..., None] + 1e-6)
    return delta, evidence


def robust_velocity(velocities, weights, lags):
    """Order-1 fit_motion: displacement WLS and two Cauchy steps, scale .15.

    Algebra adapted from flowm/experiments/continuous_velocity/kinetic/memory.py
    (fit_motion) and model.py (fit_velocity); no cross-repository import.
    """
    v, w = torch.stack(velocities, 2), torch.stack(weights, 2)
    dt = v.new_tensor(lags)
    base = w * dt.square()
    rw = base
    for iteration in range(3):
        fit = (rw[..., None] * v).sum(2) / rw.sum(2, keepdim=True).clamp_min(1e-6)
        if iteration < 2:
            rw = base / (1 + (v - fit[:, :, None]).square().sum(-1) / 0.15**2)
    return fit, w.sum(2)


def two_modes(displacement, weight, nu, lattice, lag):
    """Weighted k=2 along the principal axis, within a node's reachable cell.

    Up to two modes: require >=10% mass in each and separation >=0.1 px/frame.
    The nearer mode updates the actual node. The other is diagnostic only.
    """
    dv = (displacement / lag).flatten(-2).transpose(-1, -2)  # B,V,pixel,xy
    w = weight.flatten(-2)
    inside = ((nu[:, :, None] + dv - lattice[None, :, None]).abs() <= BOUND).all(-1)
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
    split &= (vl - vh).norm(dim=-1) >= THRESHOLD
    take_left = vl.square().sum(-1) <= vh.square().sum(-1)
    primary = torch.where(take_left[..., None], vl, vh)
    second = torch.where(take_left[..., None], vh, vl)
    wp = torch.where(take_left, sl, sh)
    ws = torch.where(take_left, sh, sl)
    primary = torch.where(split[..., None], primary, mean)
    wp = torch.where(split, wp, total)
    ws = torch.where(split, ws, torch.zeros_like(ws))
    return primary * lag, wp, nu + second, ws


def shifted_window_mask(shifts):
    """Bilinear support on the world torus, without Fourier mask ringing."""
    y, x = torch.meshgrid(torch.arange(WIN, device=shifts.device),
                          torch.arange(WIN, device=shifts.device), indexing="ij")
    coord = torch.stack([x, y], -1).float() - shifts[..., None, None, :]
    coord = coord.remainder(WORLD) + 1
    grid = (2 * coord / (WORLD + 1) - 1).reshape(-1, WIN, WIN, 2)
    mask = pad_world(torch.ones(1, 1, WIN, WIN, device=shifts.device))
    mask = F.pad(mask, (1, 1, 1, 1), mode="circular").expand(grid.shape[0], -1, -1, -1)
    return F.grid_sample(mask, grid, align_corners=True)[:, 0].reshape(*shifts.shape[:-1], WIN, WIN)


class LocalSearch:
    """7x7 residual SSD volume, sigma-2 aggregation, parabolic refinement.

    Compare m(x) to current(x+d). Shifts of the current snapshot are shared
    across nodes/lags. Candidate chunks limit transient memory (no image bank).
    """
    def __init__(self, current, candidate_chunk=7):
        self.current = current
        self.chunk = candidate_chunk
        axis = torch.arange(-1.5, 1.51, 0.5, device=current.device)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.shifts = torch.stack([xx, yy], -1).reshape(49, 2)
        full = pad_world(current)[:, None].expand(-1, 49, -1, -1, -1)
        self.shifted = spectral_shift(full, -self.shifts[None])[..., :WIN, :WIN]
        self.mask = shifted_window_mask(-self.shifts)[None, None]

    def __call__(self, transported, support, sample_indices=None):
        costs = []
        for start in range(0, 49, self.chunk):
            stop = min(start + self.chunk, 49)
            valid = support[:, :, None] * self.mask[:, :, start:stop]
            shifted = self.shifted[:, start:stop]
            if sample_indices is not None:
                shifted = shifted[sample_indices]
            diff = transported[:, :, None] - shifted[:, None]
            ssd = diff.square().sum(3)
            mass = gaussian(valid)
            cost = gaussian(ssd * valid) / mass.clamp_min(1e-6)
            costs.append(cost.masked_fill(mass < 0.5, 1e6))
        costs = torch.cat(costs, 2)
        index = costs.argmin(2)
        iy, ix = index // 7, index % 7

        def at(i):
            return costs.gather(2, i[:, :, None])[:, :, 0]

        center = at(index)
        refined = []
        for coordinate, stride in ((ix, 1), (iy, 7)):
            left = at((index - stride).clamp_min(0))
            right = at((index + stride).clamp_max(48))
            curvature = left + right - 2 * center
            offset = (0.5 * (left - right) / curvature.clamp_min(1e-8)).clamp(-0.5, 0.5)
            interior = (coordinate > 0) & (coordinate < 6) & (curvature > 1e-8)
            refined.append(-1.5 + 0.5 * (coordinate + torch.where(interior, offset, 0)))
        displacement = torch.stack(refined, 2)
        usable = (center < 1e5) & (support > 0.99)
        return displacement, usable


# The peak interpolant/Newton acceptance below is adapted from
# flowm/experiments/continuous_velocity/kinetic/transport.py (refine_peak).
# Original FloWM MIT notice: Copyright (c) 2025 Benhao Huang and Hansen
# Lillemark; spectral-shift convention also credits denzelzheng. License text
# is already preserved in this repository at tests/reference/LICENSE_flowm.
def axis_phase(size, delta, derivative=0, half=False):
    freq = (torch.fft.rfftfreq if half else torch.fft.fftfreq)(size, device=delta.device)
    omega = 2 * math.pi * freq
    angle = omega * delta[..., None]
    basis = torch.polar(torch.ones_like(angle), angle) * (1j * omega)**derivative
    functions = (torch.cos, lambda x: -torch.sin(x), lambda x: -torch.cos(x))
    nyquist = functions[derivative](math.pi * delta[..., None]) * math.pi**derivative
    return torch.where(freq.abs() == 0.5, nyquist.to(basis.dtype), basis)


def refine_peak(numerator, denominator, start):
    """Literal three-step Newton peak refinement, including rejection guards."""
    weights = start.new_full((WORLD // 2 + 1,), 2)
    weights[[0, -1]] = 1
    spectra = torch.stack([numerator, denominator])

    def evaluate(delta):
        py = [axis_phase(WORLD, delta[..., 1], m) for m in range(3)]
        px = [axis_phase(WORLD, delta[..., 0], n, half=True) * weights for n in range(3)]
        rows = [torch.einsum("s...hw,...h->s...w", spectra, p) for p in py]

        def value(m, n):
            return torch.einsum("s...w,...w->s...", rows[m], px[n]).real / WORLD**2

        f, fx, fy = value(0, 0), value(0, 1), value(1, 0)
        fxx, fxy, fyy = value(0, 2), value(1, 1), value(2, 0)
        n, d = f[0], f[1].clamp_min(1e-5)
        nx, ny, dx, dy = fx[0], fy[0], fx[1], fy[1]
        similarity = 2 * n / d
        gradient = torch.stack([2 * (nx*d - n*dx) / d**2, 2 * (ny*d - n*dy) / d**2], -1)

        def second(nab, na, nb, dab, da, db):
            return 2 * (nab/d - (na*db + nb*da)/d**2 - n*dab/d**2 + 2*n*da*db/d**3)

        hxx = second(fxx[0], nx, nx, fxx[1], dx, dx)
        hxy = second(fxy[0], nx, ny, fxy[1], dx, dy)
        hyy = second(fyy[0], ny, ny, fyy[1], dy, dy)
        return similarity, gradient, hxx, hxy, hyy, f[1]

    s0, _, _, _, _, d0 = evaluate(start)
    delta = start
    for _ in range(3):
        _, gradient, xx, xy, yy, _ = evaluate(delta)
        determinant = xx * yy - xy.square()
        safe = determinant.abs() > 1e-6 * ((xx + yy).square() + 1e-12)
        denom = torch.where(safe, determinant, torch.ones_like(determinant))
        sx = -(yy * gradient[..., 0] - xy * gradient[..., 1]) / denom
        sy = -(-xy * gradient[..., 0] + xx * gradient[..., 1]) / denom
        step = torch.stack([sx, sy], -1)
        delta = delta + torch.where(safe[..., None], step, 0).clamp(-0.5, 0.5)
    s1, _, xx, xy, yy, d1 = evaluate(delta)
    trace, determinant = xx + yy, xx * yy - xy.square()
    largest = trace / 2 + ((trace / 2).square() - determinant).sqrt().nan_to_num(0)
    accepted = (trace < 0) & (determinant > 0) & (largest < -1e-3 * trace.abs() / 2)
    accepted &= (delta - start).norm(dim=-1) <= 1
    accepted &= (d0 > 1e-5) & (d1 > 1e-5) & (d1 >= 0.5 * d0)
    accepted &= (s1 <= 1 + 1e-3) & (s1 >= s0)
    return torch.where(accepted[..., None], delta, start), accepted


def correlation_measure(old, current, camera, nu, lattice, lag, nearest=True):
    """pr1_sim.py's window-level negative control, vectorized candidate choice."""
    a, b = pad_world(old), pad_world(current)
    mask = pad_world(torch.ones_like(old[:, 0]))
    fa, fb, fm = torch.fft.rfft2(a), torch.fft.rfft2(b), torch.fft.rfft2(mask)
    numerator = (fa.conj() * fb).sum(1)
    denominator = torch.fft.rfft2(a.square().sum(1)).conj() * fm
    denominator += fm.conj() * torch.fft.rfft2(b.square().sum(1))
    surface = (2 * torch.fft.irfft2(numerator, s=(WORLD, WORLD)) /
               torch.fft.irfft2(denominator, s=(WORLD, WORLD)).clamp_min(1e-5)).clamp(-1, 1)
    neighbors = torch.stack([torch.roll(surface, (i, j), (-2, -1))
                             for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)]).amax(0)
    yy, xx = torch.meshgrid(torch.arange(WORLD, device=nu.device),
                            torch.arange(WORLD, device=nu.device), indexing="ij")
    grid = wrap(torch.stack([xx, yy], -1).float()).reshape(-1, 2)
    residual = wrap(grid[None, None] + camera[:, None, None] - lattice[None, :, None] * lag)
    region = (residual.abs() <= BOUND * lag + 0.5).all(-1)
    candidate = region & (surface >= neighbors).flatten(1)[:, None]
    score = surface.flatten(1)[:, None].expand_as(candidate)
    best = score.masked_fill(~candidate, -torch.inf).amax(-1, keepdim=True)
    if nearest:
        eligible = candidate & (score >= 0.6 * best)
        velocity = lattice[None, :, None] + residual / lag
        distance = (velocity - nu[:, :, None]).square().sum(-1)
        index = distance.masked_fill(~eligible, torch.inf).argmin(-1)
        # pr1_sim uses the first local maximum if every candidate fails the
        # relative-score cutoff (possible when all candidate scores are < 0).
        first_candidate = candidate.to(torch.int32).argmax(-1)
        index = torch.where(eligible.any(-1), index, first_candidate)
    else:
        index = score.masked_fill(~candidate, -torch.inf).argmax(-1)
    valid = candidate.any(-1)
    start = torch.where(valid[..., None], grid[index], 0)
    batch = old.shape[0]
    refined, _ = refine_peak(numerator[:, None].expand(-1, NODES, -1, -1).reshape(-1, WORLD, 26),
                             denominator[:, None].expand(-1, NODES, -1, -1).reshape(-1, WORLD, 26),
                             start.reshape(-1, 2))
    refined = refined.reshape(batch, NODES, 2)
    rounded = refined.round().long() % WORLD
    weight = surface[torch.arange(batch, device=nu.device)[:, None], rounded[..., 1], rounded[..., 0]]
    velocity = lattice[None] + wrap(refined + camera[:, None] - lattice[None] * lag) / lag
    return velocity, weight.clamp_min(0) * valid


@dataclass(frozen=True)
class Spec:
    name: str
    family: str
    lags: tuple = (1,)
    modes: bool = False
    step_clip: float | None = 0.3
    nearest: bool = True


SPECS = [
    Spec("A_LK", "A"), Spec("A_LK_no_clip", "A", step_clip=None),
    Spec("A_LK_snapshot", "S"),
    Spec("B_local", "B"), Spec("B_local_2mode", "B", modes=True),
    Spec("C_lags124", "C", (1, 2, 4)), Spec("C_lags124_2mode", "C", (1, 2, 4), True),
    Spec("C_lags1248", "C", (1, 2, 4, 8)), Spec("C_lags1248_2mode", "C", (1, 2, 4, 8), True),
    Spec("D_lock_LK", "D"), Spec("D_lock_LK_2mode", "D", modes=True),
    Spec("E_window_nearest124", "E", (1, 2, 4), step_clip=None),
    Spec("E_window_strongest124", "E", (1, 2, 4), step_clip=None, nearest=False),
    Spec("E_window_nearest1", "E", step_clip=None),
]


class Estimator:
    def __init__(self, spec, batch, device):
        self.spec = spec
        self.lattice = lattice_nodes().to(device)
        self.nu = self.lattice[None].expand(batch, -1, -1).clone()
        self.evidence = torch.zeros(batch, NODES, device=device)
        self.second = self.nu.clone()
        self.second_evidence = self.evidence.clone()
        self.history = deque(maxlen=None if spec.family == "A" else max(spec.lags))
        self.node_history = []  # only needed for the exact A replay
        self.cumulative_camera = torch.zeros(batch, 2, device=device)
        self.last_search_fraction = 0.0

    def legacy_imprint(self, action):
        """Reconstruct AdaptiveNodeCell's original overwrite/decay recurrence.

        No persistent per-node images. Do not replace sequential spectral shifts
        with a cumulative shift: Nyquist and overwrite masks prevent equivalence.
        """
        m = pad_world(self.history[0][0])[:, None].expand(-1, NODES, -1, -1, -1).clone()
        previous_camera = self.history[0][1]
        for j, (imprint, camera) in enumerate(list(self.history)[1:], 1):
            m = spectral_shift(m, self.node_history[j - 1] - (camera - previous_camera)[:, None])
            m *= 0.9
            m[..., :WIN, :WIN] = imprint[:, None]
            previous_camera = camera
        m = spectral_shift(m, self.nu - action[:, None])
        return m[..., :WIN, :WIN]

    def update(self, delta, evidence):
        self.evidence += evidence
        step = evidence[..., None] / self.evidence[..., None].clamp_min(1e-6) * delta
        if self.spec.step_clip is not None:
            step = step.clamp(-self.spec.step_clip, self.spec.step_clip)
        self.nu = self.lattice[None] + (self.nu - self.lattice[None] + step).clamp(-BOUND, BOUND)

    def step(self, current, action):
        self.cumulative_camera = self.cumulative_camera + action
        spec = self.spec
        if self.history:
            if spec.family == "A":
                m = self.legacy_imprint(action)
                d, trace = lk_displacement(m, current, return_trace=True)
                delta, evidence = pool(d, coherence(m, current, trace, d))
                self.update(delta, evidence)
            elif spec.family == "E":
                velocities, weights, lags = [], [], []
                for lag in spec.lags:
                    if len(self.history) < lag:
                        continue
                    old, camera = self.history[-lag]
                    v, w = correlation_measure(old, current, self.cumulative_camera - camera,
                                               self.nu, self.lattice, lag, spec.nearest)
                    velocities.append(v)
                    weights.append(w)
                    lags.append(lag)
                fit, evidence = robust_velocity(velocities, weights, lags)
                self.update(fit - self.nu, evidence)
            else:
                self.local_step(current)
        self.history.append((current, self.cumulative_camera))
        if spec.family == "A":
            self.node_history.append(self.nu.clone())
        return self.nu

    def local_step(self, current):
        spec = self.spec
        search = None
        velocities, weights, second_v, second_w, lags = [], [], [], [], []
        for lag in spec.lags:
            if len(self.history) < lag:
                continue
            old, camera = self.history[-lag]
            shift = self.nu * lag - (self.cumulative_camera - camera)[:, None]
            m = spectral_shift(pad_world(old)[:, None].expand(-1, NODES, -1, -1, -1), shift)[..., :WIN, :WIN]
            support = shifted_window_mask(shift)
            if spec.family in ("D", "S"):
                lk, trace = lk_displacement(m, current, return_trace=True)
                lw = coherence(m, current, trace, lk, support)
                ld, le = pool(lk, lw)
                # Evidence normalized by textured imprint mass is an agreement
                # score. Re-evaluate every step, so a node can unlock again.
                textured = m.norm(dim=2) * (trace > 0.1 * trace.mean((1, 2, 3), keepdim=True)) * support
                agreement = le / textured.sum((-2, -1)).clamp_min(1e-6)
                locked = (agreement >= 0.1) & (ld.norm(dim=-1) <= 0.3)
                if spec.family == "S":
                    self.update(ld, le)
                    return
            else:
                trace = texture_trace(m)
                locked = None
            if locked is None:
                if search is None:
                    search = LocalSearch(current)
                d, usable = search(m, support)
            else:
                # D actually skips the volume at locked nodes; the diagnostic
                # second mode there uses the LK displacement field.
                d, usable = lk.clone(), torch.ones_like(support, dtype=torch.bool)
                bi, vi = (~locked).nonzero(as_tuple=True)
                if len(bi):
                    if search is None:
                        search = LocalSearch(current)
                    selected, valid = search(m[bi, vi][:, None], support[bi, vi][:, None], bi)
                    d[bi, vi], usable[bi, vi] = selected[:, 0], valid[:, 0]
            w = coherence(m, current, trace, d, support * usable, lag)
            if spec.modes:
                delta, evidence, sv, sw = two_modes(d, w, self.nu, self.lattice, lag)
                second_v.append(sv)
                second_w.append(sw)
            else:
                delta, evidence = pool(d, w)
            if locked is not None:
                # The mode variant clusters whichever field is active: LK
                # at locked nodes, local SSD at unlocked nodes. Never replace
                # its selected primary mode with the unclustered LK average.
                if not spec.modes:
                    delta = torch.where(locked[..., None], ld, delta)
                    evidence = torch.where(locked, le, evidence)
                self.last_search_fraction = float((~locked).float().mean())
            velocities.append(self.nu + delta / lag)
            weights.append(evidence)
            lags.append(lag)
        if spec.family == "C":
            fit, evidence = robust_velocity(velocities, weights, lags)
        else:
            fit, evidence = velocities[0], weights[0]
        if spec.modes:
            sv, sw = robust_velocity(second_v, second_w, lags)
            self.second_evidence += sw
            gain = sw / self.second_evidence.clamp_min(1e-6)
            self.second += gain[..., None] * (sv - self.second)
        self.update(fit - self.nu, evidence)


def metrics(trajectory, truth, visible, secondary=None, secondary_valid=None):
    """pr1_sim.py errors, plus explicitly censored locks and same-cell pairs."""
    trajectory, truth, visible = trajectory.cpu(), truth.cpu(), visible.cpu()
    n, steps = trajectory.shape[:2]
    considered = visible[:, -10:].any(1)
    distance = (truth[:, -1, :, None] - trajectory[:, -1, None]).norm(dim=-1)
    closest = distance.amin(-1)
    matched = torch.empty(n, 5)
    assignments = torch.empty(n, 5, dtype=torch.long)
    for b in range(n):
        rows, columns = linear_sum_assignment(distance[b].numpy())
        matched[b, rows] = distance[b, rows, columns]
        assignments[b, rows] = torch.as_tensor(columns)
    distances = (truth[:, :, :, None] - trajectory[:, :, None]).norm(dim=-1).amin(-1)
    hit = (distances <= THRESHOLD) & visible
    times = torch.arange(steps)[None, :, None].expand_as(hit)
    lock = torch.where(hit, times, steps).amin(1)
    lattice = lattice_nodes()
    cells = (truth[:, -1, :, None] - lattice[None, None]).square().sum(-1).argmin(-1)
    pairs, both, mode_pairs, sequences = 0, 0, 0, 0
    for b in range(n):
        count = 0
        for i in range(5):
            for j in range(i + 1, 5):
                if not (considered[b, i] and considered[b, j] and cells[b, i] == cells[b, j]):
                    continue
                pairs += 1
                count += 1
                both += int(matched[b, i] <= THRESHOLD and matched[b, j] <= THRESHOLD)
                if secondary is not None:
                    cell = cells[b, i]
                    if secondary_valid[b, cell]:
                        candidates = torch.stack([trajectory[b, -1, cell], secondary[b, cell]])
                        d = torch.cdist(truth[b, -1, [i, j]], candidates)
                        mode_pairs += int(((d[0, 0] <= THRESHOLD) & (d[1, 1] <= THRESHOLD)) |
                                          ((d[0, 1] <= THRESHOLD) & (d[1, 0] <= THRESHOLD)))
        sequences += count > 0
    vis_closest, vis_matched, vis_lock = closest[considered], matched[considered], lock[considered]
    result = dict(n=n, visible_digits=int(considered.sum()),
                  closest_median=float(vis_closest.median()), matched_median=float(vis_matched.median()),
                  matched_gt_0_1=float((vis_matched > THRESHOLD).float().mean()),
                  lock_median_censored=float(vis_lock.float().median()),
                  never_locked=float((vis_lock == steps).float().mean()),
                  same_cell_pairs=pairs, same_cell_sequences=sequences,
                  same_cell_both_resolved=both / pairs if pairs else None,
                  own_cell_two_modes_resolved=mode_pairs / pairs if pairs and secondary is not None else None)
    details = dict(final_nodes=trajectory[:, -1].tolist(), closest_error=closest.tolist(),
                   matched_error=matched.tolist(), assignment=assignments.tolist(), visible=considered.tolist(),
                   first_lock=lock.tolist())
    if secondary is not None:
        details.update(second_mode=secondary.tolist(), second_mode_valid=secondary_valid.tolist())
    return result, details


@torch.inference_mode()
def run_estimator(spec, frames, actions, cell, batch_size, device, max_seconds=None):
    trajectories, seconds, second_valid = [], [], []
    elapsed = 0.0
    peak = 0
    for first in range(0, len(frames), batch_size):
        f, a = frames[first:first + batch_size].to(device), actions[first:first + batch_size].to(device)
        estimator = Estimator(spec, len(f), device)
        trajectory = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sync(device)
        start = time.perf_counter()
        for t in range(f.shape[1]):
            current = cell.imprint(cell.conv_u(f[:, t])).float()
            trajectory.append(estimator.step(current, a[:, t]).clone())
        sync(device)
        elapsed += time.perf_counter() - start
        trajectories.append(torch.stack(trajectory, 1).cpu())
        seconds.append(estimator.second.cpu())
        second_valid.append((estimator.second_evidence > 1e-6).cpu())
        if device.type == "cuda":
            peak = max(peak, torch.cuda.max_memory_reserved(device))
        print(f"  {spec.name}: {min(first + batch_size, len(frames))}/{len(frames)}", file=sys.stderr, flush=True)
        if max_seconds is not None and elapsed >= max_seconds:
            print(f"  {spec.name}: evaluation time budget reached; retaining completed batches.",
                  file=sys.stderr, flush=True)
            break
    return torch.cat(trajectories), torch.cat(seconds), torch.cat(second_valid), elapsed, peak


@torch.inference_mode()
def benchmark(spec, frames, actions, cell, device, repeats):
    """Steady-state B=16 wall clock; includes encoding, snapshot work and update.

    Warm up through frame 8, then time successive frames (9..). No environment
    rendering, metric calculation, device transfer or feature-memory recurrence.
    """
    indices = torch.arange(16) % len(frames)
    f, a = frames[indices].to(device), actions[indices].to(device)
    estimator = Estimator(spec, 16, device)
    samples = []
    for t in range(9 + repeats):
        sync(device)
        start = time.perf_counter()
        estimator.step(cell.imprint(cell.conv_u(f[:, t])).float(), a[:, t])
        sync(device)
        if t >= 9:
            samples.append(time.perf_counter() - start)
    return dict(batch=16, warmup_steps=9, measured_steps=repeats,
                seconds_per_step=float(np.median(samples)), samples_seconds=samples)


def format_table(rows):
    headers = ["estimator", "n", "closest", "matched", ">.1", "lock*", "never", "pair", "2mode", "s/step@16"]
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for row in rows:
        m = row["metrics"]
        fraction = lambda value: "—" if value is None else f"{100 * value:.1f}%"
        timing = row.get("timing", {}).get("seconds_per_step")
        lines.append(" | ".join([row["name"], str(m["n"]), f"{m['closest_median']:.4f}",
                                 f"{m['matched_median']:.4f}", fraction(m["matched_gt_0_1"]),
                                 f"{m['lock_median_censored']:.0f}", fraction(m["never_locked"]),
                                 fraction(m["same_cell_both_resolved"]),
                                 fraction(m["own_cell_two_modes_resolved"]),
                                 "—" if timing is None else f"{timing:.4f}"]))
    return "\n".join(lines)


def paired_bootstrap(row, reference, repeats=2000):
    """Descriptive, unadjusted intervals; resample whole sequences in pairs."""
    n = row["metrics"]["n"]
    a = np.asarray(reference["details"]["matched_error"][:n], dtype=np.float32)
    b = np.asarray(row["details"]["matched_error"], dtype=np.float32)
    visible = np.asarray(row["details"]["visible"], dtype=bool)
    assert a.shape == b.shape
    rng = np.random.default_rng(2024)
    medians, failures = [], []
    for _ in range(repeats):
        indices = rng.integers(0, n, n)
        mask = visible[indices]
        aa, bb = a[indices][mask], b[indices][mask]
        middle = (len(aa) - 1) // 2  # lower median, exactly as torch.median
        medians.append(float(np.partition(aa, middle)[middle] - np.partition(bb, middle)[middle]))
        failures.append(float((aa > 0.1).mean() - (bb > 0.1).mean()))
    return dict(n=n, replicates=repeats, unit="sequence", seed=2024,
                matched_median_reduction_ci95=np.quantile(medians, [0.025, 0.975]).tolist(),
                failure_fraction_reduction_ci95=np.quantile(failures, [0.025, 0.975]).tolist(),
                note="Descriptive paired bootstrap; not adjusted for selecting among estimators.")


def save_result(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@torch.inference_mode()
def self_check(cell):
    """Small CPU checks of signs, camera/lag units, reference parity and modes."""
    device = next(cell.parameters()).device
    batch = make_batch(2, 6, 0, EnvConfig(real_velocity=True, real_camera=True),
                       gen=torch.Generator().manual_seed(93), split="test")
    f, a = batch["frames"].to(device), batch["actions"].to(device)
    for spec in SPECS[:2]:
        estimator = Estimator(spec, 2, device)
        cell.step_clip = 0.3 if spec.step_clip is not None else math.inf
        h = torch.zeros(2, 25, 43, WORLD, WORLD, device=device)
        for t in range(6):
            h = cell(f[:, t], h, a[:, t], estimator.lattice, 1.0)
            nu = estimator.step(cell.imprint(cell.conv_u(f[:, t])), a[:, t])
            torch.testing.assert_close(nu, cell.last_nodes, atol=3e-5, rtol=1e-5)
    cell.step_clip = 0.3
    yy, xx = torch.meshgrid(torch.arange(WORLD, device=device), torch.arange(WORLD, device=device), indexing="ij")
    blob = torch.exp(-((xx - 15).square() + (yy - 15).square()) / 12)[None, None]
    velocity = torch.tensor([0.35, -0.22], device=device)
    camera = torch.tensor([0.2, -0.4], device=device)
    for lag in (1, 4, 8):
        nu = velocity + velocity.new_tensor([-0.04, 0.03])
        current = spectral_shift(blob, velocity * lag - camera)[..., :WIN, :WIN]
        m = spectral_shift(blob, nu * lag - camera)[..., :WIN, :WIN][:, None]
        d, usable = LocalSearch(current)(m, torch.ones(1, 1, WIN, WIN, device=device))
        weight = m.norm(dim=2) * usable * (m.norm(dim=2) > 0.1)
        pooled, _ = pool(d, weight)
        torch.testing.assert_close(nu + pooled[0, 0] / lag, velocity, atol=0.035, rtol=0)
    v = [velocity[None, None].expand(1, 1, 2)] * 3
    fit, _ = robust_velocity(v, [torch.ones(1, 1, device=device)] * 3, [1, 2, 4])
    torch.testing.assert_close(fit[0, 0], velocity)
    d = torch.zeros(1, 1, 2, 8, 8, device=device)
    d[:, :, 0, :4] = -0.2
    d[:, :, 0, 4:] = 0.4
    primary, _, secondary, evidence = two_modes(d, torch.ones(1, 1, 8, 8, device=device),
                                                torch.zeros(1, 1, 2, device=device),
                                                torch.zeros(1, 2, device=device), 1)
    torch.testing.assert_close(primary[0, 0, 0], d.new_tensor(-0.2))
    torch.testing.assert_close(secondary[0, 0, 0], d.new_tensor(0.4))
    assert evidence.item() > 0
    # Regression: locked D must pool its selected LK mode, not silently put
    # back the unclustered LK mean (+.1 in this deliberately bimodal field).
    from unittest.mock import patch
    mode_spec = next(s for s in SPECS if s.name == "D_lock_LK_2mode")
    estimator = Estimator(mode_spec, 1, device)
    ones = torch.ones(1, CHANNELS, WIN, WIN, device=device)
    zero_action = torch.zeros(1, 2, device=device)
    estimator.step(ones, zero_action)
    field = torch.zeros(1, NODES, 2, WIN, WIN, device=device)
    field[:, :, 0, :WIN // 2] = -0.1
    field[:, :, 0, WIN // 2:] = 0.3
    trace = torch.ones(1, NODES, WIN, WIN, device=device)
    with patch(__name__ + ".lk_displacement", return_value=(field, trace)):
        estimator.step(ones, zero_action)
    torch.testing.assert_close(estimator.nu[0, 12, 0], field.new_tensor(-0.1), atol=1e-5, rtol=0)
    torch.testing.assert_close(estimator.second[0, 12, 0], field.new_tensor(0.3), atol=1e-5, rtol=0)
    print("Self-checks passed: exact LK parity, displacement/camera signs, lag units, WLS, two modes, locked-mode routing.",
          file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--estimators", default="A,B,C,D,E", help="comma-separated families or exact row names")
    parser.add_argument("--out", type=Path, default=Path("results/gate_estimator.json"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--slow-n", type=int, help="evaluate B/C/D on this common prefix, still generate --n sequences")
    parser.add_argument("--bench-steps", type=int, default=3)
    parser.add_argument("--max-eval-seconds", type=float, help="per-estimator CPU/GPU evaluation budget; finish current batch")
    parser.add_argument("--skip-benchmark", action="store_true", help="development only")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.n < 1 or not 1 <= args.batch_size <= 32 or not 1 <= args.bench_steps <= 21:
        parser.error("n must be positive, batch size 1..32, bench steps 1..21")
    if args.slow_n is not None and not 1 <= args.slow_n <= args.n:
        parser.error("slow-n must be in 1..n")
    if args.max_eval_seconds is not None and args.max_eval_seconds <= 0:
        parser.error("max-eval-seconds must be positive")
    chosen = {x.strip() for x in args.estimators.split(",")}
    known = {"A", "B", "C", "D", "E"} | {s.name for s in SPECS}
    if chosen - known:
        parser.error(f"unknown estimators: {sorted(chosen - known)}")
    specs = [s for s in SPECS if s.name in chosen or s.family in chosen or (s.family == "S" and "A" in chosen)]
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA unavailable; use --device cpu. Only physical GPU 1 is allowed.")
        # Includes allocator overhead; transient volume chunks keep usage much lower.
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(0.9, 4.5 * 1024**3 / total), device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    cell = AdaptiveNodeCell(C=32, num_v=25).eval().to(device)
    for parameter in cell.parameters():
        parameter.requires_grad_(False)
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        if args.self_check:
            self_check(cell)
        # One CPU generation call gives exactly make_batch(n,...,seed=2024),
        # independent of processing batch size, slow-n and estimator selection.
        batch = make_batch(args.n, 30, 0, EnvConfig(real_velocity=True, real_camera=True),
                           gen=torch.Generator().manual_seed(2024), split="test", device="cpu")
        frames, actions = batch["frames"], batch["actions"]
        truth, visible = batch["digit_states"]["vel"], batch["digit_states"]["visible"]
        digest = hashlib.sha256(frames.numpy().tobytes() + actions.numpy().tobytes()).hexdigest()
        payload = dict(schema_version=1, complete=False,
                       requested_estimators=[s.name for s in specs], completed_estimators=[],
                       config=vars(args) | {"out": str(args.out)},
                       environment=batch["cfg"], seed=2024, cell_seed=0, observed_frames=30,
                       imprint="untrained AdaptiveNodeCell(C=32,num_v=25), K=8, fixed P, s=1",
                       dtype="float32; autocast and TF32 disabled", dataset_sha256=digest,
                       torch_version=torch.__version__, device=str(device),
                       source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       cuda_device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                       rows=[])
        lattice = lattice_nodes()
        fixed = lattice[None, None].expand(args.n, 30, -1, -1)
        # Exactly 25 nodes: five labelled true velocities plus first 20 vx-major lattice nodes.
        oracle = torch.cat([truth, lattice[:20][None, None].expand(args.n, 30, -1, -1)], 2)
        for name, trajectory in (("fixed_lattice", fixed), ("oracle_25", oracle)):
            m, details = metrics(trajectory, truth, visible)
            payload["rows"].append(dict(name=name, metrics=m, details=details))
        save_result(args.out, payload)
        for spec in specs:
            n = min(args.n, args.slow_n or args.n) if spec.family in ("B", "C", "D") else args.n
            requested_n = n
            trajectory, secondary, valid, elapsed, peak = run_estimator(
                spec, frames[:n], actions[:n], cell, args.batch_size, device, args.max_eval_seconds)
            n = len(trajectory)
            m, details = metrics(trajectory, truth[:n], visible[:n],
                                  secondary if spec.modes else None, valid if spec.modes else None)
            timing = {} if args.skip_benchmark else benchmark(spec, frames, actions, cell, device, args.bench_steps)
            if device.type == "cuda":
                peak = max(peak, torch.cuda.max_memory_reserved(device))
            row = dict(name=spec.name, spec=asdict(spec), metrics=m, details=details, timing=timing,
                       requested_n=requested_n, evaluation_seconds=elapsed, peak_cuda_reserved_bytes=peak)
            payload["rows"].append(row)
            payload["completed_estimators"].append(spec.name)
            if args.slow_n and n > args.slow_n:
                pm, pd = metrics(trajectory[:args.slow_n], truth[:args.slow_n], visible[:args.slow_n])
                payload["rows"].append(dict(name=spec.name + "_paired_prefix", metrics=pm, details=pd, timing=timing))
            payload["elapsed_seconds"] = time.perf_counter() - start
            save_result(args.out, payload)
        payload["table"] = format_table(payload["rows"])
        payload["complete"] = True
        reference = next((r for r in payload["rows"] if r["name"] == "A_LK"), None)
        if reference is not None:
            payload["paired_bootstrap_vs_A_LK"] = {
                r["name"]: paired_bootstrap(r, reference) for r in payload["rows"]
                if "spec" in r and r is not reference and r["metrics"]["n"] <= reference["metrics"]["n"]}
        payload["elapsed_seconds"] = time.perf_counter() - start
        save_result(args.out, payload)
        print(payload["table"])
        print(f"\n* Lock step is zero-based; never locked is censored to 30. JSON: {args.out}")


if __name__ == "__main__":
    main()
