#!/usr/bin/env python3
"""Realised joint acceleration law and perfect order-1 rendering ceilings (CPU)."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from kwm.env import EnvConfig, sample_acceleration, load_mnist, render_digits


def law_stats(cfg, n=400000, seed=0):
    v, a, stats = sample_acceleration(torch.Generator().manual_seed(seed), (n,), cfg, return_stats=True)
    mag = a.norm(dim=-1)
    nz = mag > 0
    stats.update(n=n, seed=seed, zero_fraction=float((~nz).float().mean()),
                 nonzero_quantiles=dict(zip(('10', '50', '90', '99'),
                     torch.quantile(mag[nz], torch.tensor([.1, .5, .9, .99])).tolist())) if nz.any() else {})
    stats['bins'] = []
    for lo, hi in ((cfg.accel_min, .004), (.004, .008), (.008, cfg.accel)):
        mask = (mag >= lo) & (mag < hi)
        corr = float(torch.corrcoef(torch.stack([a[mask].flatten(), v[mask].flatten()]))[0, 1]) if mask.sum() > 1 else None
        stats['bins'].append(dict(lo=lo, hi=hi, count=int(mask.sum()), corr_a_v0=corr))
    stats['median_displacement'] = {str(tau): dict(all=float(torch.quantile(.5 * mag * tau**2, .5)),
        nonzero=float(torch.quantile(.5 * mag[nz] * tau**2, .5)) if nz.any() else None) for tau in (20, 50, 150)}
    return stats


@torch.no_grad()
def ceiling(cfg, n=300, seed=1):
    """MSE averaged over future frames 1..20 / 1..50, anchored at frame 29.

    'last_finite_difference' follows the planning reference's forward difference
    p(30)-p(29). The actually observable backward difference is also reported.
    The reference's 'still' was a zero image; report both zero and last frame.
    """
    gen = torch.Generator().manual_seed(seed)
    v, a = sample_acceleration(gen, (n, cfg.n_digits), cfg)
    digits, _, _ = load_mnist('test')
    patch = digits[torch.randint(len(digits), (n, cfg.n_digits), generator=gen)]
    p0 = torch.rand(n, cfg.n_digits, 2, generator=gen) * cfg.world
    cam = (torch.rand(n, 80, 2, generator=gen) * 20 - 10).cumsum(1)
    tau = torch.arange(1, 51).float()[None, :, None, None]
    last = p0 + v * 29 + .5 * a * 29**2
    velocities = dict(mid_observation=v + a * 14.5, instantaneous=v + a * 29,
                      last_finite_difference=v + a * 29.5, backward_difference=v + a * 28.5)
    errors = {k: [] for k in (*velocities, 'still_frame', 'zero_frame')}
    for start in range(0, n, 10):
        sl = slice(start, start + 10)
        t = 29 + tau
        truth = render_digits(patch[sl], p0[sl, None] + v[sl, None] * t + .5 * a[sl, None] * t.square(), cam[sl, 30:], cfg)[0]
        for name, vv in velocities.items():
            pred = render_digits(patch[sl], last[sl, None] + vv[sl, None] * tau, cam[sl, 30:], cfg)[0]
            errors[name].append((pred - truth).square().mean((2, 3, 4)))
        still = render_digits(patch[sl], last[sl, None], cam[sl, 29:30], cfg)[0]
        errors['still_frame'].append((still - truth).square().mean((2, 3, 4)))
        errors['zero_frame'].append(truth.square().mean((2, 3, 4)))
    return dict(n=n, digits=cfg.n_digits, seed=seed, mnist_split='test',
                zero_fraction=float((a.norm(dim=-1) == 0).float().mean()),
                rows={name: {key: value for h in (20, 50) for key, value in ((f'mse{h}', float(torch.cat(parts)[:, :h].mean())), (f'se{h}', float(torch.cat(parts)[:, :h].mean(1).std() / n**.5)))}
                      for name, parts in errors.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--accel', type=float, required=True)
    p.add_argument('--accel_zero_p', type=float, default=.3)
    p.add_argument('--accel_min', type=float, default=.002)
    p.add_argument('--accel_horizon', type=float, default=180.)
    p.add_argument('--n', type=int, default=400000)
    p.add_argument('--ceiling', action='store_true')
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    cfg = EnvConfig(real_velocity=True, real_camera=True, **{k: getattr(args, k) for k in
                    ('accel', 'accel_zero_p', 'accel_min', 'accel_horizon')})
    result = dict(environment=cfg.to_dict(), law=law_stats(cfg, args.n))
    if args.ceiling:
        result['ceiling'] = ceiling(cfg)
    output = json.dumps(result, indent=2, allow_nan=False)
    print(output)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + '\n')


if __name__ == '__main__':
    main()
