"""PR2 CPU gate using the production estimator, frozen random encoder, no learning.

Invoked through estimator_gate.py --accel. No copied measurement/fit rules.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from kwm.cells import AdaptiveNodeCell
from kwm.collision import lattice_nodes
from kwm.env import EnvConfig, make_batch, load_mnist, render_digits
from kwm.lock_nodes import (NODE_DIM, CORR, MT, TIME, acceleration_fit, transport_velocity)
from scripts.assignment import linear_sum_assignment


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def encoder(**kw):
    torch.manual_seed(0)
    cell = AdaptiveNodeCell(C=32, world=50, window=32, node_update='lock', **kw).eval()
    for p in cell.parameters():
        p.requires_grad_(False)
    return cell


@torch.inference_mode()
def run(batch, cell, bs=2):
    """All fitted states; feature-memory evolution cannot affect the encoder."""
    states = []
    lattice = lattice_nodes()
    start = time.perf_counter()
    for first in range(0, len(batch['frames']), bs):
        f, actions = batch['frames'][first:first + bs], batch['actions'][first:first + bs]
        h = cell.initial_state(f[:, 0])
        state, previous = h.nodes, h.snapshot
        history = []
        for t in range(f.shape[1]):
            current = cell.imprint(cell.conv_u(f[:, t])).float()
            state = cell.lock_update(previous, current, state, actions[:, t], lattice)
            previous = current
            history.append(state.clone())
        states.append(torch.stack(history, 1))
        if (first + bs) % 10 == 0:
            print(f'order={cell.order} F={cell.nu_forget}: {first + bs}/{len(batch["frames"])}; '
                  f'{time.perf_counter() - start:.1f}s', flush=True)
    return torch.cat(states), time.perf_counter() - start


def null_table(states):
    # The requested node-step denominator: nodes with cumulative evidence at
    # frame 29, across all 30 observation steps. Also expose the endpoint rate.
    active = states[:, -1, :, 2] > 1e-6
    eligible = active[:, None].expand(states.shape[:3])
    rows = []
    for inflation in (1., 2., 4.):
        for kappa in (4., 9.):
            _, gate, _, _ = acceleration_fit(states, kappa, inflation)
            rows.append(dict(inflation=inflation, kappa=kappa,
                             open_node_steps=int(((gate > 0) & eligible).sum()),
                             node_steps=int(eligible.sum()),
                             null_rate=float((gate[eligible] > 0).float().mean()),
                             null_rate_at30=float((gate[:, -1][active] > 0).float().mean())))
    return rows


def corrected_final(state, inflation, kappa=9., dt=1.):
    state = state.clone()
    slope, gate, var, neff = acceleration_fit(state, kappa, inflation)
    now = state[..., TIME] - dt
    state[..., CORR] = gate[..., None] * slope * (now - state[..., MT])[..., None]
    velocity = transport_velocity(state, lattice_nodes(), dt, kappa, inflation)
    return velocity, slope, gate, var, neff


def matches(velocity, truth, visible):
    distance = (truth[:, :, None] - velocity[:, None]).norm(dim=-1)
    assignments = torch.stack([torch.as_tensor(linear_sum_assignment(x.numpy())[1]) for x in distance])
    errors = distance.gather(2, assignments[..., None])[..., 0]
    selected = errors[visible]
    return dict(median=float(selected.median()), mean=float(selected.mean()), n=int(visible.sum()),
                errors=errors.tolist(), assignments=assignments.tolist()), assignments


def synthetic_scene(seed=777, accel=.004):
    """One known accelerating MNIST digit, real moving camera; same renderer."""
    cfg = EnvConfig(n_digits=1, real_velocity=True, real_camera=True, accel=accel)
    gen = torch.Generator().manual_seed(seed)
    digits, _, _ = load_mnist('test')
    patch = digits[torch.randint(len(digits), (1, 1), generator=gen)]
    acceleration = torch.tensor([[[accel, 0.]]])
    velocity = torch.tensor([[[.2, -.3]]])
    t = torch.arange(30).float()[None, :, None, None]
    pos = torch.tensor([[[8., 8.]]])[:, None] + velocity[:, None] * t + .5 * acceleration[:, None] * t.square()
    # Keep the single digit observed: a real camera tracks v0 plus small
    # independent uniform jitter. This is a controlled identifiability check,
    # not another draw from the five-digit K-accel camera distribution.
    actions = velocity[:, 0, None] + torch.rand(1, 30, 2, generator=gen) * .5 - .25
    actions[:, 0] = 0
    cam = (torch.tensor([[[6., 6.]]]) + actions.cumsum(1)).remainder(50)
    frames = render_digits(patch, pos.remainder(50), cam, cfg)[0]
    return dict(frames=frames, actions=actions, digit_states=dict(acc=acceleration,
                vel=velocity[:, None] + acceleration[:, None] * (t + .5))), cfg


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--accel', type=float, required=True)
    p.add_argument('--accel_zero_p', type=float, default=.3)
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--seed', type=int, default=777)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--out', type=Path, default=Path('results/order2_gate'))
    p.add_argument('--reuse-null', action='store_true', help='reuse the completed K-real calibration only')
    p.add_argument('--reuse', action='store_true', help='reuse completed null/detection JSON; rerun the synthetic check')
    args = p.parse_args(argv)
    torch.set_num_threads(1)
    start = time.perf_counter()
    cfg = EnvConfig(real_velocity=True, real_camera=True, accel=args.accel, accel_zero_p=args.accel_zero_p)
    common = dict(n=args.n, seed=args.seed, batch_size=args.batch_size, cell_seed=0, C=32, K=8,
                  observed_frames=30, dtype='float32', device='cpu', environment=cfg.to_dict())
    if args.reuse:
        calibration = json.loads((args.out / 'null.json').read_text())
        detection = json.loads((args.out / 'detection.json').read_text())
        for saved in (calibration, detection):
            for key in common:
                if saved[key] != common[key]:
                    raise ValueError(f"reused result differs in {key}")
        rows = calibration['rows']
        inflation = calibration['chosen_inflation']
        candidates = [inflation] if calibration['passed'] else []
        bins, metrics = detection['bins'], detection['metrics']
    else:
        if args.reuse_null:
            calibration = json.loads((args.out / 'null.json').read_text())
            for key in common:
                if calibration[key] != common[key]:
                    raise ValueError(f"reused calibration differs in {key}")
            rows = calibration['rows']
            inflation = calibration['chosen_inflation']
            candidates = [inflation] if calibration['passed'] else []
            calibration['dataset_environment'] = EnvConfig(real_velocity=True, real_camera=True).to_dict()
            save(args.out / 'null.json', calibration)
        else:
            null = make_batch(args.n, 30, 0, EnvConfig(real_velocity=True, real_camera=True),
                              gen=torch.Generator().manual_seed(args.seed), split='test')
            cell = encoder(order=2)
            states, seconds = run(null, cell, args.batch_size)
            rows = null_table(states)
            candidates = [row['inflation'] for row in rows if row['kappa'] == 9 and row['null_rate'] < .02]
            inflation = min(candidates) if candidates else 4.
            calibration = dict(**common, rows=rows, chosen_inflation=inflation, passed=bool(candidates), seconds=seconds,
                               dataset_environment=EnvConfig(real_velocity=True, real_camera=True).to_dict(),
                               dataset_sha256=hashlib.sha256(null['frames'].numpy().tobytes()).hexdigest())
            save(args.out / 'null.json', calibration)
            print('NULL:', json.dumps(rows), flush=True)
        cell = encoder(order=2)
        accel = make_batch(args.n, 30, 0, cfg, gen=torch.Generator().manual_seed(args.seed), split='test')
        states, seconds = run(accel, cell, args.batch_size)
        final = states[:, -1]
        truth = accel['digit_states']['vel'][:, -1]
        visible = accel['digit_states']['visible'][:, -10:].any(1)
        velocity, slope, gate, _, _ = corrected_final(final, inflation)
        order2, assignment = matches(velocity, truth, visible)
        order1, _ = matches(lattice_nodes()[None] + final[..., :2], truth, visible)
        metrics = dict(order1=order1, order2=order2)
        bins = []
        mag = accel['digit_states']['acc'].norm(dim=-1)
        matched_gate = gate.gather(1, assignment)
        for lo, hi in ((0., .0000001), (cfg.accel_min, .004), (.004, .008), (.008, cfg.accel + 1e-8)):
            mask = (mag >= lo) & (mag < hi) & visible
            bins.append(dict(lo=lo, hi=hi, n=int(mask.sum()),
                             gate_open=float((matched_gate[mask] > 0).float().mean()) if mask.any() else None))
        for forget in (.9, .95):
            fs, elapsed = run(accel, encoder(order=1, nu_forget=forget), args.batch_size)
            metrics[f'order1_F{forget}'], _ = matches(lattice_nodes()[None] + fs[:, -1, :, :2], truth, visible)
            seconds += elapsed
        detection = dict(**common, inflation=inflation, kappa=9., bins=bins, metrics=metrics, seconds=seconds,
                         dataset_sha256=hashlib.sha256(accel['frames'].numpy().tobytes()).hexdigest())
        save(args.out / 'detection.json', detection)
    cell = encoder(order=2)
    scene, scene_cfg = synthetic_scene(args.seed)
    ss, seconds = run(scene, cell, args.batch_size)
    vv, aa, gg, _, _ = corrected_final(ss[:, -1], inflation)
    sm, assigned = matches(vv, scene['digit_states']['vel'][:, -1], torch.ones(1, 1, dtype=torch.bool))
    idx = int(assigned[0, 0])
    synthetic = dict(seed=args.seed, config=scene_cfg.to_dict(), inflation=inflation, kappa=9.,
                     camera='v0 + U[-.25,.25]^2 per step; digit remains visible',
                     node=idx, velocity_error=sm['median'], acceleration=aa[0, idx].tolist(),
                     acceleration_error=float((aa[0, idx] - scene['digit_states']['acc'][0, 0]).norm()),
                     gate=float(gg[0, idx]), seconds=seconds)
    save(args.out / 'synthetic.json', synthetic)
    elapsed = time.perf_counter() - start
    if args.reuse:
        elapsed += calibration["seconds"] + detection["seconds"]
    elif args.reuse_null:
        elapsed += calibration["seconds"]
    lines = ['# Order-2 estimator gate', '',
             f'{args.n} sequences, seed {args.seed}, batches of {args.batch_size}; frozen random C=32 encoder (seed 0), K=8.',
             'CPU, fp32, MNIST test images; Hungarian assignment before visibility filtering.', '',
             'Null denominator: all 30 node-steps for nodes with cumulative evidence at frame 29.', '',
             '| Inflation | κ | Null node-step rate | At frame 29 |', '| --- | --- | --- | --- |']
    for row in rows:
        lines.append(f"| {row['inflation']:g} | {row['kappa']:g} | {row['null_rate']:.3%} | {row['null_rate_at30']:.3%} |")
    lines += ['', f'Chosen inflation: {inflation:g}; calibration passed: {bool(candidates)}.', '',
              '| Estimator | Matched median velocity error | Mean |', '| --- | --- | --- |']
    for name, row in metrics.items():
        lines.append(f"| {name} | {row['median']:.6f} | {row['mean']:.6f} |")
    lines += ['', 'Velocity truth is the mean over [29,30], v(29.5); the instantaneous v(29) differs by a/2.', '',
              '| Acceleration magnitude bin | Visible digits | Gate open |', '| --- | --- | --- |']
    for row in bins:
        rate = f"{row['gate_open']:.3%}" if row['gate_open'] is not None else 'NA'
        lines.append(f"| [{row['lo']:g}, {row['hi']:g}) | {row['n']} | {rate} |")
    lines += ['', f"Single-digit a=(.004,0): velocity error {synthetic['velocity_error']:.6f}; "
              f"acceleration error {synthetic['acceleration_error']:.6f}; gate {synthetic['gate']:.6f}.", '',
              'Synthetic camera: v0 plus U[-.25,.25]² per step, keeping the digit visible.', '',
              f'Total elapsed CPU wall time: {elapsed:.2f} s (one thread).', '',
              'Fit/gate measurements are independent of feature-memory transport. The order-1 and order-2 arms',
              'share identical PR1 measurements; the two forgetting arms rerun the production estimator.',
              'Null fits are reused across gate settings, since the gate never feeds back into the measurement.',
              'This is an estimator diagnostic, not evidence of trained forecasting improvement.', '']
    Path('docs/GATE_order2.md').write_text('\n'.join(lines))
    save(args.out / 'summary.json', dict(**common, inflation=inflation, calibration_passed=bool(candidates),
                                        elapsed_seconds=elapsed, synthetic=synthetic))
    print('\n'.join(lines), flush=True)
