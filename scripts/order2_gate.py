"""CPU gate: production estimator, frozen encoder, matched displacement errors.

Run through estimator_gate.py --accel. Fits are shared across kappa and order
arms because neither the gate nor feature transport changes measured velocities.
Only forgetting changes the measurement search centre and requires another run.
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
from kwm.lock_nodes import (CORR, TIME, acceleration_fit, transport_velocity,
                            fit_correction, clamp_mask)
from scripts.assignment import linear_sum_assignment

FORGET = (.6, .7, .8, .9, .95)
KAPPAS = (4., 9.)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def encoder(fast=False, **kw):
    torch.manual_seed(0)
    cell = AdaptiveNodeCell(C=2 if fast else 32, imprint_k=2 if fast else 8,
                            world=32 if fast else 50, window=24 if fast else 32,
                            node_update='lock', **kw).eval()
    for p in cell.parameters():
        p.requires_grad_(False)
    return cell


@torch.inference_mode()
def run(batch, cell, bs=2):
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
        if (first + bs) % 20 == 0:
            print(f'C={cell.C} order={cell.order} F={cell.nu_forget}: '
                  f'{first + bs}/{len(batch["frames"])}; {time.perf_counter() - start:.1f}s', flush=True)
    return torch.cat(states), time.perf_counter() - start


def distribution(values):
    values = values.float()
    if not values.numel():
        return dict(n=0)
    q = values.quantile(torch.tensor([.1, .25, .5, .75, .9]))
    return dict(n=values.numel(), **dict(zip(('p10', 'p25', 'median', 'p75', 'p90'), q.tolist())),
                mean=float(values.mean()), fraction_ge4=float((values >= 4).float().mean()))


def null_table(states):
    # Count only evidence-bearing nodes at that step; also expose the
    # gate-eligible denominator to make the n_eff floor's effect visible.
    active = states[..., 2] > 1e-6
    rows = []
    for kappa in KAPPAS:
        _, gate, _, neff = acceleration_fit(states, kappa)
        eligible = active & (neff >= 4)
        rows.append(dict(kappa=kappa, open_node_steps=int(((gate > 0) & active).sum()),
                         node_steps=int(active.sum()), eligible_node_steps=int(eligible.sum()),
                         null_rate=float((gate[active] > 0).float().mean()),
                         eligible_null_rate=float((gate[eligible] > 0).float().mean()),
                         null_rate_at30=float((gate[:, -1][active[:, -1]] > 0).float().mean()),
                         neff=distribution(neff[active]), neff_final=distribution(neff[:, -1][active[:, -1]])))
    return rows


def corrected_final(state, kappa=4., dt=1.):
    state = state.clone()
    slope, gate, var, neff = acceleration_fit(state, kappa)
    state[..., CORR] = fit_correction(state, lattice_nodes()[None] + state[..., :2],
                                      state[..., TIME] - dt, kappa)
    return transport_velocity(state, lattice_nodes(), dt, kappa), slope, gate, var, neff


def assign(velocity, truth, evidence):
    """One baseline Hungarian assignment among evidence-bearing nodes."""
    result = []
    for v, target, e in zip(velocity, truth, evidence):
        candidates = (e > 1e-6).nonzero().flatten()
        distance = (target[:, None] - v[candidates][None]).norm(dim=-1)
        rows, cols = linear_sum_assignment(distance.numpy())
        assigned = torch.full((len(target),), -1, dtype=torch.long)
        assigned[torch.as_tensor(rows)] = candidates[torch.as_tensor(cols)]
        result.append(assigned)
    return torch.stack(result)


def matches(velocity, truth, visible, evidence, assignment=None):
    if assignment is None:
        assignment = assign(velocity, truth, evidence)
    valid = visible & (assignment >= 0)
    measured = velocity.gather(1, assignment.clamp_min(0)[..., None].expand(-1, -1, 2))
    errors = (measured - truth).norm(dim=-1)
    selected = errors[valid]
    return dict(median=float(selected.median()), mean=float(selected.mean()), n=int(valid.sum()),
                errors=errors.tolist(), assignments=assignment.tolist()), assignment


def motion_metrics(velocity, acceleration, truth, true_acceleration, visible, evidence, assignment):
    metrics, _ = matches(velocity, truth, visible, evidence, assignment)
    index = assignment.clamp_min(0)[..., None].expand(-1, -1, 2)
    valid = visible & (assignment >= 0)
    dv = velocity.gather(1, index) - truth
    da = acceleration.gather(1, index) - true_acceleration
    # Velocities are outgoing interval means v(29.5), so subtract the half
    # step from the quadratic term to extrapolate from position p(29).
    displacement = {}
    for tau in (20, 50):
        errors = (dv * tau + .5 * da * tau * (tau - 1)).norm(dim=-1)
        displacement[str(tau)] = dict(mean=float(errors[valid].mean()),
                                      median=float(errors[valid].median()), errors=errors.tolist())
    metrics['displacement'] = displacement
    metrics['proxy_mean'] = sum(row['mean'] for row in displacement.values()) / 2
    return metrics


def detection_bins(gate, assignment, visible, acceleration, cfg):
    selected = gate.gather(1, assignment.clamp_min(0))
    mag = acceleration.norm(dim=-1)
    rows = []
    for lo, hi in ((0., 1e-7), (cfg.accel_min, .004), (.004, .008), (.008, cfg.accel + 1e-8)):
        mask = (mag >= lo) & (mag < hi) & visible & (assignment >= 0)
        rows.append(dict(lo=lo, hi=hi, n=int(mask.sum()),
                         gate_open=float((selected[mask] > 0).float().mean()) if mask.any() else None))
    return rows


def synthetic_scene(seed=777, accel=.004, fast=False):
    """Single MNIST digit and its a=0 twin; camera tracks v0 plus jitter."""
    cfg = EnvConfig(n_digits=1, real_velocity=True, real_camera=True, accel=accel,
                    world=32 if fast else 50, window=24 if fast else 32)
    gen = torch.Generator().manual_seed(seed)
    digits, _, _ = load_mnist('test')
    patch = digits[torch.randint(len(digits), (1, 1), generator=gen)]
    acceleration = torch.tensor([[[accel, 0.]]])
    velocity = torch.tensor([[[.2, -.3]]])
    t = torch.arange(30).float()[None, :, None, None]
    pos = torch.tensor([[[8., 8.]]])[:, None] + velocity[:, None] * t + .5 * acceleration[:, None] * t.square()
    actions = velocity[:, 0, None] + torch.rand(1, 30, 2, generator=gen) * .5 - .25
    actions[:, 0] = 0
    cam = (torch.tensor([[[6., 6.]]]) + actions.cumsum(1)).remainder(cfg.world)
    frames = render_digits(patch, pos.remainder(cfg.world), cam, cfg)[0]
    return dict(frames=frames, actions=actions, digit_states=dict(acc=acceleration,
                vel=velocity[:, None] + acceleration[:, None] * (t + .5))), cfg


def synthetic_ensemble(seeds, bs=2):
    """Fast C2/K2 production measurements; select highest-evidence node,
    independently of ground-truth velocity or acceleration and of the gate."""
    result = dict(seeds=list(seeds), C=2, K=2, world=32, window=24, cell_seed=0,
                  assignment='largest cumulative evidence at frame 29',
                  camera='v0 + U[-.25,.25]^2 per step; different from the five-digit random camera', rows=[])
    for accel in (0., .004):
        scenes = [synthetic_scene(seed, accel, fast=True)[0] for seed in seeds]
        batch = {key: torch.cat([s[key] for s in scenes]) for key in ('frames', 'actions')}
        states, seconds = run(batch, encoder(fast=True, order=2), bs)
        final = states[:, -1]
        nodes = final[..., 2].argmax(1)
        sample = torch.arange(len(seeds))
        for kappa in KAPPAS:
            nu, slope, gate, _, neff = corrected_final(final, kappa)
            chosen = gate[sample, nodes]
            truth = torch.cat([s['digit_states']['vel'][:, -1, 0] for s in scenes])
            result['rows'].append(dict(accel=accel, kappa=kappa, n=len(seeds),
                open_rate=float((chosen > 0).float().mean()), gates=chosen.tolist(), nodes=nodes.tolist(),
                velocity_error=(nu[sample, nodes] - truth).norm(dim=-1).tolist(),
                acceleration_error=(slope[sample, nodes] - torch.tensor([accel, 0.])).norm(dim=-1).tolist(),
                neff=distribution(neff[sample, nodes]), seconds=seconds))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--accel', type=float, required=True)
    p.add_argument('--accel_zero_p', type=float, default=.3)
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--seed', type=int, default=777)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--synthetic-seeds', type=int, default=50)
    p.add_argument('--out', type=Path, default=Path('results/order2_gate'))
    args = p.parse_args(argv)
    if args.synthetic_seeds < 50:
        p.error('--synthetic-seeds must be at least 50')
    torch.set_num_threads(1)
    start = time.perf_counter()
    cfg = EnvConfig(real_velocity=True, real_camera=True, accel=args.accel, accel_zero_p=args.accel_zero_p)
    common = dict(n=args.n, seed=args.seed, batch_size=args.batch_size, cell_seed=0, C=32, K=8,
                  observed_frames=30, dtype='float32', device='cpu', environment=cfg.to_dict(),
                  variance='Kish sandwich: residual scatter / Ctt * sum(w^2)/sum(w)',
                  jump_sigma=6., gate_neff_floor=4, prediction='off', schema=2)
    synthetic = synthetic_ensemble(range(args.seed, args.seed + args.synthetic_seeds), args.batch_size)
    save(args.out / 'synthetic.json', synthetic)
    print('SYNTHETIC:', [(r['accel'], r['kappa'], r['open_rate']) for r in synthetic['rows']], flush=True)

    batch = make_batch(args.n, 30, 0, cfg, gen=torch.Generator().manual_seed(args.seed), split='test')
    states_by_f = {}
    for forget in (1., *FORGET):
        states_by_f[forget], _ = run(batch, encoder(order=2, nu_forget=forget), args.batch_size)
    truth, acc = batch['digit_states']['vel'][:, -1], batch['digit_states']['acc']
    visible = batch['digit_states']['visible'][:, -10:].any(1)
    # Shared ownership avoids each estimator selecting its own easiest node.
    evidence = torch.stack([s[:, -1, :, 2] for s in states_by_f.values()]).amin(0)
    assignment = assign(lattice_nodes()[None] + states_by_f[1.][:, -1, :, :2], truth, evidence)
    metrics, bins, neff_stats = {}, {}, {}
    for forget, states in states_by_f.items():
        final = states[:, -1]
        suffix = '' if forget == 1 else f'_F{forget:g}'
        mean = lattice_nodes()[None] + final[..., :2]
        name = 'order1' + suffix
        metrics[name] = motion_metrics(mean, torch.zeros_like(mean), truth, acc, visible, evidence, assignment)
        active = states[..., 2] > 1e-6
        metrics[name]['clamp_rate_obs'] = float(clamp_mask(states, order=1)[active].float().mean())
        for kappa in KAPPAS:
            nu, slope, gate, _, neff = corrected_final(final, kappa)
            name = f'order2{suffix}_k{kappa:g}'
            metrics[name] = motion_metrics(nu, gate[..., None] * slope, truth, acc, visible, evidence, assignment)
            bins[name] = detection_bins(gate, assignment, visible, acc, cfg)
            # Corrections in stored history were calculated at kappa=4.
            corrected = states.clone()
            corrected[..., CORR] = fit_correction(states, lattice_nodes()[None, None] + states[..., :2],
                                                  states[..., TIME] - 1, kappa)
            metrics[name]['clamp_rate_obs'] = float(clamp_mask(corrected, kappa=kappa)[active].float().mean())
            neff_stats[name] = distribution(neff[final[..., 2] > 1e-6])
    best_f_name = min((f'order1_F{f:g}' for f in FORGET), key=lambda n: metrics[n]['proxy_mean'])
    best_f = float(best_f_name.split('_F')[1])
    detection = dict(**common, metrics=metrics, bins=bins, neff_final=neff_stats,
                     best_forgetting=best_f, assignment='order1 Hungarian, common evidence-bearing nodes, then visibility',
                     proxy='mean Euclidean displacement error at tau=20 and 50, averaged equally (pixels)',
                     dataset_sha256=hashlib.sha256(batch['frames'].numpy().tobytes()).hexdigest())
    save(args.out / 'detection.json', detection)

    # Calibrate standalone order2 and order2 atop the best forgetting mean.
    # This bounds runtime: 6 accelerated + 2 null production passes, not 12.
    null_cfg = EnvConfig(real_velocity=True, real_camera=True)
    null = make_batch(args.n, 30, 0, null_cfg, gen=torch.Generator().manual_seed(args.seed), split='test')
    null_rows = []
    for forget in (1., best_f):
        ns, _ = run(null, encoder(order=2, nu_forget=forget), args.batch_size)
        null_rows.extend(dict(forget=forget, **r) for r in null_table(ns))
    calibration = dict(**common, rows=null_rows, dataset_environment=null_cfg.to_dict(),
                       denominator='node-steps with cumulative evidence > 1e-6 at that step',
                       dataset_sha256=hashlib.sha256(null['frames'].numpy().tobytes()).hexdigest())
    save(args.out / 'null.json', calibration)
    candidates = []
    for kappa in KAPPAS:
        rate = next(r['null_rate'] for r in null_rows if r['forget'] == 1 and r['kappa'] == kappa)
        name = f'order2_k{kappa:g}'
        candidates.append(dict(name=name, kappa=kappa, null_rate=rate,
            improvement=1 - metrics[name]['proxy_mean'] / metrics[best_f_name]['proxy_mean']))
    eligible = [r for r in candidates if r['null_rate'] < .02]
    chosen = max(eligible or candidates, key=lambda r: r['improvement'])
    combo = min((f'order2_F{best_f:g}_k{k:g}' for k in KAPPAS), key=lambda n: metrics[n]['proxy_mean'])
    summary = dict(**common, best_forgetting=best_f_name, order2=chosen, order2_plus_forgetting=combo,
                   go=chosen['null_rate'] < .02 and chosen['improvement'] >= .1,
                   criterion='K-real evidence-bearing null node-step rate < 2%; standalone order2 proxy mean <= 0.9 * best F',
                   candidates=candidates, elapsed_seconds=time.perf_counter() - start)
    save(args.out / 'summary.json', summary)
    write_report(common, calibration, detection, synthetic, summary)


def write_report(common, calibration, detection, synthetic, summary):
    """Render completed results without rerunning production measurements."""
    null_rows = calibration['rows']
    metrics, bins = detection['metrics'], detection['bins']
    best_f = detection['best_forgetting']
    chosen, combo = summary['order2'], summary['order2_plus_forgetting']
    n, seed, synthetic_n = common['n'], common['seed'], len(synthetic['seeds'])
    lines = ['# Order-2 estimator gate', '',
        f"**{'GO' if summary['go'] else 'NO-GO'}**: null < 2% and at least 10% lower displacement proxy mean than best F are both required.",
        f"Selected order 2: κ={chosen['kappa']:g}, null {chosen['null_rate']:.3%}, proxy improvement {chosen['improvement']:.2%}.", '',
        f'{n} five-digit sequences per dataset, seed {seed}; frozen C32/K8 encoder seed 0; CPU fp32, one thread.',
        'Prediction policy: off. F sweep: 0.6, 0.7, 0.8, 0.9, 0.95; κ sweep: 4, 9.',
        'All arms share the order-1 Hungarian assignment among nodes with evidence in every arm, then the last-ten-frame visibility filter.', '',
        '| F under fit | κ | Null node-step rate | Eligible (n_eff ≥ 4) rate | Final-frame rate | Final n_eff p10 / median / p90 |',
        '| --- | --- | --- | --- | --- | --- |']
    for r in null_rows:
        n = r['neff_final']
        lines.append(f"| {r['forget']:g} | {r['kappa']:g} | {r['null_rate']:.3%} | {r['eligible_null_rate']:.3%} | {r['null_rate_at30']:.3%} | {n['p10']:.2f} / {n['median']:.2f} / {n['p90']:.2f} |")
    lines += ['', 'Null denominator: evidence-bearing node-steps at each time. JSON also records eligible counts and n_eff quantiles over time.', '',
        '| F under fit | K-accel final n_eff p10 / median / p90 | Fraction n_eff ≥ 4 |', '| --- | --- | --- |']
    for forget in (1., best_f):
        name = 'order2_k4' if forget == 1 else f'order2_F{forget:g}_k4'
        neff = detection['neff_final'][name]
        lines.append(f"| {forget:g} | {neff['p10']:.2f} / {neff['median']:.2f} / {neff['p90']:.2f} | {neff['fraction_ge4']:.2%} |")
    lines += ['',
        '| Arm | Median velocity error | Mean displacement @20 | @50 | Proxy mean | Obs clamp rate |',
        '| --- | --- | --- | --- | --- | --- |']
    for name, r in metrics.items():
        lines.append(f"| {name} | {r['median']:.6f} | {r['displacement']['20']['mean']:.4f} | {r['displacement']['50']['mean']:.4f} | {r['proxy_mean']:.4f} | {r['clamp_rate_obs']:.3%} |")
    lines += ['', f"Best F by proxy mean: {best_f:g}. Best order 2 + that F: {combo}.",
        'Velocity truth is v(29.5). Displacement from p(29) is ντ + ½gατ(τ−1); error is Euclidean distance in pixels, averaged over digits.',
        'The proxy averages τ=20 and 50 equally. It uses final node velocity and acceleration; it omits future clamps, rendering, and association changes.',
        'Clamp rates count evidence-bearing node-steps, include the half-step acceleration, and report order-1 saturation too.', '',
        '| Arm | Acceleration bin | Visible assigned digits | Gate open |', '| --- | --- | --- | --- |']
    for name in ('order2_k4', 'order2_k9', f'order2_F{best_f:g}_k4', f'order2_F{best_f:g}_k9'):
        for r in bins[name]:
            rate = f"{r['gate_open']:.3%}" if r['gate_open'] is not None else 'NA'
            lines.append(f"| {name} | [{r['lo']:g}, {r['hi']:g}) | {r['n']} | {rate} |")
    lines += ['', f'Single-digit paired ensemble: {synthetic_n} seeds starting at {seed}; fast frozen C2/K2 encoder, world 32, window 24.',
        'The camera tracks v0 plus U[-.25,.25]²; this is easier than the five-digit random camera. Select the highest-evidence node, without using truth.', '',
        '| Acceleration | κ | Final gate-open rate |', '| --- | --- | --- |']
    for r in synthetic['rows']:
        lines.append(f"| {r['accel']:g} | {r['kappa']:g} | {r['open_rate']:.3%} |")
    lines += ['', f"Total runtime: {summary['elapsed_seconds']:.2f} seconds on one CPU thread (budget: 1800 seconds).",
        'κ is selected among calibrated settings by proxy mean. The +F summary uses the best order-1 F and best κ; its null rates are reported separately.',
        'Fits are reused only where measurement streams are identical. Forgetting arms rerun the production estimator.',
        'All settings are exploratory in-sample diagnostics; no trained forecasting improvement is established.', '']
    Path('docs/GATE_order2.md').write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
