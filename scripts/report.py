"""Paired multi-seed evaluation reports, using only the existing numpy dependency.

python scripts/report.py --arm lk 'results/lk_s*/eval_seed777.json' \
    --arm lock 'results/lock_s*/eval_seed777.json' --pairs lk:lock --holm
"""
import argparse
import glob
import itertools
import json
import math
from pathlib import Path
import sys
import warnings

import numpy as np


def _beta_fraction(a, b, x):
    """Modified Lentz continued fraction for the incomplete beta function."""
    tiny = 1e-300

    def nonzero(v):
        return v if abs(v) >= tiny else math.copysign(tiny, v)

    c = 1.0
    d = 1.0 / nonzero(1.0 - (a + b) * x / (a + 1.0))
    h = d
    for m in range(1, 10001):
        aa = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 / nonzero(1.0 + aa * d)
        c = nonzero(1.0 + aa / c)
        h *= d * c
        aa = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 / nonzero(1.0 + aa * d)
        c = nonzero(1.0 + aa / c)
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            return h
    raise ArithmeticError("incomplete beta continued fraction did not converge")


def regularized_beta(x, a, b):
    if not 0 <= x <= 1 or a <= 0 or b <= 0:
        raise ValueError("beta requires 0 <= x <= 1 and positive a, b")
    if x == 0 or x == 1:
        return float(x)
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        value = front * _beta_fraction(a, b, x) / a
    else:
        value = 1.0 - front * _beta_fraction(b, a, 1.0 - x) / b
    return min(1.0, max(0.0, value))


def student_t_pvalue(t, df):
    """Two-sided Student-t tail probability, without subtracting a CDF near 1."""
    if df <= 0 or math.isnan(t):
        raise ValueError("Student-t requires positive df and a non-NaN statistic")
    return regularized_beta(df / (df + t * t), df / 2.0, 0.5)


def student_t_cdf(t, df):
    tail = student_t_pvalue(t, df) / 2.0
    return tail if t < 0 else 1.0 - tail


def paired_t(differences):
    d = np.asarray(differences, dtype=np.float64)
    if len(d) < 2:
        return None, None
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    t = mean / (sd / math.sqrt(len(d))) if sd else (math.copysign(math.inf, mean) if mean else 0.0)
    return t, student_t_pvalue(t, len(d) - 1)


def sign_flip_pvalue(differences):
    """Exact two-sided randomization test, including ties and all 2**n patterns."""
    d = list(differences)
    if not d:
        raise ValueError("sign-flip test requires at least one pair")
    observed = abs(math.fsum(d))
    count = 0
    for signs in itertools.product((-1, 1), repeat=len(d)):
        value = abs(math.fsum(s * v for s, v in zip(signs, d)))
        count += value >= observed or math.isclose(value, observed, rel_tol=1e-12, abs_tol=0.0)
    return count / (2 ** len(d))


def holm_adjust(pvalues):
    """Holm step-down adjustment in original order; None denotes an undefined test."""
    result = [None] * len(pvalues)
    order = sorted((i for i, p in enumerate(pvalues) if p is not None), key=lambda i: pvalues[i])
    previous = 0.0
    for rank, i in enumerate(order):
        previous = max(previous, min(1.0, (len(order) - rank) * pvalues[i]))
        result[i] = previous
    return result


def _scalar(record, metric):
    value = record.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{record.get('_path', 'record')}: {metric!r} must be a finite scalar")
    return float(value)


def pair_records(comparator, treatment):
    """Pair by training seed, requiring equal seed sets; legacy inputs use file order."""
    if any(r.get("seed") is None for r in comparator + treatment):
        warnings.warn("missing seed: pairing by file order (explicit lists, sorted within each glob)", stacklevel=2)
        if len(comparator) != len(treatment):
            raise ValueError("file-order pairing requires equal arm lengths")
        return comparator, treatment, "file_order"

    def index(records):
        indexed = {}
        for r in records:
            seed = r["seed"]
            if isinstance(seed, bool) or not isinstance(seed, (int, str)):
                raise ValueError("seed must be an integer or string")
            if seed in indexed:
                raise ValueError(f"duplicate seed {seed!r} in an arm")
            indexed[seed] = r
        return indexed

    left, right = index(comparator), index(treatment)
    if left.keys() != right.keys():
        raise ValueError("paired arms must contain exactly the same seeds")
    return list(left.values()), [right[s] for s in left], "seed"


def _sequence_differences(left, right, metric):
    key = f"per_seq_{metric}"
    a, b = left.get(key), right.get(key)
    if a is None or b is None:
        return None
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.ndim != 1 or b.shape != a.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"{key}: paired sequence lists must be nonempty, finite, and equally sized")
    # Index-wise resampling is paired only when both results evaluated the same scenes.
    for key in ("test_seed", "n_val", "env"):
        if key in left and key in right and left[key] != right[key]:
            raise ValueError(f"paired sequences have different {key}")
    left_set, right_set = left.get("sets", {}).get("main"), right.get("sets", {}).get("main")
    if left_set is not None and right_set is not None and left_set != right_set:
        raise ValueError("paired sequences have different main set definitions")
    for record, values in ((left, a), (right, b)):
        if record.get("n_val", len(values)) != len(values):
            raise ValueError(f"per_seq_{metric} length differs from n_val")
        if not math.isclose(float(values.mean()), _scalar(record, metric), rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"per_seq_{metric} mean differs from {metric}")
    return b - a


def bootstrap_interval(comparator, treatment, metric, replicates=2000, seed=0):
    """Resample paired seeds, then paired sequence indices independently per draw."""
    n = len(comparator)
    differences = np.array([_scalar(b, metric) - _scalar(a, metric) for a, b in zip(comparator, treatment)])
    sequences = [_sequence_differences(a, b, metric) for a, b in zip(comparator, treatment)]
    rng = np.random.default_rng(seed)
    draws = rng.integers(n, size=(replicates, n))
    values = differences[draws].copy()
    for i, seq in enumerate(sequences):
        if seq is None:
            continue
        rows, cols = np.where(draws == i)
        # Bound memory for large per-sequence arrays. Every occurrence gets a fresh draw.
        batch = max(1, 1000000 // len(seq))
        for start in range(0, len(rows), batch):
            r, c = rows[start:start + batch], cols[start:start + batch]
            indices = rng.integers(len(seq), size=(len(r), len(seq)))
            values[r, c] = seq[indices].mean(axis=1)
    return np.quantile(values.mean(axis=1), [0.025, 0.975]).tolist(), sum(s is not None for s in sequences)


def build_report(arms, pairs, metrics=("mse20", "mse150"), holm=False):
    if not arms or any(not records for records in arms.values()):
        raise ValueError("each arm requires at least one evaluation")
    report = dict(metrics={}, comparisons=[], bootstrap=dict(replicates=2000, seed=0),
                  difference="treatment - comparator")
    for metric in metrics:
        summaries = {}
        for name, records in arms.items():
            values = [_scalar(r, metric) for r in records]
            summaries[name] = dict(values=[dict(seed=r.get("seed"), value=v, file=r.get("_path"))
                                          for r, v in zip(records, values)],
                                   mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                                   n=len(values))
        report["metrics"][metric] = summaries
    for pair in pairs:
        names = pair.split(":")
        if len(names) != 2 or any(name not in arms for name in names) or names[0] == names[1]:
            raise ValueError(f"invalid comparator:treatment pair: {pair}")
        left, right, pairing = pair_records(arms[names[0]], arms[names[1]])
        for metric in metrics:
            d = [_scalar(b, metric) - _scalar(a, metric) for a, b in zip(left, right)]
            t, p = paired_t(d)
            interval, sequence_seeds = bootstrap_interval(left, right, metric)
            report["comparisons"].append(dict(pair=pair, metric=metric, pairing=pairing, n=len(d), df=len(d) - 1,
                                               seeds=[r.get("seed") for r in left], differences=d,
                                               mean_difference=float(np.mean(d)), t=t, t_p=p,
                                               sign_flip_p=sign_flip_pvalue(d), bootstrap_ci95=interval,
                                               bootstrap_sequence_seeds=sequence_seeds))
    if holm:
        # Each test method has one family across all listed pairs and metrics.
        for key in ("t_p", "sign_flip_p"):
            adjusted = holm_adjust([c[key] for c in report["comparisons"]])
            for comparison, value in zip(report["comparisons"], adjusted):
                comparison[key + "_holm"] = value
    return report


def markdown(report):
    def fmt(value):
        return "NA" if value is None else f"{value:.8g}"

    lines = []
    for metric, summaries in report["metrics"].items():
        lines += [f"### {metric}", "", "| Arm | Per-seed values (seed: value) | Mean | SD | n |",
                  "|---|---|---:|---:|---:|"]
        for arm, stats in summaries.items():
            values = "; ".join(f"{v['seed'] if v['seed'] is not None else 'file#' + str(i + 1)}: {fmt(v['value'])}"
                               for i, v in enumerate(stats["values"]))
            lines.append(f"| {arm} | {values} | {fmt(stats['mean'])} | {fmt(stats['sd'])} | {stats['n']} |")
        lines.append("")
    if report["comparisons"]:
        holm = "t_p_holm" in report["comparisons"][0]
        lines += ["Differences are treatment − comparator. Intervals use 2,000 bootstrap replicates (seed 0).", "",
                  "| Pair | Metric | n | Mean difference | t (df = n−1) | Two-sided p | Exact sign-flip p | Bootstrap 95% CI |"
                  + (" Holm t p | Holm sign-flip p |" if holm else ""),
                  "|---|---|---:|---:|---:|---:|---:|---|" + ("---:|---:|" if holm else "")]
        for c in report["comparisons"]:
            lo, hi = c["bootstrap_ci95"]
            lines.append(f"| {c['pair']} | {c['metric']} | {c['n']} | {fmt(c['mean_difference'])} | {fmt(c['t'])} | "
                         f"{fmt(c['t_p'])} | {fmt(c['sign_flip_p'])} | [{fmt(lo)}, {fmt(hi)}] |"
                         + (f" {fmt(c['t_p_holm'])} | {fmt(c['sign_flip_p_holm'])} |" if holm else ""))
        lines.append("")
    return "\n".join(lines)


def load_arms(manifest=None, cli_arms=()):
    specs = []
    if manifest:
        path = Path(manifest)
        with path.open() as f:
            mapping = json.load(f)
        if not isinstance(mapping, dict):
            raise ValueError("manifest must map arm names to lists of paths/globs")
        mapping = mapping.get("arms", mapping)
        if not isinstance(mapping, dict):
            raise ValueError("manifest must map arm names to lists of paths/globs")
        for name, patterns in mapping.items():
            specs.append((name, patterns, path.parent))
    for arm in cli_arms:
        if len(arm) < 2:
            raise ValueError("--arm requires a name and at least one path/glob")
        specs.append((arm[0], arm[1:], Path.cwd()))
    arms = {}
    for name, patterns, base in specs:
        if isinstance(patterns, str):
            patterns = [patterns]
        if not isinstance(patterns, list) or not patterns:
            raise ValueError(f"arm {name}: expected a nonempty list of paths/globs")
        records = arms.setdefault(name, [])
        seen = {r["_path"] for r in records}
        for pattern in patterns:
            if not isinstance(pattern, str):
                raise ValueError(f"arm {name}: paths/globs must be strings")
            matches = sorted(glob.glob(str(base / pattern)))
            if not matches:
                raise ValueError(f"arm {name}: no files match {pattern!r}")
            for match in matches:
                path = str(Path(match).resolve())
                if path in seen:
                    raise ValueError(f"arm {name}: duplicate file {path}")
                seen.add(path)
                with open(path) as f:
                    record = json.load(f)
                records.append(dict(record, _path=path))
    return arms


def _json_safe(value):
    # JSON has no infinity literal; degenerate nonzero paired differences have infinite t.
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", nargs="?", help="JSON arm mapping; paths relative to this file")
    p.add_argument("--manifest", dest="manifest_option")
    p.add_argument("--arm", action="append", nargs="+", default=[], metavar="NAME_OR_GLOB")
    p.add_argument("--pairs", action="extend", nargs="+", default=[], metavar="COMPARATOR:TREATMENT")
    p.add_argument("--metrics", nargs="+", default=["mse20", "mse150"])
    p.add_argument("--holm", action="store_true")
    p.add_argument("--out", help="also write the Markdown table to this path")
    p.add_argument("--json", nargs="?", const="-", metavar="PATH", help="write JSON to PATH (omitted or -: stdout)")
    a = p.parse_args(argv)
    if a.manifest and a.manifest_option:
        p.error("give the manifest once")
    try:
        report = build_report(load_arms(a.manifest or a.manifest_option, a.arm), a.pairs, a.metrics, a.holm)
    except (OSError, ValueError) as e:
        p.error(str(e))
    table = markdown(report)
    encoded = json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n"
    if a.out:
        Path(a.out).write_text(table, encoding="utf-8")
    if a.json and a.json != "-":
        Path(a.json).write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded if a.json == "-" else table)


if __name__ == "__main__":
    main()
