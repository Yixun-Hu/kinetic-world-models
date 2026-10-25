# Evaluation hygiene

`kwm.train` selects `best.pt` and `best150.pt` on generated sequences from
`VAL_SEED = 12345`. Historically, `kwm.eval` scored checkpoints on that same
selection set. These scores describe performance on the set used to select the
checkpoint; they do not provide an independent test estimate.

## Separate generated test sequences

Choose a test seed before comparing models, and use it consistently across arms
and training seeds. Keep checkpoint selection on the validation set.

```bash
python -m kwm.eval --run runs/lk_s0 --ckpt last --n_val 300 \
  --test_seed 777
```

This writes `results/lk_s0/eval_seed777_last.json` and similarly suffixed figures.
For non-default seeds the checkpoint name is included automatically; an optional
`--tag extra` adds another suffix, producing `eval_seed777_last_extra.json`.
Evaluating `--ckpt best` instead writes `eval_seed777_best.json`.

`--test_seed` defaults to `None`, meaning seed 12345. Omitting it or explicitly
passing 12345 retains the original cache/output names and metric calculations.
For any other seed, `val_cache` adds `_seed<seed>` before `.pt`, and evaluation
adds `_seed<seed>_<ckpt>` before the optional tag in every output filename. The
cache generator uses the chosen seed. Cache chunking, float16 frame storage, and default cache
contents are unchanged. JSON and checkpoint metadata are additive.

Both selection and test sequences use the MNIST **test split**. A new seed holds
out generated trajectories and scenes, not a disjoint pool of digit images.
Use identical environment settings, observation lengths, sequence counts, and
cache chunk sizes across paired arms. With the sequential generator, different
sequence counts share prefixes through complete, aligned chunks; changing chunk
sizes or the final partial chunk can change the generated sequences.
Do not tune choices against the held-out scores.

The JSON `sets` object records the seeds used by each requested calculation:

| Key | Data source |
|---|---|
| `main` | `--test_seed`: MSE, PSNR, SSIM, curves/slopes, bounce errors, velocity probes, and sample figures |
| `node_velocity_error` | The first up to 64 sequences from the main cache, for adaptive models |
| `throughput` | The first throughput batch from the main cache |
| `fractional_time` | `--test_seed` with `dt=0.5`; all fractional-time baselines use this cache |
| `equiv`, `equiv_output` | Existing independent generation: boost seed 777, sequence seed `777 * 1000 + chunk_start`, chunks of 16 |

Only requested calculations appear in `sets`; diagnostic results can still
report that they were skipped for an unsupported model/environment. Entries
record the configured seed, `mnist_split`, count, and generation dimensions where
applicable. Each entry also has `role`: `selection` for seed 12345 caches and
`heldout` for other caches and the independent equivariance generator. This role
describes the generation scheme, not whether a user has subsequently tuned on it.
`--test_seed` controls main and fractional-time data; equivariance keeps its fixed
independent generation scheme.
`--only_equiv` retains previously saved metrics and their provenance when
merging results. `diagnostic_provenance` identifies the code, checkpoint, and
step used for the newly merged equivariance results, along with `seed`,
`test_seed`, and `sets`. These fields are never added to the top level of a legacy
JSON by a diagnostic merge; existing top-level metadata stays unchanged.

## JSON and checkpoint provenance

Evaluation JSON includes:

- `seed`: the training seed read from checkpoint arguments (null if unknown).
- `test_seed`, `n_val`, `ckpt`: the effective main-cache seed, evaluated sequence
  count, and checkpoint basename.
- `per_seq_mse20`, `per_seq_mse150`: floats in cache order, one per evaluated
  sequence. Each is the spatial MSE averaged over the first 20 or 150 predicted
  frames. Their means equal the existing `mse20` and `mse150` values. The
  original reductions and confidence-interval calculations are unchanged.
- `sets`: the data sources listed above.
- `provenance`: the source fingerprint, software versions, and the
  evaluation environment fingerprint under `eval_env`.
- `ckpt_provenance`: provenance copied from the checkpoint, or null for old
  checkpoints.

`kwm.provenance.source_hashes()` returns `git_head`, `git_dirty`, `software`, and `files`,
a mapping from every Python path under `kwm/` to its SHA-256. File paths are
relative to the repository root. Dirty status covers only tracked changes under
`kwm/` and `scripts/`, using `git status --porcelain --untracked-files=no -- kwm scripts`;
generated results, scratch files, and untracked files do not affect it.
`software` records Python, PyTorch, and NumPy versions plus CUDA availability
(`python`, `torch`, `numpy`, `cuda_available`). If git is unavailable or the code
is outside a repository, both git fields are null; file hashes remain available.

`env_fingerprint(cfg)` returns `{"cfg": <config dict>, "sha256": <hash>}`.
The hash covers UTF-8 JSON with sorted keys, compact separators, and unescaped
Unicode; nonfinite numbers are rejected. This fingerprints the scene configuration,
not the software environment. Evaluation records it as `eval_env`; training uses
`env` for its training configuration and `validation_env` for validation.
Training snapshots provenance at startup and writes it into `config.json` and every checkpoint it saves,
including `last`, `best`, `best150`, and retained step checkpoints. Resumed
training records the source used for the resumed invocation and preserves a
`provenance_history` list of `{step, provenance}` in `config.json` and each newly
saved checkpoint. Entries mark startup/resume boundaries at the number of steps
already completed (0 for a new run). If a checkpoint has no history, its existing
provenance is recorded at its saved step before appending the resume entry; a
legacy checkpoint records null for its unknown prior provenance. Previously
saved best checkpoints retain their own provenance and history.

Evaluation and resumed training print one warning line if a checkpoint's
source-file hashes differ from the running code's hashes. A git revision or
dirty-status difference alone does not trigger it. Old checkpoints load normally
with null provenance.

## Reports across training seeds

Provide one evaluation JSON per training seed in each arm. Quote globs so the
script expands them. For example, after evaluating both arms for each seed:

```bash
python scripts/report.py \
  --arm lk 'results/lk_s*/eval_seed777_last.json' \
  --arm lock 'results/lock_s*/eval_seed777_last.json' \
  --pairs lk:lock --metrics mse20 mse150 --holm \
  --out table.md --json report.json
```

A manifest can replace the arm arguments. Paths/globs in it are relative to the
manifest's directory; CLI paths/globs are relative to the working directory.

```json
{
  "arms": {
    "lk": ["results/lk_s0/eval_seed777_last.json", "results/lk_s1/eval_seed777_last.json"],
    "lock": ["results/lock_s0/eval_seed777_last.json", "results/lock_s1/eval_seed777_last.json"]
  }
}
```

```bash
python scripts/report.py --manifest manifest.json --pairs lk:lock --holm --out table.md
```

The manifest must use the `{"arms": {...}}` wrapper and the `--manifest PATH`
option. Multiple paths/globs per `--arm`, repeated `--arm`, and several pairs
after `--pairs` are supported. `--metrics` defaults to `mse20 mse150`; any finite
top-level scalar JSON key is accepted. Missing or nonfinite metrics are errors.
Markdown goes to stdout and optionally `--out`. `--json PATH` also writes a
machine-readable report; `--json -` puts JSON on stdout instead. A path is required.
Report output uses ASCII.

Before computing any statistics, the script validates `test_seed`, `n_val`,
the canonical SHA-256 of `env`, and `ckpt` within each arm and across each requested
pair, for every metric, even without per-sequence lists. Missing `n_val`, `env`,
or `ckpt` and any mismatches are errors. Missing/null `test_seed` implies the
legacy selection seed 12345 and emits a warning; explicit 12345 also warns that
the scores are on the selection set. Known main-set generation definitions must
agree too. The table header displays each arm's common `test_seed`, `ckpt`, and
`n_val`; the JSON `evaluations` object also includes `env_sha256`.

For every metric the table lists each seed's value, mean, sample standard
deviation (`ddof=1`), and number of seeds. Pairs use `comparator:treatment`, and
differences are **treatment minus comparator** (negative MSE favors treatment).
Seeds are matched by `seed`, not `test_seed` or filename. Duplicate seeds and
unequal seed sets are errors. Mixing present and missing/null training seeds is
an error. Only when all training seeds in a pair are missing does the script warn
and pair by file order, requiring equal lengths. Explicit lists keep their
order; matches within each glob are sorted naturally (`s2` before `s10`).

Each pair and metric reports:

- Mean paired difference and a paired t statistic with `df = n - 1`. The
  two-sided p-value uses the regularized incomplete beta function, implemented
  with a continued fraction; SciPy is not required.
- An exact two-sided sign-flip p-value: all `2^n` sign patterns are enumerated,
  counting ties as at least as extreme as the observed absolute mean. Runtime
  therefore grows exponentially with the number of training seeds.
- A percentile hierarchical-bootstrap 95% interval from 2,000 replicates with
  RNG seed 0. Each replicate samples paired training seeds with replacement,
  then samples paired sequence indices with replacement within each sampled
  seed when both arms have `per_seq_<metric>`. Repeated draws of a seed get
  independent sequence samples. Seeds without both lists use their scalar
  difference. The JSON reports how many seeds supplied sequence lists. Lists
  must have equal lengths and means matching their scalar metrics; recorded
  evaluation definitions are checked before any statistics as described above.
  Resampling both seeds and sequences double-counts within-seed noise already
  present in the seed-level variance, so the hierarchical interval is conservative.

`--holm` applies a Holm step-down adjustment across all listed pairs × metrics,
separately for the t-test and sign-flip p-value families. Raw p-values remain
in the report. With one seed, sample SD and t-test results are `NA`/null. With
constant nonzero differences the t statistic is infinite (JSON string `inf`
or `-inf`) and its p-value is zero; all-zero differences give t=0 and p=1.
With fewer than six seeds, the exact sign-flip p-value cannot be below 0.05
(its minimum is `2 / 2^n`), and bootstrap intervals with very few seeds can be
degenerate or unstable.
These tests describe variation across the supplied training seeds and sequences;
they do not account for tuning on those results.
