"""Fast CPU checks for held-out caches, provenance, and paired statistics."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from kwm import env, eval as evaluate, provenance, train
from kwm.model import build_model
from scripts import report


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_data(tmp_path, monkeypatch):
    # Keep generator/rendering intact, with a small local digit bank instead of a download.
    digits = torch.zeros(3, 28, 28)
    digits[0, 8:20, 12:16] = 1
    digits[1, 12:16, 8:20] = 0.75
    digits[2, 8:20, 8:20] = 0.5
    offsets = torch.full((3, 2), 13.5)
    halves = torch.tensor([[1.5, 5.5], [5.5, 1.5], [5.5, 5.5]])
    monkeypatch.setattr(env, "load_mnist", lambda split="train": (digits, offsets, halves))
    monkeypatch.setattr(train, "ROOT", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("chunk", [2, 50])
def test_cache_default_matches_original_generator(tiny_data, chunk):
    cfg = env.EnvConfig(world=28, window=12, real_velocity=True, real_camera=True)
    # Original cache algorithm: a CPU generator seeded with 12345 and the same chunks.
    gen = torch.Generator().manual_seed(12345)
    parts = [env.make_batch(min(chunk, 4 - i), 3, 2, cfg, "cpu", gen, split="test")
             for i in range(0, 4, chunk)]
    original = dict(frames=torch.cat([b["frames"].half() for b in parts]),
                    actions=torch.cat([b["actions"] for b in parts]),
                    actions_noisy=torch.cat([b["actions_noisy"] for b in parts]))
    cache = train.val_cache(cfg, 4, 3, 2, chunk=chunk)
    path = tiny_data / "results" / f"val_cache_{cfg.tag()}_n4_T3+2.pt"
    assert path.is_file()
    assert list(path.parent.glob("*.pt")) == [path]
    before = path.read_bytes()
    explicit = train.val_cache(cfg, 4, 3, 2, chunk=chunk, seed=train.VAL_SEED)
    assert path.read_bytes() == before
    for key in original:
        assert torch.equal(original[key], cache[key])
        assert torch.equal(cache[key], explicit[key])
    other = train.val_cache(cfg, 4, 3, 2, chunk=chunk, seed=777)
    assert path.with_name(path.stem + "_seed777.pt").is_file()
    assert not torch.equal(cache["frames"], other["frames"])
    assert not torch.equal(cache["actions"], other["actions"])


def test_source_hashes_and_environment():
    first = provenance.source_hashes()
    assert first == provenance.source_hashes()
    expected = {p.relative_to(provenance.ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (provenance.ROOT / "kwm").rglob("*.py")}
    assert first["files"] == expected
    assert first["git_head"] is None or len(first["git_head"]) == 40
    assert first["git_dirty"] is None or isinstance(first["git_dirty"], bool)
    cfg = env.EnvConfig().to_dict()
    fingerprint = provenance.env_fingerprint(cfg)
    assert fingerprint == provenance.env_fingerprint(dict(reversed(list(cfg.items()))))
    assert fingerprint["cfg"] == cfg
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    assert fingerprint["sha256"] == hashlib.sha256(canonical.encode()).hexdigest()


@pytest.mark.parametrize("error", [FileNotFoundError(), subprocess.CalledProcessError(128, "git")])
def test_source_hashes_without_git(monkeypatch, error):
    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(provenance.subprocess, "check_output", unavailable)
    result = provenance.source_hashes()
    assert result["git_head"] is result["git_dirty"] is None
    assert "kwm/provenance.py" in result["files"]


def _tiny_args():
    return vars(train.parse_args(["--model", "spectral", "--C", "2", "--world", "28", "--window", "12",
                                  "--node_spacing", "4", "--t_obs", "3", "--t_pred", "2", "--n_val", "4",
                                  "--seed", "9", "--device", "cpu"]))


@pytest.mark.parametrize("test_seed,with_provenance", [(None, False), (12345, True), (777, True)])
def test_eval_json_and_unchanged_mse(tiny_data, capsys, test_seed, with_provenance):
    args = _tiny_args()
    torch.manual_seed(4)
    model = build_model(args).eval()
    run = tiny_data / "run"
    run.mkdir()
    ck = dict(args=args, step=1, model=model.state_dict())
    if with_provenance:
        ck["provenance"] = dict(provenance.source_hashes(), env=provenance.env_fingerprint(env.EnvConfig()))
        ck["provenance"]["files"]["kwm/model.py"] = "older source"
    torch.save(ck, run / "last.pt")
    out = tiny_data / "eval"
    flags = ["--run", str(run), "--ckpt", "last", "--out", str(out), "--device", "cpu", "--bs", "3",
             "--skip_ft", "--skip_tp", "--tag", "tiny"]
    if test_seed is not None:
        flags += ["--test_seed", str(test_seed)]
    if test_seed == 777:
        flags.remove("--skip_ft")
        flags.remove("--skip_tp")
        flags += ["--equiv_boost", "--n_equiv", "2"]
    evaluate.main(flags)
    suffix = f"_seed{test_seed}" if test_seed not in (None, 12345) else ""
    res = json.loads((out / f"eval{suffix}_tiny.json").read_text())
    assert res["test_seed"] == (12345 if test_seed is None else test_seed)
    assert res["seed"] == 9 and res["ckpt"] == "last" and res["n_val"] == 4
    assert res["sets"]["main"]["seed"] == res["test_seed"]
    assert res["provenance"]["files"] == provenance.source_hashes()["files"]
    assert res["ckpt_provenance"] == ck.get("provenance")
    assert ("WARNING: evaluation source" in capsys.readouterr().out) == with_provenance
    assert len(evaluate.load_model(run, "last", "cpu")) == 3
    # Reproduce the old metric reduction exactly, including the non-full final batch.
    cache = train.val_cache(train.env_cfg_from_args(args, split="val"), 4, T_obs=3, seed=res["test_seed"])
    mse = []
    for i in range(0, 4, 3):
        fr = cache["frames"][i:i + 3].float()
        pred, _ = evaluate.rollout(model, fr, cache["actions_noisy"][i:i + 3], cache["cam"][i:i + 3, 0],
                                   3, 150, 1.0, 3, False, "cpu", aux=True)
        mse.append(((pred.clamp(0, 1) - fr[:, 3:]) ** 2).mean((2, 3, 4)))
    for horizon in (20, 150):
        values = res[f"per_seq_mse{horizon}"]
        assert len(values) == 4 and all(isinstance(v, float) for v in values)
        assert np.mean(values) == pytest.approx(res[f"mse{horizon}"], abs=1e-6)
        old_mean, old_ci = evaluate.ci95(torch.cat(mse)[:, :horizon].mean(1).numpy())
        assert res[f"mse{horizon}"] == old_mean
        assert res[f"mse{horizon}_ci"] == old_ci
    if test_seed == 777:
        assert res["sets"]["fractional_time"]["seed"] == 12345
        assert res["sets"]["throughput"]["seed"] == 777
        for key in ("equiv", "equiv_output"):
            assert res["sets"][key]["seed"] == 777
            assert res["sets"][key]["sequence_seed"] == "seed * 1000 + chunk_start"
        evaluate.main(flags + ["--only_equiv"])
        merged = json.loads((out / "eval_seed777_tiny.json").read_text())
        assert merged["per_seq_mse20"] == res["per_seq_mse20"]
        assert merged["sets"] == res["sets"]
        assert merged["provenance"] == res["provenance"]
        assert merged["diagnostic_provenance"]["equiv_output"]["ckpt_provenance"] == ck["provenance"]


def test_training_checkpoint_provenance(tiny_data):
    out = tiny_data / "training"
    train.main(["--model", "spectral", "--C", "2", "--world", "28", "--window", "12", "--node_spacing", "4",
                "--t_obs", "3", "--t_pred", "2", "--n_val", "4", "--val150_n", "1", "--bs", "2", "--steps", "1",
                "--device", "cpu", "--out", str(out), "--keep_ckpts"])
    config = json.loads((out / "config.json").read_text())
    for name in ("last.pt", "best.pt", "best150.pt", "ckpt_step1.pt"):
        ck = torch.load(out / name, map_location="cpu", weights_only=False)
        assert ck["provenance"] == config["provenance"]
        assert ck["provenance"]["files"] == provenance.source_hashes()["files"]
        assert ck["provenance"]["env"] == provenance.env_fingerprint(config["env"])


@pytest.mark.parametrize("t,df,p", [(0, 4, 1), (1, 1, 0.5), (2.776, 4, 0.050022623107),
                                    (2.7764451051977987, 4, 0.05), (2.2281388519649385, 10, 0.05),
                                    (3.16927267261695, 10, 0.01), (1.9839715184496334, 100, 0.05)])
def test_student_t_known_values(t, df, p):
    assert report.student_t_pvalue(t, df) == pytest.approx(p, abs=1e-6)
    assert report.student_t_pvalue(-t, df) == pytest.approx(p, abs=1e-6)
    assert report.student_t_cdf(t, df) == pytest.approx(1 - p / 2, abs=1e-6)
    assert report.student_t_cdf(-t, df) == pytest.approx(p / 2, abs=1e-6)


def test_student_t_analytic_tails():
    for t in (0.001, 0.5, 2, 10, 100):
        # Closed forms for the Cauchy distribution (df=1) and Student-t with df=2.
        assert report.student_t_cdf(t, 1) == pytest.approx(0.5 + math.atan(t) / math.pi, abs=1e-10)
        assert report.student_t_cdf(t, 2) == pytest.approx(0.5 + t / (2 * math.sqrt(t * t + 2)), abs=1e-10)
    assert report.paired_t([0, 0]) == (0, 1)
    assert report.paired_t([1, 1]) == (math.inf, 0)
    assert report.paired_t([1]) == (None, None)
    assert report.sign_flip_pvalue([0, 0]) == 1
    assert report.sign_flip_pvalue([1, -1]) == 1


def _synthetic_arms():
    left = [dict(seed=i, mse20=10.0, mse150=20.0, per_seq_mse20=[9.0, 11.0]) for i in range(5)]
    right = [dict(seed=i, mse20=11.0 + i, mse150=20.0 - i, per_seq_mse20=[10.0 + i, 12.0 + i]) for i in range(5)]
    return dict(lk=left, lock=list(reversed(right)))


def test_report_statistics_and_holm():
    result = report.build_report(_synthetic_arms(), ["lk:lock"], holm=True)
    first, second = result["comparisons"]
    assert first["differences"] == [1, 2, 3, 4, 5]
    assert first["t"] == pytest.approx(3 / (math.sqrt(2.5) / math.sqrt(5)))
    # df=4 has CDF 1/2 + 3*y/4 - y**3/4, where y=t/sqrt(t**2+4).
    y = first["t"] / math.sqrt(first["t"] ** 2 + 4)
    assert first["t_p"] == pytest.approx(1 - 1.5 * y + 0.5 * y ** 3, abs=1e-10)
    assert first["sign_flip_p"] == 2 / 32
    assert first["bootstrap_sequence_seeds"] == 5
    assert second["bootstrap_sequence_seeds"] == 0
    assert first["t_p_holm"] == pytest.approx(first["t_p"] * 2)
    assert second["t_p_holm"] >= first["t_p_holm"]
    assert result["metrics"]["mse20"]["lock"]["sd"] == pytest.approx(math.sqrt(2.5))
    assert result == report.build_report(_synthetic_arms(), ["lk:lock"], holm=True)
    assert report.holm_adjust([0.04, 0.01, 0.03, 0.002]) == pytest.approx([0.06, 0.03, 0.06, 0.008])
    assert report.holm_adjust([None, 0.02]) == [None, 0.02]


def test_bootstrap_resamples_sequences_as_pairs():
    # Identical sequences in the two arms must cancel for every bootstrap draw.
    records = [dict(seed=i, metric=50, per_seq_metric=[0, 100]) for i in range(3)]
    interval, n = report.bootstrap_interval(records, records, "metric")
    assert interval == [0, 0] and n == 3
    left = [dict(seed=0, metric=50, per_seq_metric=[0, 100])]
    right = [dict(seed=0, metric=50, per_seq_metric=[50, 50])]
    interval, n = report.bootstrap_interval(left, right, "metric")
    assert interval == [-50, 50] and n == 1


def test_report_pairing_and_validation():
    with pytest.warns(UserWarning, match="file order"):
        left, right, mode = report.pair_records([dict(mse20=1)], [dict(mse20=2)])
    assert mode == "file_order" and right[0]["mse20"] - left[0]["mse20"] == 1
    with pytest.raises(ValueError, match="same seeds"):
        report.pair_records([dict(seed=1)], [dict(seed=2)])
    with pytest.raises(ValueError, match="duplicate seed"):
        report.pair_records([dict(seed=1), dict(seed=1)], [dict(seed=1)])
    with pytest.raises(ValueError, match="finite scalar"):
        report.build_report(dict(a=[dict(mse20=[1, 2])]), [], ["mse20"])
    a = dict(seed=0, test_seed=1, mse20=1, per_seq_mse20=[1])
    b = dict(a, test_seed=2)
    with pytest.raises(ValueError, match="different test_seed"):
        report.build_report(dict(a=[a], b=[b]), ["a:b"], ["mse20"])
    a["sets"] = dict(main=dict(t_obs=3))
    b = dict(a, sets=dict(main=dict(t_obs=4)))
    with pytest.raises(ValueError, match="different main set"):
        report.build_report(dict(a=[a], b=[b]), ["a:b"], ["mse20"])


def test_report_cli_manifest_and_globs(tmp_path, capsys):
    manifest = {}
    for arm, records in _synthetic_arms().items():
        manifest[arm] = []
        for record in records:
            name = f"{arm}_{record['seed']}.json"
            (tmp_path / name).write_text(json.dumps(record))
            manifest[arm].append(name)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    table, data = tmp_path / "table.md", tmp_path / "report.json"
    report.main([str(path), "--pairs", "lk:lock", "--holm", "--out", str(table), "--json", str(data)])
    assert capsys.readouterr().out == table.read_text()
    assert "Per-seed" in table.read_text() and "Holm" in table.read_text()
    result = json.loads(data.read_text())
    assert result["comparisons"][0]["sign_flip_p"] == 0.0625
    # Exercise the standalone script, including arbitrary scalar metric selection.
    output = subprocess.check_output([sys.executable, str(Path(report.__file__)),
                                      "--arm", "lk", str(tmp_path / "lk_*.json"),
                                      "--arm", "lock", str(tmp_path / "lock_*.json"),
                                      "--pairs", "lk:lock", "--metrics", "mse20", "--json"], text=True)
    assert json.loads(output)["comparisons"][0]["t_p"] == result["comparisons"][0]["t_p"]
