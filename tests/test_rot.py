"""CPU C4 acceptance: permutations, transport, tied layers, complete recurrence, training."""
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest
import torch

from kwm.gconv import GConvLift, GConvRegular, GConvInvariantOut, MomentProjection
from kwm.lock_nodes import LockState
from kwm.model import KineticWM, build_model
from kwm.rot import (rot_map, rot_vec, rot_regular, rot_nodes, rot_state, rot_episode,
                     rotation_perms, normalized_error)
from kwm.transport import stream
from test_galilean import _blob


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def episode():
    """Six random observed frames of three independently moving blobs."""
    g = torch.Generator().manual_seed(65)
    centers = torch.tensor([[7., 8.], [22., 10.], [15., 25.]]) + torch.rand(3, 2, generator=g)
    velocity = torch.randn(3, 2, generator=g) * .4
    actions = torch.randn(1, 10, 2, generator=g) * .3
    amplitudes = .7 + .3 * torch.rand(6, 3, generator=g)
    obs = []
    for t in range(6):
        positions = centers + t * velocity - actions[0, :t + 1].sum(0)
        obs.append(sum(amplitudes[t, i] * _blob(32, p[0], p[1], s=2.) for i, p in enumerate(positions)))
    return torch.stack(obs)[None, :, None], actions[:, :6], actions[:, 6:]


@pytest.mark.parametrize("k", [1, 2, 3])
def test_action_and_transport(k):
    torch.manual_seed(12)
    for size in (2, 31, 32, 50):
        window = 2 if size == 2 else 32
        x = torch.randn(1, 3, 4, size, size)
        shift = torch.tensor([3, -2])
        rotated_shift = rot_vec(shift, k)
        assert torch.equal(rot_map(torch.roll(x, (-2, 3), (-2, -1)), k, window),
                           torch.roll(rot_map(x, k, window), tuple(rotated_shift.flip(0).tolist()), (-2, -1)))
        assert torch.equal(rot_map(rot_map(x, k, window), -k, window), x)
        if size == window:
            assert torch.equal(rot_map(x, k, window), torch.rot90(x, -k, (-2, -1)))
        nodes, actions = torch.randn(3, 2), torch.randn(1, 2)
        expected = rot_map(stream(x, nodes, actions, c4=True), k, window)
        actual = stream(rot_map(x, k, window), rot_vec(nodes, k), rot_vec(actions, k), c4=True)
        assert normalized_error(expected, actual).max() < 1e-5
    # The observation square is invariant inside the world torus.
    padded = torch.zeros(50, 50)
    padded[:32, :32] = torch.randn(32, 32)
    assert torch.equal(rot_map(padded, k)[:32, :32], rot_map(padded[:32, :32], k))
    assert torch.count_nonzero(rot_map(padded, k)[32:]) == 0


def test_legacy_nyquist_counterexample():
    torch.manual_seed(81)
    x, v, a = torch.randn(1, 2, 4, 50, 50), torch.randn(2, 2), torch.randn(1, 2)
    error = normalized_error(rot_map(stream(x, v, a)), stream(rot_map(x), rot_vec(v), rot_vec(a)))
    assert error.item() > .01


@pytest.mark.parametrize("k", [1, 2, 3])
def test_layers(k):
    torch.manual_seed(17)
    for layer, regular_in, regular_out in ((GConvLift(in_ch=2, F=2), False, True),
                                           (GConvRegular(F_in=2, F_out=3), True, True),
                                           (GConvRegular(2, 3, 1), True, True),
                                           (GConvInvariantOut(F_in=2, out_ch=2), True, False)):
        x = torch.randn(2, 8 if regular_in else 2, 16, 16)
        rin, rout = (rot_regular if regular_in else rot_map), (rot_regular if regular_out else rot_map)
        expected, actual = rout(layer(x), k, 16), layer(rin(x, k, 16))
        assert normalized_error(expected, actual).max() < 1e-6
    layer = MomentProjection(2)
    mass, mean, spread, mx = torch.randn(2, 8, 16, 16), torch.randn(2, 2, 16, 16), torch.rand(2, 1, 16, 16), torch.randn(2, 8, 16, 16)
    expected = rot_regular(layer(torch.cat([mass, mean, spread, mx], 1)), k, 16)
    rm = rot_vec(rot_map(mean, k, 16).movedim(1, -1), k).movedim(-1, 1)
    actual = layer(torch.cat([rot_regular(mass, k, 16), rm, rot_map(spread, k, 16), rot_regular(mx, k, 16)], 1))
    assert normalized_error(expected, actual).max() < 1e-6


def state_errors(expected, actual, C):
    if isinstance(expected, LockState):
        fields = zip(expected, actual)
    else:
        # Separate metadata: its large evidence must not hide feature/offset errors.
        fields = [(expected[:, :, s], actual[:, :, s]) for s in
                  (slice(0, C), slice(C, C + 2), slice(C + 2, C + 3), slice(C + 3, None))]
    return [normalized_error(a, b).max().item() for a, b in fields]


def measure_model(rot, update, policy="flowm", gradients=True):
    obs, ain, aout = episode()
    torch.manual_seed(9)
    model = KineticWM(C=8, rot=rot, node_update=update, readout_rel=True).eval()
    model.dec_input, model.freeze_after = policy, 2
    target = obs[:, -4:].flip(1)
    records, baseline = [], None
    for k in range(4):
        model.zero_grad(set_to_none=True)
        o, ai, ap, _ = rot_episode(obs, ain, aout, k=k)
        with torch.set_grad_enabled(gradients):
            pred, aux = model(o, ai, ap, return_states=[5])
            if gradients:
                (pred - rot_map(target, k)).square().mean().backward()
        grads = {name: p.grad.detach().clone() for name, p in model.named_parameters()} if gradients else {}
        memory = aux["states"][5]
        if k == 0:
            baseline = pred.detach(), memory, grads
            continue
        expected = rot_state(baseline[1], k, model.nodes(), model.C)
        errors = state_errors(expected, memory, model.C)
        if isinstance(memory, LockState):
            metadata = expected.nodes[..., :2], memory.nodes[..., :2]
            errors.append(normalized_error(*metadata).max().item())
            assert rot == "none" or torch.equal(expected.nodes[..., 3], memory.nodes[..., 3])
        # The velocity action must include node relabelling as well as vector rotation.
        source = rotation_perms(model.nodes())[(-k) % 4]
        velocities = rot_nodes(model.node_velocity(baseline[1])[..., 0, 0][:, source], k)
        errors.append(normalized_error(velocities, model.node_velocity(memory)[..., 0, 0]).max().item())
        # Relative L2 of the whole parameter gradient, analogous to the memory
        # metric. Also check each tensor, allowing fp32 absolute roundoff when
        # a scalar derivative happens to be close to zero.
        grad_error = 0.
        if gradients:
            grad_error = normalized_error(torch.cat([g.flatten() for g in baseline[2].values()])[None],
                                          torch.cat([g.flatten() for g in grads.values()])[None]).item()
            if rot == "c4":
                for name, g in grads.items():
                    torch.testing.assert_close(g, baseline[2][name], rtol=1e-5, atol=1e-7)
        records.append(dict(k=k, memory=max(errors), output=normalized_error(rot_map(baseline[0], k), pred).max().item(),
                            gradients=grad_error))
    return records


@pytest.mark.parametrize("update", ["lk", "lock"])
@pytest.mark.parametrize("policy", ["flowm", "zero", "zero_then_frozen"])
def test_model_equivariance(update, policy):
    records = measure_model("c4", update, policy, gradients=policy != "zero_then_frozen")
    print(update, policy, records)
    assert all(max(r[key] for key in ("memory", "output", "gradients")) < 1e-5 for r in records)


def test_none_is_not_equivariant():
    records = measure_model("none", "lock")
    print("none", records)
    assert all(min(r[key] for key in ("memory", "output", "gradients")) > .01 for r in records)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@torch.no_grad()
def test_weak_memory_readout(seed):
    """Near-empty blob tails must not turn FFT noise into a velocity signal."""
    torch.manual_seed(seed)
    model = KineticWM(C=8, rot="c4", node_update="lock").eval()
    obs, ain, aout = episode()
    pred = model(obs, ain, aout)
    robs, rain, raout, _ = rot_episode(obs, ain, aout)
    actual = model(robs, rain, raout)
    assert normalized_error(rot_map(pred), actual).max() < 1e-5


@pytest.mark.parametrize("update", ["lk", "lock"])
def test_c4_checkpoint_frozen_nodes(update):
    torch.manual_seed(42)
    obs, ain, aout = torch.rand(1, 3, 1, 8, 8), torch.randn(1, 3, 2), torch.randn(1, 2, 2)
    results = []
    for checkpoint in (False, True):
        torch.manual_seed(43)
        model = KineticWM(C=4, world=16, window=8, rot="c4", node_update=update,
                          node_update_pred=False, grad_ckpt=checkpoint).train()
        pred = model(obs, ain, aout)
        pred.square().mean().backward()
        results.append((pred, [p.grad for p in model.parameters()]))
    assert torch.equal(results[0][0], results[1][0])
    assert all(torch.equal(a, b) for a, b in zip(results[0][1], results[1][1]))


@torch.no_grad()
def test_lock_search_roundoff_on_digits():
    """Norm/dot SSD cancellation used to change a mode in held-out sequence 22."""
    from kwm.env import EnvConfig, make_batch
    batch = make_batch(32, 6, 150, EnvConfig(real_velocity=True, real_camera=True),
                       gen=torch.Generator().manual_seed(777), split="test")
    obs, ain, aout = batch["frames"][20:24, :6], batch["actions"][20:24, :6], batch["actions"][20:24, 6:10]
    torch.manual_seed(9)
    model = KineticWM(C=8, rot="c4", node_update="lock", readout_rel=True).eval()
    pred = model(obs, ain, aout)
    robs, rain, raout, _ = rot_episode(obs, ain, aout)
    actual = model(robs, rain, raout)
    assert normalized_error(rot_map(pred), actual).max() < 1e-5


def _reference_model():
    """Load the three byte-for-byte snapshots in an isolated relative-import package."""
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("_pre_rot")
    package.__path__ = [str(root / "kwm")]
    sys.modules[package.__name__] = package
    for name in ("lock_nodes", "cells", "model"):
        spec = importlib.util.spec_from_file_location(f"_pre_rot.{name}", root / "tests/reference" / f"{name}_pre_rot.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules["_pre_rot.model"].KineticWM


@pytest.mark.parametrize("kind,update,amp,checkpoint", [
    ("flowm", "lk", False, False), ("spectral", "lk", False, True),
    ("adaptive", "lk", False, False), ("adaptive", "lock", False, True),
    ("adaptive", "lk", True, True), ("adaptive", "lock", True, True),
])
def test_none_bit_identical(kind, update, amp, checkpoint):
    reference = _reference_model()
    models = []
    for cls in (reference, KineticWM):
        torch.manual_seed(23)
        models.append(cls(kind=kind, C=4, world=16, window=8, node_update=update, grad_ckpt=checkpoint))
    old, new = models
    assert old.state_dict().keys() == new.state_dict().keys()
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())
    obs, ain, aout = torch.rand(2, 4, 1, 8, 8), torch.randn(2, 4, 2), torch.randn(2, 2, 2)
    results = []
    for model in models:
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
            pred, aux = model(obs, ain, aout, return_states=[3])
            loss = (pred.float() - .3).square().mean()
        loss.backward()
        results.append((pred, aux["states"][3]))
    assert torch.equal(results[0][0], results[1][0])
    if update == "lock" and kind == "adaptive":
        assert all(torch.equal(a, b) for a, b in zip(results[0][1], results[1][1]))
    else:
        assert torch.equal(results[0][1], results[1][1])
    for (name, p), (_, q) in zip(old.named_parameters(), new.named_parameters()):
        assert torch.equal(p.grad, q.grad), name


def test_flags_projection_episode_and_eval(tmp_path):
    from kwm.train import parse_args
    from kwm.eval import load_model, rotation_check
    assert parse_args([]).rot == "none"
    torch.manual_seed(0)
    args = vars(parse_args(["--rot", "c4", "--C", "8", "--node_update", "lock"]))
    model = build_model(args).eval()
    torch.save(dict(args=args, model=model.state_dict(), step=0), tmp_path / "best.pt")
    loaded, _, _ = load_model(tmp_path, "best", "cpu")
    assert loaded.rot == "c4"
    with pytest.raises(ValueError, match="divisible"):
        KineticWM(C=3, rot="c4")
    obs, ain, aout = episode()
    state = dict(cent=torch.tensor([[[15., 14.], [40., 40.]]]), vel=torch.tensor([[[.2, -.7], [1., 2.]]]))
    *_, rotated = rot_episode(obs, ain, aout, state, 1)
    assert torch.equal(rotated["cent"], torch.tensor([[[17., 15.], [41., 40.]]]))
    assert torch.equal(rotated["vel"], rot_vec(state["vel"]))
    for k in (1, 2, 3):
        u = model.cell.conv_u(obs[:, 0])
        assert normalized_error(model.cell.imprint(rot_regular(u, k)), rot_regular(model.cell.imprint(u), k)).max() < 1e-6
    cache = dict(frames=torch.cat([obs, obs[:, :4]], 1), actions_noisy=torch.cat([ain, aout], 1))
    assert rotation_check(loaded, cache, 6, "cpu") < 1e-5


def test_eval_rot_check_cli(tmp_path, monkeypatch):
    import kwm.eval as evaluate
    from kwm.train import parse_args
    torch.manual_seed(24)
    args = vars(parse_args(["--model", "spectral", "--rot", "c4", "--C", "4", "--world", "16",
                           "--window", "8", "--t_obs", "3"]))
    model = build_model(args).eval()
    torch.save(dict(args=args, model=model.state_dict(), step=0), tmp_path / "best.pt")
    requests = []

    def cache(cfg, n, T_obs, device, seed):
        requests.append((n, seed))
        return dict(frames=torch.rand(n, 5, 1, 8, 8), actions_noisy=torch.randn(n, 5, 2))

    monkeypatch.setattr(evaluate, "val_cache", cache)
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(dict(existing_metric=42, rot_equiv_error=99)))
    evaluate.main(["--run", str(tmp_path), "--out", str(tmp_path), "--n_val", "2", "--device", "cpu",
                   "--rot_check", "--only_equiv"])
    result = json.loads(path.read_text())
    assert requests[-1] == (32, 777)
    assert result["existing_metric"] == 42 and result["rot_equiv_error"] < 1e-5
    assert "rot_equiv_error" in result["diagnostic_provenance"]
    assert result["diagnostic_provenance"]["rot_equiv_error"]["sets"]["rot_equiv_error"]["role"] == "heldout"
