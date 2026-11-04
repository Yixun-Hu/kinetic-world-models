"""Named lattices, static metadata pinning, and their existing C4 action."""
import pytest
import torch

from kwm.collision import named_nodes, reflection_perms
from kwm.env import EnvConfig, make_batch
from kwm.lock_nodes import LockState, lock_measurement
from kwm.model import KineticWM, build_model
from kwm.rot import rot_regular, rot_state, rot_vec, rotation_perms
from kwm.train import parse_args
from kwm.transport import spectral_shift


@pytest.mark.parametrize("name,count", [("v17", 17), ("v9", 9), ("v5", 5)])
def test_named_lattice(name, count):
    nodes = named_nodes(name)
    assert nodes.shape == (count, 2) and nodes.dtype == torch.float32
    assert torch.equal(nodes[0], torch.zeros(2))
    assert len(nodes.unique(dim=0)) == count
    assert torch.equal(nodes, named_nodes(name))
    for k, perm in enumerate(rotation_perms(nodes)):
        assert perm[0] == 0 and len(perm.unique()) == count
        assert torch.equal(nodes[perm], rot_vec(nodes, k))
    for axis, perm in enumerate(reflection_perms(nodes)):
        reflected = nodes.clone()
        reflected[:, axis] *= -1
        assert perm[0] == 0 and len(perm.unique()) == count
        assert torch.equal(nodes[perm], reflected)
    if name == "v17":
        assert set(map(tuple, nodes[1:].tolist())) == {(x, y) for x in (-.75, -.25, .25, .75)
                                                     for y in (-.75, -.25, .25, .75)}
    if name == "v5":
        assert nodes.tolist() == [[0., 0.], [1., 0.], [0., 1.], [-1., 0.], [0., -1.]]


def test_named_options():
    with pytest.raises(ValueError, match="unknown"):
        named_nodes("grid")
    defaults = parse_args([])
    assert defaults.lattice == "grid" and defaults.pin_static is False
    assert parse_args(["--lattice", "v17"]).pin_static is True
    assert parse_args(["--lattice", "v17", "--pin_static", "off"]).pin_static is False
    assert parse_args(["--pin_static"]).pin_static is True
    assert KineticWM(lattice="v17", C=4).cell.pin_static is True
    assert KineticWM(lattice="v17", C=4, pin_static=False).cell.pin_static is False


@pytest.mark.parametrize("name", ["v17", "v9"])
@pytest.mark.parametrize("update", ["lk", "lock"])
@torch.no_grad()
def test_named_2d_env_smoke(name, update):
    torch.manual_seed(21)
    args = parse_args(["--model", "adaptive", "--lattice", name, "--node_update", update, "--C", "4"])
    model = build_model(args).eval()
    batch = make_batch(1, 2, 1, EnvConfig(real_velocity=True, real_camera=True),
                       gen=torch.Generator().manual_seed(37), split="test")
    prediction, aux = model(batch["frames"][:, :2], batch["actions"][:, :2], batch["actions"][:, 2:],
                            record_nodes=True, return_states=[1])
    assert prediction.shape == (1, 1, 1, 32, 32) and torch.isfinite(prediction).all()
    assert torch.equal(model.fixed_nodes, named_nodes(name))
    assert torch.equal(aux["node_traj"][:, :, 0], torch.zeros_like(aux["node_traj"][:, :, 0]))
    state = aux["states"][1]
    metadata = state.nodes[:, 0] if update == "lock" else state[:, 0, model.C:model.C + 3]
    assert torch.count_nonzero(metadata) == 0
    assert torch.count_nonzero(state.features[:, 0] if update == "lock" else state[:, 0, :model.C]) > 0


@pytest.mark.parametrize("update", ["lk", "lock"])
@pytest.mark.parametrize("update_nodes", [False, True])
def test_static_pin_blocks_metadata_gradients(update, update_nodes):
    torch.manual_seed(5)
    model = KineticWM(C=4, world=16, window=8, lattice="v9", node_update=update)
    frame, action = torch.randn(1, 1, 8, 8), torch.tensor([[.3, -.2]])
    state = model.initial_state(frame)
    if update == "lock":
        state.nodes[:, 0] = torch.tensor([.2, -.1, 2., 1.])
        state = state._replace(nodes=state.nodes.requires_grad_(), snapshot=torch.randn_like(state.snapshot))
    else:
        state[:, 0, model.C:model.C + 3] = .2
        state.requires_grad_()
    out = model.cell(frame, state, action, model.nodes(), update_nodes=update_nodes)
    metadata = out.nodes[:, 0] if update == "lock" else out[:, 0, model.C:model.C + 3]
    assert torch.count_nonzero(metadata) == 0
    metadata.sum().backward()
    assert model.cell.n_s.grad is None or model.cell.n_s.grad == 0
    original = state.nodes.grad[:, 0] if update == "lock" else state.grad[:, 0, model.C:model.C + 3]
    assert torch.count_nonzero(original) == 0


@pytest.mark.parametrize("k", [1, 2, 3])
@torch.no_grad()
def test_v17_lock_state_measurement_rotation(k):
    torch.manual_seed(13)
    size, channels = 25, 4
    y, x = torch.meshgrid(torch.arange(size).float(), torch.arange(size).float(), indexing="ij")
    previous = torch.stack([torch.exp(-((x-12)**2 + (y-11)**2)/9) *
                            (1 + .2 * torch.sin(x*.6 + c) * torch.cos(y*.4 - c)) for c in range(channels)])[None]
    nodes = named_nodes("v17")
    metadata = torch.zeros(1, len(nodes), 4)
    metadata[:, 1:, :2] = .03 * torch.randn(1, len(nodes)-1, 2)
    state = LockState(torch.randn(1, len(nodes), channels, size, size), metadata, previous)
    action = torch.tensor([[.11, -.07]])
    current = spectral_shift(previous, torch.tensor([[.3, -.6]]) - action, c4=True)

    def measure(s, image, a):
        return lock_measurement(s.snapshot, image, nodes[None] + s.nodes[..., :2], nodes, a, size, c4=True)

    delta, evidence, locked = measure(state, current, action)
    rotated = rot_state(state, k, nodes, channels, size)
    actual = measure(rotated, rot_regular(current, k, size), rot_vec(action, k))
    source = rotation_perms(nodes)[(-k) % 4]
    expected = rot_vec(delta[:, source], k), evidence[:, source], locked[:, source]
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-4)
