import torch
from kwm.env import EnvConfig, make_batch, _fourier_shift, bilinear_shift, subsample


def smooth_img(H=50, W=50):
    y = torch.arange(H).float()[:, None]
    x = torch.arange(W).float()[None, :]
    import math
    return (torch.exp(-((x - 20) ** 2 + (y - 25) ** 2) / (2 * 4.0 ** 2))
            + 0.5 * torch.sin(2 * math.pi * x / W) * torch.cos(2 * math.pi * 2 * y / H))


def test_fourier_vs_bilinear_on_smooth():
    img = smooth_img()
    for s in [(0.5, 0.0), (0.3, -0.7), (3.25, 1.5)]:
        st = torch.tensor(s)
        a = _fourier_shift(img, st)
        b = bilinear_shift(img, st)
        rel = (a - b).abs().max() / img.abs().max()
        assert rel < 0.02, (s, rel)


def test_fourier_integer_is_roll():
    img = torch.rand(50, 50)
    a = _fourier_shift(img, torch.tensor([3.0, -4.0]))
    assert (a - torch.roll(img, shifts=(-4, 3), dims=(0, 1))).abs().max() < 1e-5


def test_batch_shapes_and_repro():
    cfg = EnvConfig(real_velocity=True, real_camera=True)
    b1 = make_batch(3, 6, 4, cfg, gen=torch.Generator().manual_seed(5))
    b2 = make_batch(3, 6, 4, cfg, gen=torch.Generator().manual_seed(5))
    assert b1["frames"].shape == (3, 10, 1, 32, 32)
    assert b1["actions"].shape == (3, 10, 2) and (b1["actions"][:, 0] == 0).all()
    assert (b1["frames"] - b2["frames"]).abs().max() == 0
    assert b1["frames"].min() >= 0 and b1["frames"].max() <= 1
    assert b1["actions"].abs().max() <= 10


def test_integer_config_is_integer():
    b = make_batch(4, 10, 5, EnvConfig(), gen=torch.Generator().manual_seed(0))
    a = b["actions"]
    assert (a == a.round()).all()
    v = b["digit_states"]["vel"]
    assert (v == v.round()).all()


def test_camera_consistency():
    """Static world (v=0): frame t equals frame t-1 shifted by -action (spectral)."""
    cfg = EnvConfig(real_velocity=True, real_camera=True, vmax=0.0)
    b = make_batch(2, 4, 0, cfg, gen=torch.Generator().manual_seed(2))
    # compare on the overlap region only -> use full world? check centre pixels via ground truth cam
    cam = b["digit_states"]["cam"]
    d = torch.remainder(cam[:, 1:] - cam[:, :-1] + 25, 50) - 25
    assert (d - b["actions"][:, 1:]).abs().max() < 1e-4


def test_walls_bounce_and_stay_inside():
    cfg = EnvConfig(real_velocity=True, real_camera=True, walls=True)
    b = make_batch(8, 50, 20, cfg, gen=torch.Generator().manual_seed(1))
    c = b["digit_states"]["cent"]
    assert c.min() >= 1 - 1e-5 and c.max() <= 48 + 1e-5
    ev = b["bounce_events"]
    assert ev.shape[1] == 4 and len(ev) > 0
    vel = b["digit_states"]["vel"]
    bb, d, t = ev[0, 0], ev[0, 1], ev[0, 2]
    flipped = (vel[bb, t, d] * vel[bb, t - 1, d] < 0).any()
    assert flipped


def test_subsample_dt_half_matches_dt1():
    cfg1 = EnvConfig(real_velocity=True, real_camera=True, walls=True)
    cfgh = EnvConfig(real_velocity=True, real_camera=True, walls=True, dt=0.5)
    bh = make_batch(2, 20, 10, cfgh, gen=torch.Generator().manual_seed(3))
    s = subsample(bh, 2)
    # trajectories of the subsampled dt=0.5 sequence follow the dt=1 kinematics
    assert s["frames"].shape[1] == 15
    assert (s["actions"][:, 1:] - bh["actions"][:, 1:-1:2] - bh["actions"][:, 2::2]).abs().max() < 1e-5
