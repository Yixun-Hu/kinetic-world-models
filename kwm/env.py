"""Kinetic MNIST World: online, pure-torch generation of FloWM's 2D MNIST World with continuous switches.

Coordinates: x = column (W axis), y = row (H axis); 2-vectors are (x, y).
World is a WxW canvas (default 50). The camera window is the win x win
(default 32) patch whose *top-left* corner is at the real camera position cam;
the world is periodic for the camera (torus), exactly like FloWM.

Switches (EnvConfig):
  real_velocity  real-valued digit velocities v ~ U[-vmax, vmax]^2 (else integers)
  real_camera    real-valued camera motion    a ~ U[-amax, amax]^2 (else integers)
  walls          reflecting walls: 1-px gray (0.5) line on the world border rows/cols
      0 and W-1; a digit's ink-bbox *centroid* is confined to [1, W-2] and its
      velocity component flips sign (elastic) when its ink bbox touches the wall
      (bounce_rule='bbox', default: the bbox edge is confined to [1, W-2]);
      bounce_rule='centroid' confines the bbox centroid
      instead, and the ink beyond the wall is hidden.
      The camera still wraps around the (tiled) walled box.
      walls=False: digits wrap on the torus (original FloWM).
  odometry_noise / odometry_bias: actions_noisy = (1 + bias) * a + noise * eps
  dt  time step per frame (fractional-time evaluation uses dt=0.5); the ego
      velocity is constant within each unit time interval, so a dt=0.5
      sequence subsampled at even frames equals the dt=1 trajectory.

Rendering: every digit (28x28 at the patch top-left) is placed at its real
world position by a Fourier phase shift of a zero-padded WxW canvas (exact for
integer positions = torch.roll), digits are combined with max, masked to the
wall interior (walls), walls drawn, then the whole world is phase-shifted by -cam
and the top-left win x win crop is the observation; clamped to [0, 1].
This deliberately does not use kwm.transport (independent code path).
"""
import math
import os
from dataclasses import dataclass, asdict

import torch

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@dataclass
class EnvConfig:
    world: int = 50
    window: int = 32
    n_digits: int = 5
    real_velocity: bool = False
    real_camera: bool = False
    walls: bool = False
    odometry_noise: float = 0.0
    odometry_bias: float = 0.0
    vmax: float = 2.0
    amax: float = 10.0
    dt: float = 1.0
    v_ring: float = 0.0             # >0: sample velocities with v_ring < ||v||_inf <= vmax (unseen-speed evaluation)
    scene_boost: float = 0.0        # >0: add one per-sequence boost u ~ U[-b, b]^2 to every digit
    bounce_rule: str = "bbox"       # 'bbox' (default) | 'centroid'

    def tag(self):
        s = f"w{self.world}_vel{int(self.real_velocity)}_cam{int(self.real_camera)}_walls{int(self.walls)}"
        if self.odometry_noise or self.odometry_bias:
            s += f"_odo{self.odometry_noise:g}-{self.odometry_bias:g}"
        if self.dt != 1.0:
            s += f"_dt{self.dt:g}"
        if self.walls and self.bounce_rule != "bbox":
            s += f"_{self.bounce_rule}"
        if self.vmax != 2.0:
            s += f"_vmax{self.vmax:g}"
        if self.v_ring:
            s += f"_ring{self.v_ring:g}"
        if self.scene_boost:
            s += f"_sb{self.scene_boost:g}"
        return s

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- MNIST ----
_MNIST = {}


def load_mnist(split="train", root=DATA_ROOT):
    """Returns digits [N,28,28] float in [0,1], centroid offsets [N,2] (x,y), half-size [N,2]."""
    key = (split, root)
    if key not in _MNIST:
        from torchvision.datasets import MNIST
        ds = MNIST(root, train=(split == "train"), download=True)
        d = ds.data.float() / 255.0                                       # [N,28,28]
        ink = d > 0.1
        rows = ink.any(dim=2)                                               # [N,28]
        cols = ink.any(dim=1)
        ar = torch.arange(28).float()
        big = 1e3
        ymin = torch.where(rows, ar, big).amin(1)
        ymax = torch.where(rows, ar, -big).amax(1)
        xmin = torch.where(cols, ar, big).amin(1)
        xmax = torch.where(cols, ar, -big).amax(1)
        off = torch.stack([(xmin + xmax) / 2, (ymin + ymax) / 2], 1)       # half-integers
        half = torch.stack([(xmax - xmin) / 2, (ymax - ymin) / 2], 1)
        _MNIST[key] = (d, off, half)
    return _MNIST[key]


# ------------------------------------------------------------- helpers ----
def _fourier_shift(img, shift):
    """img [..., H, W], shift [..., 2] (dx, dy) broadcast over leading dims."""
    H, W = img.shape[-2:]
    ky = torch.fft.fftfreq(H, device=img.device)[:, None]
    kx = torch.fft.rfftfreq(W, device=img.device)[None, :]
    ang = -2 * math.pi * (kx * shift[..., 0, None, None] + ky * shift[..., 1, None, None])
    return torch.fft.irfft2(torch.fft.rfft2(img) * torch.polar(torch.ones_like(ang), ang), s=(H, W))


def bilinear_shift(img, shift):
    """Periodic bilinear shift (used only in tests as an independent reference)."""
    H, W = img.shape[-2:]
    dx, dy = float(shift[0]), float(shift[1])
    ix, iy = math.floor(dx), math.floor(dy)
    fx, fy = dx - ix, dy - iy
    r = lambda a, sy, sx: torch.roll(a, shifts=(sy, sx), dims=(-2, -1))
    return ((1 - fx) * (1 - fy) * r(img, iy, ix) + fx * (1 - fy) * r(img, iy, ix + 1)
            + (1 - fx) * fy * r(img, iy + 1, ix) + fx * fy * r(img, iy + 1, ix + 1))


def _uniform(gen, shape, lo, hi):
    return lo + (hi - lo) * torch.rand(shape, generator=gen)


def _randint(gen, lo, hi, shape):
    """Integers in [lo, hi] inclusive; lo/hi may be tensors broadcast to shape."""
    lo = torch.as_tensor(lo, dtype=torch.float32)
    hi = torch.as_tensor(hi, dtype=torch.float32)
    u = torch.rand(shape, generator=gen)
    return torch.floor(lo + u * (hi - lo + 1)).clamp(max=hi)


def in_view_mask(cent, half, cam, world, win):
    """Does the digit ink bbox overlap the camera window (torus)? cent/half/cam [...,2] -> [...] bool."""
    rel = torch.remainder(cent - cam, world)
    ov = (rel - half < win) | (rel + half > world)
    return ov.all(-1)


# ---------------------------------------------------------- generation ----
@torch.no_grad()
def make_batch(B, T_obs, T_pred, cfg: EnvConfig, device="cpu", gen=None, split="train", boost=None, return_world=False):
    """Generate B sequences of T = T_obs + T_pred frames.

    Returns dict:
      frames         [B,T,1,win,win] float in [0,1]
      actions        [B,T,2] true camera displacement cam_t - cam_{t-1} (actions[:,0]=0)
      actions_noisy  [B,T,2] what the model receives
      bounce         [B,T,n_digits] bool, digit reflected while moving into frame t
      bounce_events  [E,4] long (b, digit, t, in_view)
      digit_states   dict: cent [B,T,nd,2] ink-bbox centroid (world, wrapped),
                     vel [B,T,nd,2] velocity used to move from t to t+1,
                     half [B,nd,2], cam [B,T,2] camera top-left (wrapped),
                     visible [B,T,nd] bool
      cfg            cfg.to_dict()
      scene_boost    [B,2] the per-sequence boost actually added (zeros if none)
    boost: optional [B,2] (or [2]) velocity added to every digit after all random draws, so the same
    `gen` state renders the identical scene with boosted velocities (equivariance tests).
    return_world: also return world [B,T,1,W,W], the full world canvas before the camera crop (visualisation).
    All randomness comes from `gen` (CPU torch.Generator) -> reproducible.
    """
    if gen is None:
        gen = torch.Generator().manual_seed(0)
    T = T_obs + T_pred
    W, win, nd, dt = cfg.world, cfg.window, cfg.n_digits, cfg.dt
    digits, offs, halfs = load_mnist(split)
    idx = torch.randint(0, digits.shape[0], (B, nd), generator=gen)
    patch, off, half = digits[idx], offs[idx], halfs[idx]                  # [B,nd,28,28], [B,nd,2]

    # velocities (optionally restricted to the ring v_ring < ||v||_inf <= vmax by rejection)
    def draw_v(shape):
        if cfg.real_velocity:
            return _uniform(gen, shape, -cfg.vmax, cfg.vmax)
        return _randint(gen, -cfg.vmax, cfg.vmax, shape)
    vel = draw_v((B, nd, 2))
    if cfg.v_ring:
        assert cfg.v_ring < cfg.vmax
        bad = vel.abs().amax(-1) <= cfg.v_ring                              # [B,nd]
        while bad.any():
            vel[bad] = draw_v((int(bad.sum()), 2))
            bad = vel.abs().amax(-1) <= cfg.v_ring

    # bounds for the reflected coordinate q (= centroid for 'centroid', bbox edge-aware for 'bbox')
    if cfg.walls:
        lo = torch.full((B, nd, 2), 1.0)
        hi = torch.full((B, nd, 2), W - 2.0)
        if cfg.bounce_rule == "bbox":
            lo = lo + half
            hi = hi - half
        else:
            assert cfg.bounce_rule == "centroid"
        if cfg.real_velocity:
            cent = lo + (hi - lo) * torch.rand((B, nd, 2), generator=gen)
        else:
            # patch top-left must be an integer -> centroid = p + off
            p = _randint(gen, torch.ceil(lo - off), torch.floor(hi - off), (B, nd, 2))
            cent = p + off
    else:
        if cfg.real_velocity:
            cent = _uniform(gen, (B, nd, 2), 0, W) + off
        else:
            cent = _randint(gen, 0, W - 1, (B, nd, 2)) + off
        lo = hi = None

    # camera: start + per-unit-interval ego velocity u_j
    n_units = int(math.ceil((T - 1) * dt)) + 2
    if cfg.real_camera:
        cam0 = _uniform(gen, (B, 2), 0, W)
        u = _uniform(gen, (B, n_units, 2), -cfg.amax, cfg.amax)
    else:
        cam0 = _randint(gen, 0, W - 1, (B, 2))
        u = _randint(gen, -cfg.amax, cfg.amax, (B, n_units, 2))
    cu = torch.cat([torch.zeros(B, 1, 2), torch.cumsum(u, 1)], 1)          # cam offset at integer times
    tt = torch.arange(T).float() * dt
    j = torch.floor(tt + 1e-9).long()
    fr = (tt - j.float())[None, :, None]
    cam_unw = cam0[:, None] + cu[:, j] + fr * u[:, j]                      # [B,T,2]
    actions = torch.zeros(B, T, 2)
    actions[:, 1:] = cam_unw[:, 1:] - cam_unw[:, :-1]
    cam = torch.remainder(cam_unw, W)

    noisy = (1 + cfg.odometry_bias) * actions + cfg.odometry_noise * torch.randn((B, T, 2), generator=gen)
    noisy[:, 0] = 0

    # boosts are applied after every other random draw (the rest of the scene is unchanged)
    sb = torch.zeros(B, 2)
    if cfg.scene_boost:
        sb = _uniform(gen, (B, 2), -cfg.scene_boost, cfg.scene_boost)
    if boost is not None:
        sb = sb + torch.as_tensor(boost, dtype=torch.float32).expand(B, 2)
    vel = vel + sb[:, None]

    # digit trajectories with elastic reflection
    cents = torch.zeros(B, T, nd, 2)
    vels = torch.zeros(B, T, nd, 2)
    bounce = torch.zeros(B, T, nd, dtype=torch.bool)
    c, v = cent.clone(), vel.clone()
    for t in range(T):
        if t > 0:
            c = c + v * dt
            if cfg.walls:
                below, above = c < lo, c > hi
                c = torch.where(below, 2 * lo - c, c)
                c = torch.where(above, 2 * hi - c, c)
                flip = below | above
                v = torch.where(flip, -v, v)
                bounce[:, t] = flip.any(-1)
            else:
                c = torch.remainder(c, W)
        cents[:, t] = c
        vels[:, t] = v

    visible = in_view_mask(cents, half[:, None], cam[:, :, None], W, win)  # [B,T,nd]
    ev = bounce.nonzero()                                                  # [E,3] (b,t,d)
    bounce_events = torch.stack([ev[:, 0], ev[:, 2], ev[:, 1],
                                 visible[ev[:, 0], ev[:, 1], ev[:, 2]].long()], 1) if len(ev) else torch.zeros(0, 4, dtype=torch.long)

    # ---- rendering on device ----
    patch = patch.to(device)
    canvas = torch.zeros(B, nd, W, W, device=device)
    canvas[:, :, :28, :28] = patch
    Fc = torch.fft.rfft2(canvas)                                           # [B,nd,W,W//2+1]
    ky = torch.fft.fftfreq(W, device=device)[:, None]
    kx = torch.fft.rfftfreq(W, device=device)[None, :]
    if cfg.walls:
        interior = torch.zeros(W, W, device=device)
        interior[1:W - 1, 1:W - 1] = 1
        wall = torch.zeros(W, W, device=device)
        wall[0, :] = wall[-1, :] = wall[:, 0] = wall[:, -1] = 0.5
    pos = (cents - off[:, None]).to(device)                                # patch top-left [B,T,nd,2]
    camd = cam.to(device)
    frames = torch.zeros(B, T, 1, win, win, device=device)
    worlds = torch.zeros(B, T, 1, W, W, device=device) if return_world else None
    for t in range(T):
        p = pos[:, t]
        ang = -2 * math.pi * (kx * p[..., 0, None, None] + ky * p[..., 1, None, None])
        img = torch.fft.irfft2(Fc * torch.polar(torch.ones_like(ang), ang), s=(W, W))  # [B,nd,W,W]
        world = img.max(dim=1).values                                       # [B,W,W]
        if cfg.walls:
            world = torch.maximum(world * interior, wall)
        if return_world:
            worlds[:, t, 0] = world
        view = _fourier_shift(world, -camd[:, t])
        frames[:, t, 0] = view[:, :win, :win]
    frames = frames.clamp(0, 1)
    extra = dict(world=worlds.clamp(0, 1)) if return_world else {}

    return dict(**extra,
        frames=frames,
        actions=actions.to(device),
        actions_noisy=noisy.to(device),
        bounce=bounce.to(device),
        bounce_events=bounce_events,
        digit_states=dict(cent=cents.to(device), vel=vels.to(device), half=half.to(device),
                          cam=camd, visible=visible.to(device)),
        cfg=cfg.to_dict(),
        scene_boost=sb.to(device),
    )


def subsample(batch, k):
    """Take every k-th frame of a batch (actions are summed over the skipped interval)."""
    out = dict(batch)
    out["frames"] = batch["frames"][:, ::k]
    for key in ("actions", "actions_noisy"):
        a = batch[key]
        cum = torch.cumsum(a, 1)[:, ::k]
        na = torch.zeros_like(cum)
        na[:, 1:] = cum[:, 1:] - cum[:, :-1]
        out[key] = na
    ds = batch["digit_states"]
    out["digit_states"] = dict(cent=ds["cent"][:, ::k], vel=ds["vel"][:, ::k], half=ds["half"],
                               cam=ds["cam"][:, ::k], visible=ds["visible"][:, ::k])
    T = batch["frames"].shape[1]
    b = batch["bounce"]
    # a bounce in (t-k, t] is attributed to subsampled frame t/k
    bnc = torch.zeros_like(b[:, ::k])
    for i, t in enumerate(range(0, T, k)):
        if t > 0:
            bnc[:, i] = b[:, t - k + 1:t + 1].any(1)
    out["bounce"] = bnc
    ev = batch["bounce_events"]
    if len(ev):
        tn = (ev[:, 2] + k - 1) // k
        keep = tn * k < T
        ev = torch.stack([ev[:, 0], ev[:, 1], tn, ev[:, 3]], 1)[keep]
    out["bounce_events"] = ev
    return out
