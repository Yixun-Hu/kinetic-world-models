"""Spectral (Fourier-phase) transport on periodic 2D maps.

Conventions (used everywhere in kwm):
  * maps are real tensors [..., H, W]; H = rows = y axis, W = columns = x axis
  * a shift / displacement is a 2-vector (dx, dy) in pixels: content at (x, y)
    moves to (x + dx, y + dy) (periodic). Integer (dx, dy) is exactly
    torch.roll(h, shifts=(dy, dx), dims=(-2, -1)).
"""
import math
import torch


def _phase(H, W, shift, device):
    """Complex phase exp(-2*pi*i*(kx*dx + ky*dy)) of shape shift.shape[:-1] + (H, W//2+1)."""
    ky = torch.fft.fftfreq(H, device=device, dtype=torch.float32)          # [H]
    kx = torch.fft.rfftfreq(W, device=device, dtype=torch.float32)         # [W//2+1]
    dx = shift[..., 0, None, None].float()
    dy = shift[..., 1, None, None].float()
    ang = -2 * math.pi * (kx[None, :] * dx + ky[:, None] * dy)            # [..., H, W//2+1]
    return torch.polar(torch.ones_like(ang), ang)


def spectral_shift(h, shift_xy):
    """Shift real periodic maps by real displacements via an rfft2 phase ramp.

    h:        [..., H, W] real
    shift_xy: [..., 2] (dx, dy); its leading dims must broadcast against
              h.shape[:-2] *from the left*, e.g. h [B,V,C,H,W] with shift [B,2]
              (one shift per batch), [B,V,2] (per batch and velocity node) or
              [V,2] / [2]. Missing trailing dims of h are filled with 1s.
    Differentiable w.r.t. both h and shift_xy. Computed in float32 and cast back.
    """
    H, W = h.shape[-2:]
    lead = h.shape[:-2]
    s = shift_xy
    if s.dim() == 1:
        s = s.expand(*lead, 2) if len(lead) else s
    # left-align: s [B,2] vs lead (B,V,C) -> view s as [B,1,1,2]
    while s.dim() - 1 < len(lead):
        s = s.unsqueeze(-2)
    ph = _phase(H, W, s, h.device)                                         # [b..., H, W//2+1]
    Hf = torch.fft.rfft2(h if h.dtype == torch.float32 else h.float())
    if ph.requires_grad:                       # learnable nodes: keep Hf for d/d(shift)
        Hf = Hf * ph
    else:
        Hf.mul_(ph)
    out = torch.fft.irfft2(Hf, s=(H, W))
    return out.to(h.dtype)


def stream(h, nodes, a, dt=1.0):
    """Kinetic streaming + ego-motion in one phase multiplication.

    h:     [B, V, C, H, W] memory in ego coordinates; channel group i holds the
           population moving with world velocity nodes[i].
    nodes: [V, 2] velocity nodes (vx, vy) in px per unit time (world frame),
           or [B, V, 2] per-sample nodes (adaptive nodes, --oracle_nodes).
    a:     [B, 2] camera displacement for this step (already includes dt).
    dt:    float time step.
    Displacement of node i in batch b:  nodes[(b,) i] * dt - a[b].
    """
    if nodes.dim() == 2:
        nodes = nodes[None]
    disp = nodes * dt - a[:, None, :]                                      # [B, V, 2]
    return spectral_shift(h, disp)


def phase_correlation(x, y, reg=1e-3, newton_iters=10):
    """Estimate the real shift d such that y ~= spectral_shift(x, d).

    x, y: [..., H, W] real. Returns [..., 2] (dx, dy), wrapped to [-W/2, W/2) x [-H/2, H/2).
    1) cross-power spectrum Y X^* with *regularized* whitening |.| + reg*max|.|
       (pure whitening amplifies round-off at frequencies with no power);
    2) integer peak of its inverse FFT;
    3) Newton refinement of d on the band-limited (trigonometric) interpolant
       c(d) = Re sum_k R(k) exp(2 pi i k.d), which is exact for pure translations.
    """
    H, W = x.shape[-2:]
    lead = x.shape[:-2]
    X = torch.fft.fft2(x.float().reshape(-1, H, W))
    Y = torch.fft.fft2(y.float().reshape(-1, H, W))
    R = Y * X.conj()
    mag = R.abs()
    R = R / (mag + reg * mag.amax(dim=(-2, -1), keepdim=True) + 1e-12)
    r = torch.fft.ifft2(R).real                                            # [N, H, W]
    idx = r.reshape(r.shape[0], -1).argmax(dim=-1)
    d = torch.stack([(idx % W).float(), (idx // W).float()], dim=-1)       # [N, 2]
    ky = torch.fft.fftfreq(H, device=x.device)[:, None].expand(H, W)
    kx = torch.fft.fftfreq(W, device=x.device)[None, :].expand(H, W)
    k = torch.stack([kx, ky], dim=-1) * 2 * math.pi                        # [H, W, 2]
    for _ in range(newton_iters):
        th = (k[None] * d[:, None, None, :]).sum(-1)                       # [N, H, W]
        z = R * torch.polar(torch.ones_like(th), th)
        # f = Re sum z ; grad = Re sum i k z = -sum k Im z ; hess = -sum k k^T Re z
        g = -(k[None] * z.imag[..., None]).sum((1, 2))                      # [N, 2]
        Hs = -(k[None, ..., :, None] * k[None, ..., None, :] * z.real[..., None, None]).sum((1, 2))
        Hs = Hs - 1e-6 * torch.eye(2, device=x.device)
        step = torch.linalg.solve(Hs, g[..., None])[..., 0]
        d = d - step.clamp(-0.5, 0.5)
    dx = (d[:, 0] + W / 2) % W - W / 2
    dy = (d[:, 1] + H / 2) % H - H / 2
    return torch.stack([dx, dy], dim=-1).reshape(*lead, 2)


def _gauss5(sigma=1.0, device="cpu"):
    g = torch.exp(-(torch.arange(5, device=device).float() - 2) ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]                            # [1,1,5,5]


def lk_displacement(h, u, lam_rel=1e-2, sigma=1.0, eps=1e-3, max_disp=1.0, return_trace=False,
                    *, per_node_damping=False):
    """One-step Lucas-Kanade estimate of the extra per-step displacement of transported content.

    h: [B, V, C, h, w] transported maps on the observed window (the adaptive-node model uses its observation
    imprints); u: [B, C, h, w] the new observation in the same space. If the observation shows the content
    shifted by d relative to h, u(x) = h(x - d) ~ h(x) - grad h . d, so with e = h - u:
        d = (G + lam I)^-1 b,   G = sum_w sum_C grad h grad h^T,   b = sum_w sum_C grad h * e
    (central differences, 5x5 Gaussian window w). Damping lam = lam_rel * (tr G(x)/2 + mean_(nodes,pixels) tr G/2) + eps:
    the local part bounds the condition number on edges (aperture problem), the global part sends d -> 0 where
    there is no texture. d is clamped to +-max_disp px. Returns d [B, V, 2, h, w] (dx, dy).
    Differentiable (closed-form 2x2 inverse); always computed in float32 (autocast disabled: G in bf16 cancels).
    With per_node_damping=True, the reference trace averages pixels separately
    for each node, so a node's damping is independent of other lattice nodes.
    """
    with torch.autocast(device_type=h.device.type, enabled=False):
        d, tr2 = _lk_fp32(h.float(), u.float(), lam_rel, sigma, eps, max_disp, per_node_damping)
    return (d, tr2) if return_trace else d


def _lk_fp32(h, u, lam_rel, sigma, eps, max_disp, per_node_damping=False):
    B, V, C, H, W = h.shape
    hf = h
    e = hf - u[:, None]
    gx = 0.5 * (torch.roll(hf, -1, -1) - torch.roll(hf, 1, -1))
    gy = 0.5 * (torch.roll(hf, -1, -2) - torch.roll(hf, 1, -2))
    terms = torch.stack([(gx * gx).sum(2), (gx * gy).sum(2), (gy * gy).sum(2),
                         (gx * e).sum(2), (gy * e).sum(2)], 2)               # [B,V,5,H,W]
    k = _gauss5(sigma, h.device)
    t = torch.nn.functional.conv2d(torch.nn.functional.pad(terms.reshape(B * V * 5, 1, H, W), (2, 2, 2, 2),
                                                           mode="replicate"), k).reshape(B, V, 5, H, W)
    gxx, gxy, gyy, bx, by = t.unbind(2)
    tr2 = 0.5 * (gxx + gyy)
    dims = (2, 3) if per_node_damping else (1, 2, 3)
    lam = lam_rel * (tr2 + tr2.mean(dims, keepdim=True)) + eps              # [B,V,H,W]
    a, c = gxx + lam, gyy + lam
    det = (a * c - gxy * gxy).clamp_min(lam * lam)                            # PSD + lam I: det >= lam^2
    dx = (c * bx - gxy * by) / det
    dy = (a * by - gxy * bx) / det
    return torch.stack([dx, dy], 2).clamp(-max_disp, max_disp), tr2
