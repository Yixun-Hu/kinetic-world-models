"""Frozen AdaptiveNodeCell and all repository helpers from main@302d8f8.

Self-contained bitwise reference: no imports from the live kwm tree.
"""
import torch
import torch.nn as nn

import math
import torch.nn.functional as F

WALL_NORMALS = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])


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


def _gauss5(sigma=1.0, device="cpu"):
    g = torch.exp(-(torch.arange(5, device=device).float() - 2) ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def lk_displacement(h, u, lam_rel=1e-2, sigma=1.0, eps=1e-3, max_disp=1.0, return_trace=False):
    """One-step Lucas-Kanade estimate of the extra per-step displacement of transported content.

    h: [B, V, C, h, w] transported maps on the observed window (the adaptive-node model uses its observation
    imprints); u: [B, C, h, w] the new observation in the same space. If the observation shows the content
    shifted by d relative to h, u(x) = h(x - d) ~ h(x) - grad h . d, so with e = h - u:
        d = (G + lam I)^-1 b,   G = sum_w sum_C grad h grad h^T,   b = sum_w sum_C grad h * e
    (central differences, 5x5 Gaussian window w). Damping lam = lam_rel * (tr G(x)/2 + mean_(nodes,pixels) tr G/2) + eps:
    the local part bounds the condition number on edges (aperture problem), the global part sends d -> 0 where
    there is no texture. d is clamped to +-max_disp px. Returns d [B, V, 2, h, w] (dx, dy).
    Differentiable (closed-form 2x2 inverse); always computed in float32 (autocast disabled: G in bf16 cancels).
    """
    with torch.autocast(device_type=h.device.type, enabled=False):
        d, tr2 = _lk_fp32(h.float(), u.float(), lam_rel, sigma, eps, max_disp)
    return (d, tr2) if return_trace else d


def _lk_fp32(h, u, lam_rel, sigma, eps, max_disp):
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
    lam = lam_rel * (tr2 + tr2.mean((1, 2, 3), keepdim=True)) + eps           # [B,V,H,W]
    a, c = gxx + lam, gyy + lam
    det = (a * c - gxy * gxy).clamp_min(lam * lam)                            # PSD + lam I: det >= lam^2
    dx = (c * bx - gxy * by) / det
    dy = (a * by - gxy * bx) / det
    return torch.stack([dx, dy], 2).clamp(-max_disp, max_disp), tr2


def _support(h, eps, eps_rel):
    """Mass support per velocity group: ||h(x,nu)||_C > max(eps, eps_rel * max_x ||h(.,nu)||). [B,V,H,W] float."""
    m = h.detach().float().norm(dim=2)                                       # [B,V,H,W]
    thr = torch.clamp(eps_rel * m.amax((-2, -1), keepdim=True), min=eps)
    return (m > thr).to(h.dtype)


def _dilate(c, s, K):
    """K x (3x3 max-pool, circular) restricted to the support s, within each velocity group. c,s [B,V,H,W]."""
    B, V, H, W = c.shape
    c = c.reshape(B * V, 1, H, W)
    s = s.reshape(B * V, 1, H, W)
    for _ in range(K):
        c = F.max_pool2d(F.pad(c, (1, 1, 1, 1), mode="circular"), 3, stride=1) * s
    return c.reshape(B, V, H, W)


def rigid_bounce_back(h, nodes, perm_x, perm_y, wall, K=28, eps=1e-3, eps_rel=0.05, return_moved=False):
    """Rigid-contact-propagation bounce-back.

    h [B,V,C,H,W]; nodes [V,2]; perm_x/perm_y [V]; wall [B,4,H,W] (channel convention above).
    Per axis (x walls, then y walls):
      1. contact seed  c0(x,nu) = sum_k w_k(x) * 1[nu heading into wall k] * supp(x,nu)
      2. propagate within the SAME velocity group: c <- maxpool3x3(c) * supp, K times
         (connected co-moving mass = one rigid body)
      3. move the flagged mass nu -> R nu:  h'(nu) = (1 - c(nu)) h(nu) + c(R nu) h(R nu)
    Mass conserving, commutes with mirrors, translation equivariant, identity without walls.
    supp = ||h||_C > max(eps, eps_rel * per-(b,nu) max) -- the relative threshold keeps the
    slowly spreading low-level activations of a learned memory from merging everything into one body.
    The flags are not differentiated (indicator + max-pool); gradients flow through h and w.
    nodes may also be per-sample [B,V,2] (adaptive nodes). return_moved: also return, per axis, the feature mass
    moved out of each node, [B,V] (used to reflect the adaptive node offsets along with the content).
    """
    normals = WALL_NORMALS.to(nodes.device, nodes.dtype)
    moved = []
    for ks, perm in (((0, 1), perm_x), ((2, 3), perm_y)):
        s = _support(h, eps, eps_rel)                                         # [B,V,H,W]
        c = 0
        for k in ks:
            g = (nodes.detach() @ normals[k] < -1e-6).to(h.dtype)
            g = g[None, :, None, None] if g.dim() == 1 else g[:, :, None, None]   # [1|B,V,1,1]
            c = c + wall[:, k][:, None] * g * s
        c = _dilate(c.clamp(0, 1), s, K)[:, :, None]                          # [B,V,1,H,W]
        if return_moved:
            moved.append((c[:, :, 0] * h.detach().float().norm(dim=2)).sum((-2, -1)))
        h = h - c * h + (c * h)[:, perm]
    return (h, moved) if return_moved else h


def _fixed_projection(C, K, seed=1234):
    """Fixed random 1x1 projection P [K, C]: orthonormal rows if K <= C, orthonormal columns otherwise."""
    g = torch.Generator().manual_seed(seed)
    if K <= C:
        q, _ = torch.linalg.qr(torch.randn(C, K, generator=g))
        return q.T.contiguous()
    q, _ = torch.linalg.qr(torch.randn(K, C, generator=g))
    return q.contiguous()


def int_roll_batch(h, shift):
    """h [B, ..., H, W]; shift [B, 2] integer-valued (dx, dy). Per-sample torch.roll
    exactly like FloWM's roll_batch_with_v (shifts=(dy, dx))."""
    s = shift.round().long().tolist()
    # unbind (not h[b]): the backward of B separate h[b] selects materialises B full-size
    # zero tensors (O(B^2) memory traffic); unbind's backward is a single stack.
    return torch.stack([torch.roll(hb, shifts=(s[b][1], s[b][0]), dims=(-2, -1))
                        for b, hb in enumerate(h.unbind(0))], 0)


class _Base(nn.Module):
    def __init__(self, in_ch, C, world, window):
        super().__init__()
        self.C, self.world, self.window = C, world, window
        self.conv_u = nn.Conv2d(in_ch, C, 3, padding=1, padding_mode="circular", bias=False)
        self.conv_h = nn.Conv2d(C, C, 3, padding=1, padding_mode="circular", bias=False)
        self.act = nn.ReLU()

    def lift(self, u_t, V):
        """Encode the observation and write it into the top-left window of a world map.
        u_t [B,1,win,win] -> [B,V,C,world,world] (expanded view over V)."""
        u = self.conv_u(u_t)                                                 # [B,C,win,win]
        B, C, h, w = u.shape
        full = u.new_zeros(B, C, self.world, self.world)
        full[:, :, :h, :w] = u
        return full[:, None].expand(-1, V, -1, -1, -1)

    def mix(self, hs):
        """Shared 3x3 conv over x for every velocity group: [B,V,C,H,W] -> same."""
        B, V, C, H, W = hs.shape
        return self.conv_h(hs.reshape(B * V, C, H, W)).reshape(B, V, C, H, W)


class AdaptiveNodeCell(_Base):
    """Per-sequence, per-node adaptive velocities nu_i(t) = lattice_i + rho_i(t), |rho| <= bound.

    State h [B, V, C + 3 + K, H, W]: features | rho (2 channels) and cumulative evidence W (1 channel), both
    spatially constant | observation imprint bank m (K channels: fixed random orthogonal projection P of the
    encoded observation, decaying by gamma per step).
    Step:
      1. spectral stream of everything by the per-sample node velocities nu_i(t)*dt - a (exact);
      2. Lucas-Kanade registration on the observed window between the transported imprint m and the new
         imprint P(U*f): per-pixel displacement d(x);
      3. object-level average delta_i = sum_x w d / sum_x w, with w = |m| * texture gate * |d| gate *
         coherence exp(-|m - P(U*f)|^2 / sigma^2) (only pixels where imprint and observation agree);
      4. evidence-weighted running mean: W += sum_x w, rho += clip(s * (sum_x w / W) * delta / dt, +-step_clip),
         with a single learned scalar s;
      5. features as in KineticCell; the imprint decays and is overwritten on the observed window.
    With update_nodes = False (prediction phase, --node_update_pred off) rho, W and the imprints are frozen.
    self.last_nodes keeps nu_i(t+1) for diagnostics.
    """

    extra = 3

    def __init__(self, in_ch=1, C=64, world=50, window=32, num_v=25, spacing=1.0, imprint_k=8, gamma=0.9,
                 lam_rel=1e-2, tr_rel=0.1, gate=0.6, bound=0.75, coh_rel=0.5, step_clip=0.3, s_init=1.0):
        super().__init__(in_ch, C, world, window)
        self.num_v, self.half, self.K_imp = num_v, spacing / 2, imprint_k
        self.gamma, self.lam_rel, self.tr_rel, self.gate = gamma, lam_rel, tr_rel, gate
        self.bound, self.coh_rel, self.step_clip = bound, coh_rel, step_clip
        self.n_s = nn.Parameter(torch.tensor(float(s_init)))
        self.register_buffer("P", _fixed_projection(C, imprint_k))               # fixed, not trained
        self.last_nodes = None
        self.update_nodes = True

    def imprint(self, u):
        return torch.einsum("kc,bchw->bkhw", self.P.to(u.dtype), u)

    def offsets(self, h):
        """rho [B,V,2] from the spatially constant state channels."""
        return h[:, :, self.C:self.C + 2].float().mean((-2, -1))

    def bounce(self, h, nodes, perm_x, perm_y, wall, K=28):
        """Rigid bounce-back on the per-sample nodes nu_i(t) (experimental). Content (features + imprint) of a node
        that hits a wall moves to the node whose lattice velocity is the reflected one; that node's offset becomes
        the reflected offset (rho_x -> -rho_x for x walls) and inherits the evidence W, when most of the source
        node's mass moved."""
        C, E = self.C, self.extra
        nu = nodes[None].float() + self.offsets(h)
        content = torch.cat([h[:, :, :C], h[:, :, C + E:]], 2)
        content, moved = rigid_bounce_back(content, nu, perm_x, perm_y, wall, K=K, return_moved=True)
        st = h[:, :, C:C + E].float().mean((-2, -1))                               # [B,V,E] node state
        tot = h[:, :, :C].detach().float().norm(dim=2).sum((-2, -1)) + 1e-6        # [B,V]
        for axis, perm in ((0, perm_x), (1, perm_y)):
            go = (moved[axis] > 0.5 * tot)                                          # [B,V]: node i bounced
            src = st.clone()
            src[:, :, axis] = -src[:, :, axis]                                      # reflected offset
            tgt = torch.zeros_like(go)
            tgt[:, perm] = go                                                       # targets perm(i) of bounced i
            new = st.clone()
            new[:, perm] = torch.where(go[..., None], src, st[:, perm])
            st = torch.where(tgt[..., None], new, st)
        B, V, _, H, W = h.shape
        fields = st[..., None, None].expand(B, V, E, H, W).to(h.dtype)
        return torch.cat([content[:, :, :C], fields, content[:, :, C:]], 2)

    def forward(self, u_t, h_prev, action, nodes, dt=1.0, collision=None):
        C, K, E = self.C, self.K_imp, self.extra
        if collision is not None:                       # wall bounce on the adaptive nodes
            h_prev = collision(h_prev)
        rho = self.offsets(h_prev)
        nu = nodes[None].float() + rho                                              # [B,V,2]
        hs = stream(h_prev, nu, action, dt)                                         # exact, per-sample nodes
        feat, m = hs[:, :, :C], hs[:, :, C + E:]
        win = u_t.shape[-1]
        u = self.conv_u(u_t)
        pu = self.imprint(u)
        mw = m[:, :, :, :win, :win]
        d, tr2 = lk_displacement(mw, pu, self.lam_rel, return_trace=True)          # [B,V,2,w,w], [B,V,w,w]
        with torch.no_grad():
            w = mw.float().norm(dim=2)                                              # imprint mass
            w = w * (tr2 > self.tr_rel * tr2.mean((1, 2, 3), keepdim=True)).float()
            w = w * (d.abs().amax(2) <= self.gate).float()
            sig = self.coh_rel * pu.float().norm(dim=1).mean((1, 2)) + 1e-6           # [B]
            mis = (mw.float() - pu.float()[:, None]).pow(2).sum(2)                   # [B,V,w,w]
            w = w * torch.exp(-mis / sig[:, None, None, None] ** 2)
        if not self.update_nodes:
            w = torch.zeros_like(w)       # no evidence from the model's own predictions: rho, W frozen
        wsum = w.sum((-2, -1))                                                      # [B,V] evidence this step
        delta = (w[:, :, None] * d).sum((-2, -1)) / (wsum[:, :, None] + 1e-6)     # [B,V,2]
        Wc = h_prev[:, :, C + 2:C + 3].float().mean((-2, -1))                      # [B,V,1] cumulative evidence
        W_new = Wc + wsum[..., None]
        gain = wsum[..., None] / W_new.clamp_min(1e-6)                              # running-mean gain
        step = (self.n_s * gain * delta / dt).clamp(-self.step_clip, self.step_clip)
        rho_new = (rho + step).clamp(-self.bound, self.bound)
        self.last_nodes = (nodes[None].float() + rho_new).detach()
        h_new = self.act(self.lift(u_t, self.num_v) + self.mix(feat))
        B, V, _, H, W = feat.shape
        fields = torch.cat([rho_new, W_new], 2)[..., None, None].expand(B, V, E, H, W)
        m_new = self.gamma * m.float()
        if self.update_nodes:            # imprints are written only from real observations when frozen
            m_new = torch.cat([torch.cat([pu.float()[:, None].expand(B, V, K, win, win), m_new[:, :, :, :win, win:]], 4),
                               m_new[:, :, :, win:]], 3)
        return torch.cat([h_new, fields.to(h_new.dtype), m_new.to(h_new.dtype)], 2)
