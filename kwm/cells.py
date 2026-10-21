"""Recurrent cells. Memory h: [B, V, C, H, W] in ego coordinates (the camera
window is always the top-left [:win, :win] corner, as in FloWM).

FloWMCell        faithful re-implementation of FloWM's FERNN_Cell_w_VChannels
                 (integer velocity lattice, torch.roll streaming, ego roll by -action,
                 3x3 circular conv without bias, ReLU).
KineticCell      same recurrence, but streaming + ego motion is one spectral phase
                 shift by (nu_i * dt - a) for real node velocities, with an optional
                 collision operator applied before streaming (collide, then stream).
AdaptiveNodeCell KineticCell with per-sequence node velocities nu_i(t) = lattice_i + rho_i(t),
                 estimated online from the observations (see the class docstring).
"""
import torch
import torch.nn as nn

from .transport import stream, lk_displacement


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


class FloWMCell(_Base):
    def __init__(self, in_ch=1, C=64, world=50, window=32, v_range=2):
        super().__init__(in_ch, C, world, window)
        # identical ordering to FloWM: vx outer, vy inner
        self.v_list = [(vx, vy) for vx in range(-v_range, v_range + 1) for vy in range(-v_range, v_range + 1)]
        self.num_v = len(self.v_list)
        self.register_buffer("nodes", torch.tensor(self.v_list).float(), persistent=False)

    def forward(self, u_t, h_prev, action, dt=1.0):
        if dt != 1.0:
            raise ValueError("FloWM (integer roll) cannot stream a fractional time step")
        hs = torch.stack([torch.roll(hv, shifts=(vy, vx), dims=(2, 3))
                          for hv, (vx, vy) in zip(h_prev.unbind(1), self.v_list)], 1)   # intrinsic flow
        hs = int_roll_batch(hs, -action)                                    # ego motion (integerised)
        return self.act(self.lift(u_t, self.num_v) + self.mix(hs))


class KineticCell(_Base):
    """forward(u_t, h_prev, action, nodes [V,2] or [B,V,2], dt, collision(h) -> h or None)."""

    def __init__(self, in_ch=1, C=64, world=50, window=32, num_v=25):
        super().__init__(in_ch, C, world, window)
        self.num_v = num_v

    def forward(self, u_t, h_prev, action, nodes, dt=1.0, collision=None):
        if collision is not None:
            h_prev = collision(h_prev)
        hs = stream(h_prev, nodes, action, dt)                               # spectral, exact
        return self.act(self.lift(u_t, self.num_v) + self.mix(hs))


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
        from .collision import rigid_bounce_back
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
