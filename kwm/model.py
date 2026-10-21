"""KineticWM: models flowm | spectral | adaptive | oracle.

  flowm     faithful FloWM replica: integer velocity lattice {-2..2}^2, torch.roll streaming, max readout.
  spectral  streaming + ego motion is one Fourier phase shift by nu_i * dt - a, with real values.
  adaptive  spectral transport with adaptive velocity nodes nu_i(t) = lattice_i + rho_i(t), estimated online per
            sequence (cells.AdaptiveNodeCell).
  oracle    diagnostic: spectral transport with the nodes set to each sequence's true digit velocities + (0,0),
            passed as forward(..., nodes=oracle_nodes_from(vel0)).
  --collision bounce (spectral or adaptive, experimental): rigid bounce-back on reflecting walls.

forward(obs [B,T_in,1,win,win], actions_in [B,T_in,2], actions_out [B,T_pred,2],
        target [B,T_pred,1,win,win] | None, tf_ratio, dt, wall0 [B,4,W,W] | None)
  -> preds [B,T_pred,1,win,win]   (optionally aux dict)

Protocol (identical to FloWM's FlowEquivariantRNN.forward):
  encode: for t < T_in: h = cell(obs[t], h, a_in[t])
  predict: for t < T_pred: frame = target[t] w.p. tf_ratio (training) else prev
           h = cell(frame, h, a_out[t]); out = decode(h[..., :win, :win]); prev = out
  (dec_input='zero' feeds zeros instead of prev: pure free-running memory.)

Memory h [B,V,C(+extra),W,W] in the ego frame, window = top-left corner.
Readout: 'max'     -> max over V (FloWM)
         'moments' -> [mass (C), mean velocity (2), velocity spread (1), max (C)] -> 1x1 conv to C, where
                      w_i(x) = softmax_i(beta * e_i / max_j e_j), e_i = ||h_i(x)||_C
                      mass = sum w h, mean = sum w nu, spread = sum w |nu - mean|^2
                      readout_rel: nu is taken relative to the observer, nu - a_t/dt (Galilean invariant)
Decoder (FloWM, decoder_conv_layers=1): conv3x3 C->C, ReLU, conv3x3 C->1, circular, no bias.
"""
import math
import random

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

from .cells import FloWMCell, KineticCell, AdaptiveNodeCell
from .collision import (lattice_nodes, reflection_perms, bounce_back, rigid_bounce_back,
                        known_wall_field, WallEncoder, VMixing, MLPCollision)
from .transport import spectral_shift

MODELS = ("flowm", "spectral", "adaptive", "oracle")
COLLISIONS = ("none", "bounce", "vmix", "mlp")


def make_wall0(cam0, world, band=3):
    """Known wall field in the ego frame at t=0: world field shifted by -cam0. cam0 [B,2] -> [B,4,W,W]."""
    f = known_wall_field(world, band, cam0.device)[None].expand(cam0.shape[0], -1, -1, -1)
    return spectral_shift(f, -cam0).clamp(0, 1)


class KineticWM(nn.Module):
    def __init__(self, kind="adaptive", C=32, world=50, window=32, in_ch=1,
                 node_spacing=1.0, vmax=2.0, readout=None,
                 collision="none", wall_mode="known", bounce_mode="directional", wall_band=3,
                 bounce="rigid", rigid_K=28, rigid_eps=1e-3, rigid_eps_rel=0.05,
                 dec_input="flowm", beta_init=10.0, grad_ckpt=False,
                 n_digits=5, imprint_k=8, imprint_gamma=0.9,
                 node_update_pred=False, readout_rel=True):
        super().__init__()
        if kind not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")
        if collision not in COLLISIONS:
            raise ValueError(f"collision must be one of {COLLISIONS}")
        self.kind, self.C, self.world, self.window = kind, C, world, window
        self.dec_input, self.grad_ckpt = dec_input, grad_ckpt
        self.node_update_pred = node_update_pred   # adaptive: keep updating nodes / imprints on the model's own predictions
        self.readout_rel = readout_rel             # moment readout uses velocities relative to the observer
        self._rel = None
        self.oracle = kind == "oracle"            # nodes = per-sequence true digit velocities + (0,0)
        self._nodes_ovr = None
        self.state_extra = 0                       # adaptive: rho (2) + W (1) + imprint (K) state channels
        self.collision_kind = collision
        self.wall_mode, self.bounce_mode, self.wall_band = wall_mode, bounce_mode, wall_band
        self.bounce, self.rigid_K, self.rigid_eps, self.rigid_eps_rel = bounce, rigid_K, rigid_eps, rigid_eps_rel
        if bounce not in ("rigid", "pointwise"):
            raise ValueError(bounce)

        if kind == "flowm":
            self.cell = FloWMCell(in_ch, C, world, window, v_range=2)
            readout = "max"
            if collision != "none":
                raise ValueError("flowm is the faithful FloWM replica: no collision")
        else:
            if self.oracle:
                if collision != "none":
                    raise ValueError("the oracle diagnostic has no collision operator")
                V = n_digits + 1
            else:
                nodes = lattice_nodes(node_spacing, vmax)
                self.register_buffer("fixed_nodes", nodes)
                px, py = reflection_perms(nodes)
                self.register_buffer("perm_x", px)
                self.register_buffer("perm_y", py)
                V = nodes.shape[0]
            if kind == "adaptive":
                if collision not in ("none", "bounce"):
                    raise ValueError("adaptive supports --collision none or bounce")
                self.cell = AdaptiveNodeCell(in_ch, C, world, window, num_v=V, spacing=node_spacing,
                                             imprint_k=imprint_k, gamma=imprint_gamma)
                self.state_extra = self.cell.extra + self.cell.K_imp
            else:
                self.cell = KineticCell(in_ch, C, world, window, num_v=V)
        self.num_v = self.cell.num_v
        self._node_traj = None
        self.readout = readout or "moments"

        # collision modules
        self.coll = None
        self.wall_enc = None
        if collision == "vmix":
            self.coll = VMixing(self.num_v)
        elif collision == "mlp":
            self.coll = MLPCollision(self.num_v, C)
        elif collision == "bounce":
            if wall_mode == "learned":
                self.wall_enc = WallEncoder()
            elif wall_mode != "known":
                raise ValueError(wall_mode)

        if self.readout == "moments":
            self.log_beta = nn.Parameter(torch.tensor(math.log(beta_init)))
            self.proj = nn.Conv2d(2 * C + 3, C, 1, bias=False)
        dec = [nn.Conv2d(C, C, 3, padding=1, padding_mode="circular", bias=False), nn.ReLU(),
               nn.Conv2d(C, 1, 3, padding=1, padding_mode="circular", bias=False)]
        self.decoder_conv = nn.Sequential(*dec)

    # ------------------------------------------------------------------
    def nodes(self):
        if self._nodes_ovr is not None:
            return self._nodes_ovr                                            # [B,V,2] oracle nodes
        if self.kind == "flowm":
            return self.cell.nodes
        return self.fixed_nodes

    def perms(self):
        return self.perm_x, self.perm_y

    def velocity_weights(self, hw, beta=None):
        """hw [B,V,C,h,w] -> softmax weights over V [B,V,h,w]."""
        e = hw.norm(dim=2)
        e = e / (e.amax(1, keepdim=True) + 1e-6)
        if beta is None:
            beta = self.log_beta.exp() if hasattr(self, "log_beta") else 10.0
        return torch.softmax(beta * e, dim=1)

    def node_velocity(self, h):
        """Per-pixel velocity of every node on the window: [B,V,2,win,win]."""
        B, V = h.shape[:2]
        nodes = self.nodes()
        nv = (nodes if nodes.dim() == 3 else nodes[None].expand(B, -1, -1))[..., None, None]   # [B,V,2,1,1]
        if self.kind == "adaptive":                                            # per-sample adaptive nodes
            nv = (nodes[None].float() + self.cell.offsets(h))[..., None, None]
        return nv.expand(-1, -1, -1, self.window, self.window)

    def decode(self, h, aux=None):
        hw = h[:, :, :self.C, :self.window, :self.window]                    # [B,V,C,win,win]
        mx = hw.amax(1)
        if self.readout == "max":
            z = mx
        else:
            vel = self.node_velocity(h)                                       # [B,V,2,win,win]
            w = self.velocity_weights(hw)                                     # [B,V,win,win]
            mass = (w[:, :, None] * hw).sum(1)
            if self._rel is not None:                                         # Galilean invariant: v - a/dt
                vel = vel - self._rel[:, None, :, None, None].to(vel.dtype)
            mean = (w[:, :, None] * vel).sum(1)                               # [B,2,win,win]
            spread = (w * ((vel - mean[:, None]) ** 2).sum(2)).sum(1, keepdim=True)
            z = self.proj(torch.cat([mass, mean.to(mass.dtype), spread.to(mass.dtype), mx], 1))
        if aux is not None:
            with torch.no_grad():
                vel = self.node_velocity(h).float()
                w = self.velocity_weights(hw).float()
                aux["mean_velocity"].append((w[:, :, None] * vel).sum(1))
                am = hw.float().norm(dim=2).argmax(1)                         # [B,win,win]
                aux["argmax_velocity"].append(vel.gather(1, am[:, None, None].expand(-1, 1, 2, -1, -1))[:, 0])
        return self.decoder_conv(z)

    # ------------------------------------------------------------------
    def _step(self, frame, h, a, dt, wall):
        """One recurrence step. wall (ego frame of h) or None. Returns (h_new, wall_new)."""
        if self.kind == "flowm":
            return self.cell(frame, h, a, dt), wall
        nodes = self.nodes()
        coll = None
        if self.collision_kind == "bounce" and self.kind == "adaptive":
            px, py = self.perms()
            coll = lambda hh: self.cell.bounce(hh, nodes, px, py, wall, self.rigid_K)
        elif self.collision_kind == "bounce":
            px, py = self.perms()
            if self.bounce == "rigid":
                coll = lambda hh: rigid_bounce_back(hh, nodes, px, py, wall, self.rigid_K,
                                                    self.rigid_eps, self.rigid_eps_rel)
            else:
                coll = lambda hh: bounce_back(hh, nodes, px, py, wall, self.bounce_mode)
        elif self.coll is not None:
            coll = self.coll
        h_new = self.cell(frame, h, a, nodes, dt, coll)
        if self._node_traj is not None and self.kind == "adaptive":
            self._node_traj.append(self.cell.last_nodes)
        if wall is not None:
            wall = spectral_shift(wall, -a)                                   # static (nu=0) transport
            if self.wall_enc is not None:
                ww = self.wall_enc(frame)                                     # overwrite the observed window
                wall = torch.cat([torch.cat([ww, wall[:, :, :self.window, self.window:]], 3),
                                  wall[:, :, self.window:]], 2)
            else:
                wall = wall.clamp(0, 1)
        return h_new, wall

    def step(self, frame, h, a, dt, wall):
        if self.grad_ckpt and self.training and torch.is_grad_enabled():
            if wall is None:
                return ckpt.checkpoint(lambda f, hh, aa: self._step(f, hh, aa, dt, None),
                                       frame, h, a, use_reentrant=False)
            return ckpt.checkpoint(lambda f, hh, aa, ww: self._step(f, hh, aa, dt, ww),
                                   frame, h, a, wall, use_reentrant=False)
        return self._step(frame, h, a, dt, wall)

    def forward(self, obs, actions_in, actions_out, target=None, tf_ratio=0.0, dt=1.0,
                wall0=None, return_aux=False, nodes=None, return_states=None, record_nodes=False):
        """nodes: [B,V,2] per-sequence nodes, required iff kind == 'oracle'.
        return_states: iterable of observation-step indices t; the full memory after processing frame t
        is returned in aux["states"][t] (equivariance tests). Implies return_aux.
        record_nodes (adaptive): aux["node_traj"] [B,T,V,2] = nu_i(t+1) after every step."""
        if self.oracle and nodes is None:
            raise ValueError("the oracle model needs nodes=[B,V,2] (see oracle_nodes_from)")
        self._nodes_ovr = nodes if self.oracle else None
        self._node_traj = [] if (record_nodes and self.kind == "adaptive") else None
        try:
            out = self._forward(obs, actions_in, actions_out, target, tf_ratio, dt, wall0, return_aux or record_nodes,
                                set(return_states or ()))
            if self._node_traj is not None:
                out[1]["node_traj"] = torch.stack(self._node_traj, 1)               # [B,T,V,2] nu_i(t+1)
            return out
        finally:
            self._nodes_ovr = None
            self._node_traj = None
            self._rel = None
            if hasattr(self.cell, "update_nodes"):
                self.cell.update_nodes = True

    def _forward(self, obs, actions_in, actions_out, target, tf_ratio, dt, wall0, return_aux, keep=()):
        B, T_in = obs.shape[:2]
        T_pred = actions_out.shape[1]
        dev = obs.device
        h = obs.new_zeros(B, self.num_v, self.C + self.state_extra, self.world, self.world)
        wall = None
        if self.collision_kind == "bounce":
            if self.wall_mode == "known":
                if wall0 is None:
                    raise ValueError("known-wall bounce-back needs wall0 (see make_wall0)")
                wall = wall0
            else:
                wall = obs.new_zeros(B, 4, self.world, self.world)
        states = {}
        for t in range(T_in):
            h, wall = self.step(obs[:, t], h, actions_in[:, t], dt, wall)
            if t in keep:
                states[t] = h.detach().float().clone()
        prev = obs[:, -1]
        if self.kind == "adaptive" and not self.node_update_pred:
            self.cell.update_nodes = False
        return_aux = return_aux or bool(keep)
        aux = {"mean_velocity": [], "argmax_velocity": []} if return_aux else None
        outs = []
        for t in range(T_pred):
            if self.training and target is not None and random.random() < tf_ratio:
                frame = target[:, t]
            elif self.dec_input == "zero":
                frame = torch.zeros_like(prev)
            else:
                frame = prev
            h, wall = self.step(frame, h, actions_out[:, t], dt, wall)
            self._rel = actions_out[:, t] / dt if self.readout_rel else None
            out = self.decode(h, aux)
            outs.append(out)
            prev = out
        preds = torch.stack(outs, 1)
        if return_aux:
            out = {k: torch.stack(v, 1) for k, v in aux.items()}
            if keep:
                out["states"] = states
            return preds, out
        return preds


def oracle_nodes_from(vel0):
    """Per-sequence oracle nodes: (0,0) + the true digit velocities. vel0 [B,nd,2] -> [B,nd+1,2]."""
    return torch.cat([vel0.new_zeros(vel0.shape[0], 1, 2), vel0], 1)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build_model(args):
    """args: namespace / dict with the train.py CLI fields (a checkpoint's saved args)."""
    g = (lambda k, d=None: args.get(k, d)) if isinstance(args, dict) else (lambda k, d=None: getattr(args, k, d))
    return KineticWM(kind=g("model"), C=g("C", 32), world=g("world", 50), window=g("window", 32),
                     node_spacing=g("node_spacing", 1.0), readout=g("readout", None),
                     collision=g("collision", "none"),
                     wall_mode=g("wall_mode", "known"), bounce_mode=g("bounce_mode", "directional"),
                     wall_band=g("wall_band", 3), dec_input=g("dec_input", "flowm"),
                     bounce=g("bounce", "rigid"), rigid_K=g("rigid_K", 28), rigid_eps=g("rigid_eps", 1e-3),
                     rigid_eps_rel=g("rigid_eps_rel", 0.05),
                     imprint_k=g("imprint_k", 8), imprint_gamma=g("imprint_gamma", 0.9),
                     node_update_pred=(g("node_update_pred", "off") == "on"),
                     readout_rel=(g("readout_rel", "on") == "on"),
                     grad_ckpt=g("grad_ckpt", False))
