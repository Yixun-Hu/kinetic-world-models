"""Velocity lattices and collision operators (walls; experimental).

Memory layout everywhere: h [B, V, C, H, W]; V = velocity nodes, C = features,
H = rows (y), W = cols (x). nodes [V, 2] = (vx, vy) world-frame velocities.
Wall fields: wall [B, 4, H, W] in [0,1], channels
  0 'left'   wall, inward normal +x  (reflects populations with vx < 0)
  1 'right'  wall, inward normal -x  (reflects vx > 0)
  2 'top'    wall, inward normal +y  (row 0 side; reflects vy < 0)
  3 'bottom' wall, inward normal -y  (reflects vy > 0)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

WALL_NORMALS = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])


# ---------------------------------------------------------------- nodes ----
def lattice_nodes(spacing=1.0, vmax=2.0):
    """Square lattice in [-vmax, vmax]^2, ordered vx-major like FloWM's v_list."""
    n = int(round(vmax / spacing))
    g = torch.arange(-n, n + 1).float() * spacing
    return torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)  # [(2n+1)^2, 2]


def reflection_perms(nodes, tol=1e-4):
    """perm_x[i] = index of R_x nodes[i] = (-vx, vy); perm_y likewise. Asserts closure."""
    out = []
    for flip in (torch.tensor([-1.0, 1.0]), torch.tensor([1.0, -1.0])):
        r = nodes * flip
        d = torch.cdist(r, nodes)
        j = d.argmin(1)
        assert (d.min(1).values < tol).all(), "node set not closed under reflection"
        out.append(j)
    return out[0], out[1]


# ----------------------------------------------------------- bounce-back ----
def known_wall_field(world, band=3, device="cpu"):
    """World-frame directional wall field [4, W, W] for the env's border walls.
    Band = wall pixel + (band-1) interior pixels on the inner side of each wall."""
    w = torch.zeros(4, world, world, device=device)
    w[0, :, :band] = 1
    w[1, :, world - band:] = 1
    w[2, :band, :] = 1
    w[3, world - band:, :] = 1
    return w


def bounce_back(h, nodes, perm_x, perm_y, wall, mode="directional"):
    """Structured bounce-back collision (pointwise in x, a permutation in v).

    h [B,V,C,H,W]; nodes [V,2]; perm_x/perm_y [V] long; wall [B,4,H,W].
    mode 'symmetric':  h' = (1-w_n) h(nu) + w_n h(R_n nu), w_x = max(left,right).
    mode 'directional' (LBM half-way style, mass conserving, no trapping):
        for each wall k with inward normal n_k and gate g_k(nu) = 1[nu . n_k < 0]
        (population heading into the wall):
        h'(nu) = (1 - w_k g_k(nu)) h(nu) + w_k g_k(R nu) h(R nu)
    Both commute with the corresponding spatial+velocity reflection and are the
    identity when wall == 0.
    """
    if mode == "symmetric":
        wx = torch.maximum(wall[:, 0], wall[:, 1])[:, None, None]          # [B,1,1,H,W]
        wy = torch.maximum(wall[:, 2], wall[:, 3])[:, None, None]
        h = (1 - wx) * h + wx * h[:, perm_x]
        h = (1 - wy) * h + wy * h[:, perm_y]
        return h
    assert mode == "directional"
    normals = WALL_NORMALS.to(nodes.device, nodes.dtype)
    # the two walls of one axis act simultaneously (order-free, so the operator
    # commutes with the mirror that swaps them); x-axis pair first, then y-axis pair
    for ks, perm in (((0, 1), perm_x), ((2, 3), perm_y)):
        delta = 0
        for k in ks:
            g = (nodes.detach() @ normals[k] < -1e-6).to(h.dtype)[None, :, None, None, None]   # [1,V,1,1,1]
            w = wall[:, k][:, None, None]                                                    # [B,1,1,H,W]
            delta = delta - w * g * h + w * g[:, perm] * h[:, perm]
        h = h + delta
    return h


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


class WallEncoder(nn.Module):
    """Learned-wall mode: observation [B,1,win,win] -> wall field [B,4,win,win] in [0,1]."""

    def __init__(self, hidden=8):
        super().__init__()
        self.c1 = nn.Conv2d(1, hidden, 5, padding=2, padding_mode="circular")
        self.c2 = nn.Conv2d(hidden, 4, 5, padding=2, padding_mode="circular")
        nn.init.constant_(self.c2.bias, -3.0)

    def forward(self, x):
        return torch.sigmoid(self.c2(F.relu(self.c1(x))))


# ------------------------------------------------------ control collisions ----
class VMixing(nn.Module):
    """FERNN-style full-parameter V-mixing: h'(nu) = h(nu) + sum_nu' M[nu,nu'] h(nu').
    Content independent, shared over C and x; zero-init (starts as identity)."""

    def __init__(self, V):
        super().__init__()
        self.M = nn.Parameter(torch.zeros(V, V))

    def forward(self, h, **_):
        return h + torch.einsum("ij,bjchw->bichw", self.M, h)


class MLPCollision(nn.Module):
    """Learned collision: per-pixel, per-feature MLP across V, gated by local features.
    h'(x) = h(x) + gate(x) * MLP_V(h(x, :)), gate = sigmoid(1x1 conv of max_V h)."""

    def __init__(self, V, C, hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(V, hidden)
        self.fc2 = nn.Linear(hidden, V)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.gate = nn.Conv2d(C, 1, 1)

    def forward(self, h, **_):
        B, V, C, H, W = h.shape
        g = torch.sigmoid(self.gate(h.amax(1)))[:, None]                    # [B,1,1,H,W]
        z = h.permute(0, 2, 3, 4, 1)                                         # [B,C,H,W,V]
        z = self.fc2(F.relu(self.fc1(z))).permute(0, 4, 1, 2, 3)             # [B,V,C,H,W]
        return h + g * z
