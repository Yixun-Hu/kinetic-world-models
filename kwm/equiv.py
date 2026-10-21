"""Action of a real Galilean boost on the kinetic memory, and the equivariance error.

Memory h [B, V, C, H, W] on a square velocity lattice nodes = Delta * Z^2 (|V| = n^2), ego coordinates.
A boost nu_hat in R^2 applied to the scene since frame 0 acts on the memory after t steps as
  * velocity: nu -> nu + nu_hat. k = round(nu_hat / Delta) permutes the nodes; a fixed lattice cannot represent
    the fractional part nu_hat - k Delta, which is dropped;
  * position: every population is displaced by nu_hat * t (spectral shift).
Populations pushed off the finite lattice are dropped.
For the adaptive-node model the boost moves every node velocity nu_i = lattice_i + rho_i by nu_hat
(boost_transform_nodes).
"""
import torch

from .transport import spectral_shift


def _lattice_index(nodes, spacing):
    """Integer lattice coordinates of the nodes and a lookup table coord -> node index."""
    ij = torch.round(nodes / spacing).long()                                # [V,2]
    if not torch.allclose(ij.float() * spacing, nodes, atol=1e-4):
        raise ValueError("boost_transform needs nodes on a square lattice")
    lo = ij.min(0).values
    n = int((ij.max(0).values - lo).max().item()) + 1
    table = torch.full((n, n), -1, dtype=torch.long, device=nodes.device)
    table[ij[:, 0] - lo[0], ij[:, 1] - lo[1]] = torch.arange(len(nodes), device=nodes.device)
    return ij, lo, n, table


def boost_transform(h, nodes, spacing, C, nu_hat, t):
    """Apply the boost nu_hat ([B,2]) to the feature channels of memory h ([B,V,>=C,H,W]) observed for t steps."""
    B, V, _, H, W = h.shape
    dev = h.device
    ij, lo, n, table = _lattice_index(nodes.to(dev).float(), spacing)
    k = torch.round(nu_hat / spacing)                                         # [B,2]
    feat = h[:, :, :C]
    tgt = (ij[None, :, :, None, None].float() + k[:, None, :, None, None]).expand(B, V, 2, H, W)
    rel = tgt.long() - lo[None, :, None, None]
    valid = ((rel >= 0) & (rel < n)).all(2)                                    # [B,V,H,W]
    idx = table[rel[:, :, 0].clamp(0, n - 1), rel[:, :, 1].clamp(0, n - 1)]   # [B,V,H,W]
    valid = valid & (idx >= 0)
    idx = torch.where(valid, idx, torch.zeros_like(idx))
    vm = valid[:, :, None].to(h.dtype)
    out = torch.zeros_like(feat).scatter_add_(1, idx[:, :, None].expand_as(feat), feat * vm)
    return spectral_shift(out, nu_hat * t)


def target_mask(nodes, spacing, nu_hat):
    """[B,V] bool: node j can receive a population under the boost, i.e. its source node j - round(nu_hat/Delta)
    exists on the lattice. Edge nodes without a source get content in the boosted run from velocities the
    lattice cannot hold (every node receives the lifted input), so they are excluded from the comparison."""
    ij, lo, n, table = _lattice_index(nodes.float(), spacing)
    k = torch.round(nu_hat / spacing).long().to(ij.device)                 # [B,2]
    src = ij[None] - k[:, None] - lo[None, None]                            # [B,V,2]
    return ((src >= 0) & (src < n)).all(-1)


def interior_mask(nodes, spacing):
    """[V] bool: nodes not on the lattice boundary (their content is not affected by what the lattice cannot hold)."""
    ij, lo, n, table = _lattice_index(nodes.float(), spacing)
    rel = ij - lo
    return ((rel > 0) & (rel < n - 1)).all(-1)


def equiv_error(h_transformed, h_boosted, C, mask=None):
    """Per-sequence normalised L2 error on the feature channels (optionally only nodes in mask [B,V]). -> [B]"""
    a, b = h_transformed[:, :, :C], h_boosted[:, :, :C]
    if mask is not None:
        m = mask[:, :, None, None, None].to(a.dtype)
        a, b = a * m, b * m
    a, b = a.flatten(1), b.flatten(1)
    return (a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-8)


def boost_transform_nodes(h, nodes, spacing, C, nu_hat, bound):
    """Comoving boost action on an adaptive-node memory: every node velocity nu_i = lattice_i + rho_i moves by u.
    Lattice part: channels permuted by k = round(u / spacing); rho += u - k*spacing; if |rho| exceeds the
    (overlapping-cell) bound the node carries to the neighbouring lattice node (rho -= +-spacing). W is unchanged.
    h [B,V,C+3+K,H,W] (features | rho, W | imprint). Node-level permutation; nodes pushed off the lattice are
    dropped. Returns (h', target_mask [B,V])."""
    B, V, _, H, W = h.shape
    E = 3
    ij, lo, n, table = _lattice_index(nodes.to(h.device).float(), spacing)
    rho = h[:, :, C:C + 2].float().mean((-2, -1))                                # [B,V,2]
    k = torch.round(nu_hat / spacing)                                             # [B,2]
    rho = rho + (nu_hat - k * spacing)[:, None]
    c = (rho > bound).float() - (rho < -bound).float()
    rho = rho - c * spacing
    tgt = ij[None].float() + k[:, None] + c                                       # [B,V,2]
    rel = tgt.long() - lo[None, None]
    valid = ((rel >= 0) & (rel < n)).all(-1)
    idx = table[rel[..., 0].clamp(0, n - 1), rel[..., 1].clamp(0, n - 1)]
    valid = valid & (idx >= 0)
    idx = torch.where(valid, idx, torch.zeros_like(idx))                          # [B,V]
    st = torch.cat([rho, h[:, :, C + 2:C + E].float().mean((-2, -1))], 2)       # node state [B,V,E]
    vm = valid.to(h.dtype)
    content = torch.cat([h[:, :, :C], h[:, :, C + E:]], 2) * vm[:, :, None, None, None]
    ci = idx[:, :, None, None, None].expand_as(content)
    out_c = torch.zeros_like(content).scatter_add_(1, ci, content)               # features + imprint: summed
    # node state at a collision (overlapping cells: a carried node can land where another node shifts to):
    # content-mass-weighted average, plain mean if no mass
    w = h[:, :, :C].float().norm(dim=2).sum((-2, -1)) * vm + 1e-9 * vm           # [B,V]
    num = torch.zeros_like(st).scatter_add_(1, idx[..., None].expand_as(st), st * w[..., None])
    den = torch.zeros_like(w).scatter_add_(1, idx, w)
    st_new = num / den.clamp_min(1e-12)[..., None]
    out = torch.cat([out_c[:, :, :C], st_new[..., None, None].expand(B, V, E, H, W).to(h.dtype), out_c[:, :, C:]], 2)
    tmask = torch.zeros(B, V, dtype=torch.bool, device=h.device)
    tmask.scatter_(1, idx, valid)
    return out, tmask
