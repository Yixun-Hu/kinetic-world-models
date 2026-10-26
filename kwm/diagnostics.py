"""Read-only memory diagnostics; truth never enters the prediction dynamics."""
import torch

from .lock_nodes import features


def digit_masks(cent, half, cam, world, win):
    """Periodic digit bounding boxes clipped to the observed window."""
    j = torch.arange(win, device=cent.device).float()
    wx = cam[:, None, 0, None] + j
    wy = cam[:, None, 1, None] + j
    dx = torch.remainder(wx - cent[..., 0, None] + world / 2, world) - world / 2
    dy = torch.remainder(wy - cent[..., 1, None] + world / 2, world) - world / 2
    mx = dx.abs() <= half[..., 0, None] + 0.5
    my = dy.abs() <= half[..., 1, None] + 0.5
    return my[..., :, None] & mx[..., None, :]


@torch.no_grad()
def node_concentration(feat, nodes, velocity, masks, visible):
    """[B,digit] nearest-node squared feature energy / all-node energy in its bbox.

    Invisible/empty boxes are NaN (excluded from summaries); visible boxes with
    no memory energy score zero. Exact nearest-velocity ties use the first node.
    """
    if nodes.ndim == 2:
        nodes = nodes[None].expand(feat.shape[0], -1, -1)
    closest = (nodes[:, None] - velocity[:, :, None]).square().sum(-1).argmin(-1)
    energy = feat.float().square().sum(2)
    local = torch.einsum("bvhw,bdhw->bdv", energy, masks.float())
    selected = local.gather(-1, closest[..., None]).squeeze(-1)
    fraction = (selected / local.sum(-1).clamp_min(1e-30)).clamp(0, 1)
    return fraction.masked_fill(~(visible & masks.any((-2, -1))), float("nan"))


def concentration_at(model, h, truth, t):
    masks = digit_masks(truth["cent"][:, t], truth["half"], truth["cam"][:, t],
                        model.world, model.window)
    feat = features(h)[:, :, :model.C, :model.window, :model.window]
    return node_concentration(feat, model.node_velocity(h)[..., 0, 0],
                              truth["vel"][:, t], masks, truth["visible"][:, t])


@torch.no_grad()
def effective_node_count(feat):
    """[B] mean over pixels of exp(entropy(node squared-energy distribution)).

    Compute entropy per pixel, before spatial averaging; zero-energy pixels
    have zero effective nodes. Feature channels only, without metadata.
    """
    energy = feat.float().square().sum(2)
    total = energy.sum(1, keepdim=True)
    share = energy / total.clamp_min(1e-30)
    count = (-(share * share.clamp_min(1e-30).log()).sum(1)).exp()
    return count.masked_fill(total[:, 0] == 0, 0).mean((-2, -1))


def effective_count_at(model, h):
    return effective_node_count(features(h)[:, :, :model.C, :model.window, :model.window])


def effective_count_summary(values):
    """values [B,1+T_pred], ordinary observation state followed by predictions."""
    out = {str(t): float(values[:, t].mean()) for t in (0, 1, 10, 20) if t < values.shape[1]}
    out["mean_prediction"] = float(values[:, 1:].mean()) if values.shape[1] > 1 else None
    return out


def prediction_truth(cache, sl, T_obs, T_pred, device):
    """Compact cache labels for aux only, indexed from the first prediction."""
    truth = {k: cache[k][sl, T_obs:T_obs + T_pred].to(device)
             for k in ("cent", "cam", "vel", "visible")}
    truth["half"] = cache["half"][sl].to(device)
    return truth


def concentration_summary(values, horizons=(1, 10, 20)):
    """Pool visible digit instances at each requested predicted frame."""
    out = {}
    for t in horizons:
        if t <= values.shape[1]:
            v = values[:, t - 1]
            v = v[torch.isfinite(v)]
            out[str(t)] = float(v.mean()) if v.numel() else None
    return out
