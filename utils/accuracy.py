"""
Metrics / decoders for cryo-EM particle picking.

- `dice_score` / `jaccard_score`: classic segmentation metrics (auxiliary mask head).
- `decode_heatmap`: CenterNet-style local-maximum extraction with score thresholding
  and a light peak NMS (radius in pixels).
- `keypoint_f1`: per-image keypoint P/R/F1 with per-GT-radius matching tolerance.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ────────────────────────────────────────────────────────────────
#  Segmentation metrics (legacy)
# ────────────────────────────────────────────────────────────────

def dice_score(target, pred, smooth=1e-4):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (2.0 * intersection + smooth) / (union + intersection + smooth)


def jaccard_score(target, pred, smooth=1e-4):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


# ────────────────────────────────────────────────────────────────
#  Heatmap decoding
# ────────────────────────────────────────────────────────────────

@torch.no_grad()
def decode_heatmap(
    prob: torch.Tensor,
    score_threshold: float = 0.3,
    nms_radius: int = 10,
    max_predictions: int = 2000,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Decode a batched heatmap (sigmoid-probs) into particle coordinates.

    A pixel is a peak iff it equals the max of a (2*nms_radius + 1) window
    centered on it AND exceeds ``score_threshold``. Output is per-image.

    Parameters
    ----------
    prob : (B, 1, H, W) float tensor in [0, 1]
    score_threshold : minimum score to keep
    nms_radius : radius (in pixels) of the local-maximum pooling window.
                 Typical: ``peak_nms_scale * particle_radius``.
    max_predictions : cap on the number of peaks kept per image (by score).

    Returns
    -------
    list of (coords_xy, scores) for each image in the batch, where
        coords_xy : (K, 2) float tensor, **(x, y)** in image coordinates
        scores    : (K,)  float tensor
    """
    assert prob.dim() == 4 and prob.size(1) == 1, f"expected (B,1,H,W), got {prob.shape}"
    kernel = max(int(nms_radius) * 2 + 1, 3)
    padding = kernel // 2
    pooled = F.max_pool2d(prob, kernel_size=kernel, stride=1, padding=padding)
    peaks = (prob == pooled) & (prob > score_threshold)

    B = prob.size(0)
    results: list[tuple[torch.Tensor, torch.Tensor]] = []
    for b in range(B):
        mask = peaks[b, 0]
        ys, xs = mask.nonzero(as_tuple=True)
        scores = prob[b, 0, ys, xs]
        if scores.numel() > max_predictions:
            top = torch.topk(scores, max_predictions)
            idx = top.indices
            xs, ys, scores = xs[idx], ys[idx], top.values
        coords = torch.stack([xs.float(), ys.float()], dim=-1)  # (K, 2)
        # sort by score desc for downstream matching
        order = torch.argsort(scores, descending=True)
        results.append((coords[order], scores[order]))
    return results


# ────────────────────────────────────────────────────────────────
#  Keypoint F1
# ────────────────────────────────────────────────────────────────

def _match_greedy(
    pred_xy: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_xy: torch.Tensor,
    gt_radii: torch.Tensor,
    match_radius_scale: float,
) -> tuple[int, int, int]:
    """Greedy score-descending nearest-neighbor matching.

    A predicted point is a TP iff some unclaimed gt point is within
    ``match_radius_scale * gt_radius`` pixels and it is the closest such gt.

    Returns (tp, fp, fn).
    """
    n_pred = pred_xy.size(0)
    n_gt = gt_xy.size(0)
    if n_gt == 0:
        return 0, n_pred, 0
    if n_pred == 0:
        return 0, 0, n_gt

    # assume already sorted by score desc
    claimed = torch.zeros(n_gt, dtype=torch.bool)
    max_d = match_radius_scale * gt_radii  # (n_gt,)

    tp = 0
    for i in range(n_pred):
        d = torch.linalg.norm(gt_xy - pred_xy[i].unsqueeze(0), dim=1)  # (n_gt,)
        d = d.clone()
        d[claimed] = float("inf")
        # valid matches: distance <= max_d
        valid = d <= max_d
        if valid.any():
            best = torch.argmin(d)
            if valid[best]:
                claimed[best] = True
                tp += 1
    fp = n_pred - tp
    fn = n_gt - int(claimed.sum().item())
    return tp, fp, fn


def _match_hungarian(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    gt_radii: torch.Tensor,
    match_radius_scale: float,
) -> tuple[int, int, int]:
    """Optimal bipartite matching using the Hungarian algorithm."""
    n_pred = pred_xy.size(0)
    n_gt = gt_xy.size(0)
    if n_gt == 0:
        return 0, n_pred, 0
    if n_pred == 0:
        return 0, 0, n_gt

    BIG = 1e6
    d = torch.cdist(pred_xy, gt_xy).cpu().numpy()            # (P, G)
    max_d = (match_radius_scale * gt_radii).cpu().numpy()    # (G,)
    cost = d.copy()
    cost[d > max_d[None, :]] = BIG

    row_ind, col_ind = linear_sum_assignment(cost)
    tp = 0
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < BIG:
            tp += 1
    fp = n_pred - tp
    fn = n_gt - tp
    return tp, fp, fn


def keypoint_f1(
    pred_results: list[tuple[torch.Tensor, torch.Tensor]],
    gt_coords_list: list[torch.Tensor],
    gt_radii_list: list[torch.Tensor],
    match_radius_scale: float = 0.5,
    use_hungarian: bool = False,
) -> dict:
    """Aggregate keypoint Precision/Recall/F1 over a batch of images.

    pred_results : list of (coords_xy (K,2), scores (K,)), one per image
    gt_coords_list, gt_radii_list : per-image GT points at the same resolution
    match_radius_scale : k in `match threshold = k * gt_radius`
    use_hungarian : if True and scipy is available, use Hungarian matching;
                    otherwise greedy-by-score.
    """
    tot_tp = tot_fp = tot_fn = 0
    per_image: list[dict] = []
    for (pred_xy, pred_scores), gt_xy, gt_radii in zip(
        pred_results, gt_coords_list, gt_radii_list
    ):
        if use_hungarian and _HAS_SCIPY:
            tp, fp, fn = _match_hungarian(pred_xy, gt_xy, gt_radii, match_radius_scale)
        else:
            tp, fp, fn = _match_greedy(
                pred_xy, pred_scores, gt_xy, gt_radii, match_radius_scale
            )
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_image.append({"tp": tp, "fp": fp, "fn": fn, "p": p, "r": r, "f1": f})

    P = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    R = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    F = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return {
        "precision": P,
        "recall": R,
        "f1": F,
        "tp": tot_tp,
        "fp": tot_fp,
        "fn": tot_fn,
        "per_image": per_image,
    }


def image_mean_radius(gt_radii: torch.Tensor, fallback: float = 19.0) -> float:
    """Return a scalar radius estimate for an image: mean of gt_radii or a fallback."""
    if gt_radii.numel() == 0:
        return float(fallback)
    return float(gt_radii.mean().item())
