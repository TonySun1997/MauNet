"""
Dataset for supervised cryo-EM particle picking (heatmap-based keypoint + aux mask).

Expected raw layout (per EMPIAR):
    <root>/<EMPIAR_ID>/train/images/<base>.png
    <root>/<EMPIAR_ID>/train/masks/<base>_mask.png
    <root>/<EMPIAR_ID>/train/particle_coordinates/<base>.csv   (columns: x,y,radius)
    <root>/<EMPIAR_ID>/train/false_positives/<base>.csv        (optional, same schema)

A single __getitem__ returns a dict:
    {
        "image":     (1, H, W) float32, z-scored or /255
        "mask":      (1, H, W) float32 in [0,1], auxiliary segmentation GT
        "heatmap":   (1, H, W) float32 in [0,1], Gaussian-rendered particle centers
        "fp_weight": (1, H, W) float32, per-pixel negative-loss weight (1 everywhere,
                     fp_neg_weight inside FP disks)
        "gt_coords": (N, 2) float32 (x, y) at training resolution, original (un-augmented) coords
        "gt_radii":  (N,)  float32 at training resolution, original (un-augmented) radii
        "path":      str, the input image absolute path (for logging)
        "empiar_id": str
    }

For training (augment=True) the 2D maps (image/mask/heatmap/fp_weight) are augmented
synchronously with random flips and 90° rotations; brightness/contrast is applied to
image only. `gt_coords` / `gt_radii` are *not* transformed (kept in original space) so
they should be ignored during training — loss uses the pre-rendered heatmap.
For validation (augment=False) `gt_coords` / `gt_radii` remain valid and are used for
keypoint F1 evaluation.
"""

from __future__ import annotations

import csv
import glob
import os
import random
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import config


# ──────────────────────────────────────────────────────────────────────
#  Path helpers
# ──────────────────────────────────────────────────────────────────────

def _image_to_mask_path(img_path: str) -> str:
    base, ext = os.path.splitext(img_path)
    return os.path.join(
        os.path.dirname(os.path.dirname(img_path)),
        "masks",
        os.path.basename(base) + "_mask" + ext,
    )


def _image_to_csv_path(img_path: str, subdir: str) -> str:
    base = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(
        os.path.dirname(os.path.dirname(img_path)),
        subdir,
        base + ".csv",
    )


def _infer_empiar_id(img_path: str) -> str:
    # <root>/<EMPIAR_ID>/{train,val}/images/<base>.png → <EMPIAR_ID>
    parts = os.path.normpath(img_path).split(os.sep)
    # Walk up to find the split directory ("train" or "val") and return its parent
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] in ("train", "val") and i >= 1:
            return parts[i - 1]
    return "unknown"


# ──────────────────────────────────────────────────────────────────────
#  Dataset discovery & splitting
# ──────────────────────────────────────────────────────────────────────

def scan_cryoem_root(root: str, split: str = "train") -> list[str]:
    """Return all image paths under ``<root>/*/<split>/images/*.{png,jpg,...}``.

    ``split`` is either ``"train"`` or ``"val"``.  Image stems with the
    standard cryo-EM extensions (png/jpg/jpeg/tif/tiff, case-insensitive)
    are collected from every EMPIAR sub-directory.
    """
    exts = ("png", "jpg", "jpeg", "tif", "tiff")
    paths: list[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(root, "*", split, "images", f"*.{ext}")))
        paths.extend(glob.glob(os.path.join(root, "*", split, "images", f"*.{ext.upper()}")))
    return sorted(set(paths))


def build_train_val_split(
    root: str,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str], str]:
    """Resolve the train / val image paths for the dataset.

    Two layouts are supported:

    1. **Pre-split layout (preferred)** — each EMPIAR contains both
       ``train/`` and ``val/`` sub-folders::

           <root>/<EMPIAR>/train/images/*.png
           <root>/<EMPIAR>/val/images/*.png

       In this case the user-provided split is used verbatim — no random
       sub-splitting is done.  ``val_ratio`` and ``seed`` are ignored.

    2. **Train-only layout (fallback)** — each EMPIAR contains only a
       ``train/`` sub-folder.  In this case ``val_ratio`` of every EMPIAR's
       images are randomly assigned to validation, so every EMPIAR is
       represented in both splits.

    Returns
    -------
    (train_paths, val_paths, mode)
        ``mode`` is ``"pre_split"`` or ``"random_split"`` — useful for
        logging which strategy was actually used.
    """
    train_explicit = scan_cryoem_root(root, split="train")
    val_explicit = scan_cryoem_root(root, split="val")

    # Layout 1: respect the user's pre-split directories
    if train_explicit and val_explicit:
        return sorted(train_explicit), sorted(val_explicit), "pre_split"

    # Layout 2: fallback random split within each EMPIAR's train/
    if not train_explicit:
        return [], [], "pre_split"

    groups: dict[str, list[str]] = {}
    for p in train_explicit:
        groups.setdefault(_infer_empiar_id(p), []).append(p)

    rng = random.Random(seed)
    train_paths: list[str] = []
    val_paths: list[str] = []
    for empiar_id in sorted(groups.keys()):
        files = sorted(groups[empiar_id])
        rng.shuffle(files)
        n_val = max(1, int(round(len(files) * val_ratio))) if len(files) > 1 else 0
        val_paths.extend(files[:n_val])
        train_paths.extend(files[n_val:])
    return sorted(train_paths), sorted(val_paths), "random_split"


# ──────────────────────────────────────────────────────────────────────
#  CSV reading
# ──────────────────────────────────────────────────────────────────────

def _read_xyr_csv(csv_path: str) -> np.ndarray:
    """Read a CSV with header "x,y,radius" and return (N, 3) float array.
    If the file is missing or empty, return a (0, 3) empty array."""
    if not os.path.isfile(csv_path):
        return np.zeros((0, 3), dtype=np.float32)
    rows: list[list[float]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header_seen = False
        for row in reader:
            if not row:
                continue
            if not header_seen:
                header_seen = True
                # Skip header if non-numeric
                try:
                    float(row[0])
                except ValueError:
                    continue
            try:
                x = float(row[0])
                y = float(row[1])
                r = float(row[2]) if len(row) > 2 else 0.0
            except (ValueError, IndexError):
                continue
            rows.append([x, y, r])
    if not rows:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────
#  Heatmap / weight-map rendering
# ──────────────────────────────────────────────────────────────────────

def render_gaussian_heatmap(
    H: int,
    W: int,
    xyr: np.ndarray,
    sigma_scale: float = 1.0 / 3.0,
) -> np.ndarray:
    """Render a CenterNet-style Gaussian heatmap.

    xyr : (N, 3) array of (x, y, radius) at the target (H, W) resolution.
    """
    hm = np.zeros((H, W), dtype=np.float32)
    if xyr.shape[0] == 0:
        return hm
    for x, y, r in xyr:
        sigma = max(r * sigma_scale, 1.0)
        win = max(int(np.ceil(3.0 * sigma)), 1)
        x_i, y_i = int(round(x)), int(round(y))
        x0, x1 = max(0, x_i - win), min(W, x_i + win + 1)
        y0, y1 = max(0, y_i - win), min(H, y_i + win + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        xs = np.arange(x0, x1, dtype=np.float32)[None, :]
        ys = np.arange(y0, y1, dtype=np.float32)[:, None]
        g = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2.0 * sigma * sigma))
        hm[y0:y1, x0:x1] = np.maximum(hm[y0:y1, x0:x1], g)
    return hm


def render_fp_weight_map(
    H: int,
    W: int,
    xyr_fp: np.ndarray,
    k_fp: float = 3.0,
) -> np.ndarray:
    """Render a per-pixel negative-loss weight map.

    Baseline = 1.0 everywhere. Inside each FP disk of radius r,
    weight = max(current, k_fp). Overlaps take the maximum.
    """
    wmap = np.ones((H, W), dtype=np.float32)
    if xyr_fp.shape[0] == 0:
        return wmap
    for x, y, r in xyr_fp:
        r_int = max(int(round(r)), 1)
        x_i, y_i = int(round(x)), int(round(y))
        x0, x1 = max(0, x_i - r_int), min(W, x_i + r_int + 1)
        y0, y1 = max(0, y_i - r_int), min(H, y_i + r_int + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        xs = np.arange(x0, x1, dtype=np.float32)[None, :]
        ys = np.arange(y0, y1, dtype=np.float32)[:, None]
        disk = ((xs - x) ** 2 + (ys - y) ** 2) <= float(r * r)
        patch = wmap[y0:y1, x0:x1]
        wmap[y0:y1, x0:x1] = np.where(disk, np.maximum(patch, k_fp), patch)
    return wmap


# ──────────────────────────────────────────────────────────────────────
#  Augmentation (2D maps in lock-step)
# ──────────────────────────────────────────────────────────────────────

def _augment_maps(maps: list[np.ndarray]) -> list[np.ndarray]:
    """Apply identical random flip / 90° rot to a list of HxW numpy arrays."""
    if random.random() < 0.5:
        maps = [np.flip(m, axis=1) for m in maps]
    if random.random() < 0.5:
        maps = [np.flip(m, axis=0) for m in maps]
    k = random.randint(0, 3)
    if k:
        maps = [np.rot90(m, k) for m in maps]
    return [np.ascontiguousarray(m) for m in maps]


def _image_photometric(image: np.ndarray) -> np.ndarray:
    """Photometric jitter on a [0,1] image only."""
    alpha = random.uniform(0.8, 1.2)  # contrast
    beta = random.uniform(-0.1, 0.1)  # brightness
    return np.clip(alpha * image + beta, 0.0, 1.0).astype(np.float32)


def _normalize(image: np.ndarray) -> np.ndarray:
    """Per-image z-score normalization, then clip to a reasonable range."""
    mean = image.mean()
    std = image.std()
    if std < 1e-6:
        return (image - mean).astype(np.float32)
    normed = (image - mean) / std
    return np.clip(normed, -5.0, 5.0).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────────────────

class CryoEMDataset(Dataset):
    """Heatmap-based cryo-EM particle picking dataset.

    Parameters
    ----------
    image_paths : list[str]
        List of image paths (absolute or relative). Their siblings
        `masks/` / `particle_coordinates/` / `false_positives/` are inferred.
    augment : bool
        If True, apply flip/rotation/photometric augmentation and do not return valid
        gt_coords (heatmap is pre-rendered and flipped in lock-step). If False, return
        original-space gt_coords for keypoint F1 evaluation.
    target_size : tuple[int, int] | None
        (H, W) to resize everything to. Default: (config.input_image_height,
        config.input_image_width).
    sigma_scale : float | None
        σ = sigma_scale * radius. Default: config.heatmap_sigma_scale.
    fp_neg_weight : float | None
        Per-pixel negative-loss weight inside FP disks. Default: config.fp_neg_weight.
    normalize : bool | None
        Per-image z-score. Default: config.normalize_image.
    """

    def __init__(
        self,
        image_paths: list[str],
        augment: bool = False,
        target_size: Optional[tuple[int, int]] = None,
        sigma_scale: Optional[float] = None,
        fp_neg_weight: Optional[float] = None,
        normalize: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.image_paths = list(image_paths)
        self.augment = augment
        self.target_h, self.target_w = target_size or (
            config.input_image_height, config.input_image_width
        )
        self.sigma_scale = config.heatmap_sigma_scale if sigma_scale is None else sigma_scale
        self.fp_neg_weight = config.fp_neg_weight if fp_neg_weight is None else fp_neg_weight
        self.normalize = config.normalize_image if normalize is None else normalize

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        mask_path = _image_to_mask_path(img_path)
        particle_csv = _image_to_csv_path(img_path, "particle_coordinates")
        fp_csv = _image_to_csv_path(img_path, "false_positives")

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # mask is required for the auxiliary task; fall back to zeros
            mask = np.zeros_like(image)

        H_orig, W_orig = image.shape[:2]
        H_out, W_out = self.target_h, self.target_w

        # Resize image + mask
        if (H_orig, W_orig) != (H_out, W_out):
            image = cv2.resize(image, (W_out, H_out), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (W_out, H_out), interpolation=cv2.INTER_NEAREST)
        scale_x = W_out / float(W_orig)
        scale_y = H_out / float(H_orig)
        scale_r = float(np.sqrt(scale_x * scale_y))  # isotropic proxy for radius

        # Read CSVs and rescale to target resolution
        xyr_pos = _read_xyr_csv(particle_csv)
        xyr_fp = _read_xyr_csv(fp_csv)
        if xyr_pos.size:
            xyr_pos = xyr_pos.copy()
            xyr_pos[:, 0] *= scale_x
            xyr_pos[:, 1] *= scale_y
            xyr_pos[:, 2] *= scale_r
        if xyr_fp.size:
            xyr_fp = xyr_fp.copy()
            xyr_fp[:, 0] *= scale_x
            xyr_fp[:, 1] *= scale_y
            xyr_fp[:, 2] *= scale_r

        # Render targets at (H_out, W_out)
        heatmap = render_gaussian_heatmap(H_out, W_out, xyr_pos, self.sigma_scale)
        fp_weight = render_fp_weight_map(H_out, W_out, xyr_fp, self.fp_neg_weight)

        # Normalize mask to [0,1]
        mask = mask.astype(np.float32) / 255.0

        # Image to [0,1] float first (photometric jitter needs this)
        image_f = image.astype(np.float32) / 255.0

        if self.augment:
            # Photometric on image only
            image_f = _image_photometric(image_f)
            # Geometric synchronously on 4 maps
            image_f, mask, heatmap, fp_weight = _augment_maps(
                [image_f, mask, heatmap, fp_weight]
            )

        # Normalization (after photometric so z-score is of the final intensity)
        if self.normalize:
            image_f = _normalize(image_f)

        # To tensors (1, H, W)
        image_t = torch.from_numpy(image_f).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        heatmap_t = torch.from_numpy(heatmap).unsqueeze(0).float()
        fp_weight_t = torch.from_numpy(fp_weight).unsqueeze(0).float()

        if self.augment:
            # gt_coords are invalidated by geometric aug; return empty tensors
            gt_coords_t = torch.zeros((0, 2), dtype=torch.float32)
            gt_radii_t = torch.zeros((0,), dtype=torch.float32)
        else:
            gt_coords_t = torch.from_numpy(xyr_pos[:, :2].astype(np.float32)) \
                if xyr_pos.size else torch.zeros((0, 2), dtype=torch.float32)
            gt_radii_t = torch.from_numpy(xyr_pos[:, 2].astype(np.float32)) \
                if xyr_pos.size else torch.zeros((0,), dtype=torch.float32)

        return {
            "image": image_t,
            "mask": mask_t,
            "heatmap": heatmap_t,
            "fp_weight": fp_weight_t,
            "gt_coords": gt_coords_t,
            "gt_radii": gt_radii_t,
            "path": img_path,
            "empiar_id": _infer_empiar_id(img_path),
        }


# ──────────────────────────────────────────────────────────────────────
#  Collate (variable-length gt_coords per sample)
# ──────────────────────────────────────────────────────────────────────

def cryoem_collate(batch: list[dict]) -> dict:
    images = torch.stack([b["image"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    heatmaps = torch.stack([b["heatmap"] for b in batch], dim=0)
    fp_weights = torch.stack([b["fp_weight"] for b in batch], dim=0)
    return {
        "image": images,
        "mask": masks,
        "heatmap": heatmaps,
        "fp_weight": fp_weights,
        "gt_coords": [b["gt_coords"] for b in batch],
        "gt_radii": [b["gt_radii"] for b in batch],
        "path": [b["path"] for b in batch],
        "empiar_id": [b["empiar_id"] for b in batch],
    }
