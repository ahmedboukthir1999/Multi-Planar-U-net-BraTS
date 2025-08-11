#!/usr/bin/env python3
# -----------------------------------------------------------------------------#
# visualize_augment2d.py – quick visual sanity-check for each 2-D augmentation  #
# -----------------------------------------------------------------------------#
"""Visualise the effect of *each* individual augmentation implemented in
`augment2d.py`.

Given one slice saved as a PyTorch ``.pt`` file, the script renders a single
figure with eight panels: the original slice plus **one augmentation per
panel** (flip-h, flip-v, 90° rotation, gamma, noise, affine, elastic).

The loader is deliberately lenient – your ``.pt`` can be:
  • a *tensor* holding the image only (shape **C×H×W**)  
  • a *dict* with keys ``image`` & ``mask`` (or ``img`` & ``msk``)           

If the mask is not found inside the first file, pass it via ``--mask_pt``.

Usage
-----
```bash
python visualize_augment2d.py \
        --image_pt BraTS2D_PT_Slices/BraTS2021_00001/axial/109.pt \
        --out_png  slice_aug_vis.png
```

• Omit ``--mask_pt`` when both tensors live in the same file.  
• ``matplotlib`` is required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# low-level augmentation helpers
from augment2d import (
    _random_gamma,
    _add_noise,
    _random_affine,
    _elastic_deform,
)

plt.rcParams.update({"font.size": 9})

# ------------------------------------------------------------------------- #
# --------------------------- util helpers -------------------------------- #


def _safe_numpy(x):
    """Torch tensor → NumPy (copy)."""
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def load_pt(path: Path):
    """Load ``.pt`` that may be tensor or *dict of* tensors/arrays."""
    obj = torch.load(path, map_location="cpu")
    if torch.is_tensor(obj):
        return _safe_numpy(obj)
    if isinstance(obj, dict):
        return {k: _safe_numpy(v) for k, v in obj.items()}
    raise TypeError(f"Unsupported object type in {path}: {type(obj)}")


def overlay(ax, img2d: np.ndarray, msk2d: np.ndarray, title: str):
    ax.imshow(img2d, cmap="gray", interpolation="none")
    ax.imshow(np.ma.masked_where(msk2d == 0, msk2d), cmap="jet", alpha=0.35, interpolation="none")
    ax.set_title(title)
    ax.axis("off")

# ------------------------------------------------------------------------- #
# ----------------------------- main -------------------------------------- #


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--image_pt", required=True, help="PyTorch .pt with slice image or dict containing image & mask")
    ap.add_argument("--mask_pt", help="Optional .pt containing mask if not inside --image_pt")
    ap.add_argument("--out_png", help="Save figure here instead of displaying interactively")
    args = ap.parse_args()

    data = load_pt(Path(args.image_pt))

    if isinstance(data, dict):
        keys = set(data.keys())

        # Select image tensor
        for k in ("img", "image", "vol", "slice", "pred"):
            if k in data:
                img = data[k]
                break
        else:
            raise KeyError(f"No image tensor found in {args.image_pt}. Expected key 'image' or similar.")

        # Select mask tensor
        for k in ("msk", "mask", "label", "seg"):
            if k in data:
                msk = data[k]
                break
        else:
            if args.mask_pt is not None:
                msk = load_pt(Path(args.mask_pt))
            else:
                raise ValueError("Mask not found inside image_pt and --mask_pt not provided.")

    else:
        img = data
        if args.mask_pt is None:
            raise ValueError("--mask_pt required when image_pt does not contain a mask.")
        msk = load_pt(Path(args.mask_pt))

    if img.ndim != 3 or msk.ndim != 2:
        raise ValueError(f"Expected shapes (C,H,W) & (H,W) – got {img.shape} & {msk.shape}.")

    img0 = img[0]  # first modality for display (assumed 0-1 float)

    # ---------------- generate aug variants --------------------------- #
    variants = [
        (img0, msk, "original"),
        (np.flip(img, axis=2)[0], np.flip(msk, axis=1), "flip-h"),
        (np.flip(img, axis=1)[0], np.flip(msk, axis=0), "flip-v"),
        (np.rot90(img, k=1, axes=(1, 2))[0], np.rot90(msk, k=1, axes=(0, 1)), "rot 90°"),
        (_random_gamma(img.copy())[0], msk, "gamma"),
        (_add_noise(img.copy())[0], msk, "noise"),
    ]

    img_aff, msk_aff = _random_affine(img.copy(), msk.copy())
    variants.append((img_aff[0], msk_aff, "affine"))

    img_el, msk_el = _elastic_deform(img.copy(), msk.copy())
    variants.append((img_el[0], msk_el, "elastic"))

    # ---------------- plot -------------------------------------------- #
    n = len(variants)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.5))

    for ax, (im, ma, title) in zip(axes, variants):
        overlay(ax, im, ma, title)

    plt.tight_layout()

    if args.out_png:
        plt.savefig(args.out_png, dpi=150, bbox_inches="tight")
        print(f"✅  Saved → {Path(args.out_png).resolve()}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
