#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#  make_axial_png.py                                                          #
# --------------------------------------------------------------------------- #
"""
Create a side-by-side PNG (GT vs axial prediction) for a single patient,
using the axial slice with the largest number of GT labels.

Input
-----
  • axial prediction .pt  : <PRED_DIR>/<pid>_axial_pred.pt
  • GT mask seg.nii.gz    : <DATA_ROOT>/<pid>/<pid>_seg.nii.gz

Output
------
  • <OUT_DIR>/<pid>_gt_vs_pred.png
"""

import random
from pathlib import Path

import numpy as np
import torch
import nibabel as nib
from PIL import Image, ImageDraw, ImageFont

# ------------------------- 🔧 CONFIG -------------------------------------- #
DATA_ROOT    = Path("./BraTS2021")           # original dataset root
PRED_DIR     = Path("./pred_volumes/axial")  # contains *_axial_pred.pt
PATIENTS_TXT = Path("./splits/test.txt")     # list of patient IDs
OUT_DIR      = Path("./axial_pngs")          # where PNGs will be saved
# -------------------------------------------------------------------------- #

def load_pred(pt_path: Path) -> np.ndarray:
    arr = torch.load(pt_path, map_location="cpu")
    if isinstance(arr, dict):
        arr = next(iter(arr.values()))
    return arr.numpy().astype(np.uint8)

def load_gt(nii_path: Path) -> np.ndarray:
    return nib.load(str(nii_path)).get_fdata().astype(np.uint8)

def label_to_rgb(mask: np.ndarray) -> np.ndarray:
    lut = {
        0: (0, 0, 0),         # background
        1: (255, 165, 0),     # edema
        2: (0, 255, 0),       # necrosis
        3: (255, 0, 0),       # enhancing
        4: (255, 0, 0),       # label 4 (rare)
    }
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for k, c in lut.items():
        rgb[mask == k] = c
    return rgb

def find_largest_gt_slice(gt: np.ndarray) -> int:
    return np.argmax([np.count_nonzero(gt[:, :, i]) for i in range(gt.shape[2])])

def add_labels(img: Image.Image, labels: list[str]) -> Image.Image:
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    W, H = img.size
    w2 = W // 2

    draw.text((w2//2 - 30, 5), labels[0], font=font, fill=(255, 255, 255))
    draw.text((w2 + w2//2 - 30, 5), labels[1], font=font, fill=(255, 255, 255))

    return img

def main():
    # Choose random patient
    pids = [ln.strip() for ln in PATIENTS_TXT.read_text().splitlines() if ln.strip()]
    pid  = random.choice(pids)
    print(f"🧠  Patient : {pid}")

    pred_path = PRED_DIR / f"{pid}_axial_pred.pt"
    gt_path   = DATA_ROOT / pid / f"{pid}_seg.nii.gz"

    pred = load_pred(pred_path)   # (240,240,155)
    gt   = load_gt(gt_path)       # (240,240,155)

    assert pred.shape == gt.shape == (240,240,155), "Shape mismatch!"

    best_k = find_largest_gt_slice(gt)
    print(f"📸  Selected slice: {best_k} (most labeled GT)")

    gt_rgb   = label_to_rgb(gt[:, :, best_k])
    pred_rgb = label_to_rgb(pred[:, :, best_k])

    side_by_side = np.concatenate([gt_rgb, pred_rgb], axis=1)
    img = Image.fromarray(side_by_side)
    img = add_labels(img, ["Ground Truth", "Prediction"])

    out_path = OUT_DIR / f"{pid}_gt_vs_pred.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"✅  Saved PNG → {out_path}")

if __name__ == "__main__":
    main()
