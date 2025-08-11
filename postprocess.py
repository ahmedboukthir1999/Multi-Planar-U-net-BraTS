#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#  postprocess_plane_volumes.py                                               #
# --------------------------------------------------------------------------- #
"""
Clean per-plane BraTS prediction volumes *before* fusion.

Input   : <IN_DIR>/<pid>_<plane>_pred.pt          (tensor or {"pred":tensor})
Output  : <OUT_DIR>/<pid>_<plane>_pred_clean.pt   (same format)

Processing
----------
1. For each class 1-3:
      • binary_closing  (optional, --closing ITER)
      • connected components, keep regions ≥ --min_size voxels
2. Background (0) untouched.
"""

import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from scipy.ndimage import label, binary_closing, generate_binary_structure

# ------------------ helpers ------------------------------------------------ #
def load_pt(path: Path) -> torch.Tensor:
    arr = torch.load(path, map_location="cpu")
    if isinstance(arr, dict): arr = next(iter(arr.values()))
    return arr.clone()              

def save_pt(arr: torch.Tensor, orig: torch.Tensor | dict, out: Path):
    if isinstance(orig, dict):
        cleaned = orig.copy()
        cleaned[list(orig.keys())[0]] = arr
        torch.save(cleaned, out)
    else:
        torch.save(arr, out)

def clean_volume(vol: np.ndarray, min_size: int, closing_iter: int) -> np.ndarray:
    cleaned = np.zeros_like(vol)
    structure = generate_binary_structure(3, 2) 

    for cls in [1, 2, 3]:
        binary = vol == cls
        if closing_iter:
            binary = binary_closing(binary, structure=structure,
                                    iterations=closing_iter)
        labeled, n = label(binary, structure=structure)
        for i in range(1, n + 1):
            comp = labeled == i
            if comp.sum() >= min_size:
                cleaned[comp] = cls
    return cleaned


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Post-process per-plane prediction volumes before fusion.")
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_size", type=int, default=100)
    ap.add_argument("--closing", type=int, default=0)
    args = ap.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    vols = sorted(in_dir.glob("*_pred.pt"))
    print(f"🔍  Found {len(vols)} prediction volumes")

    for p in tqdm(vols, unit="vol"):
        orig = torch.load(p, map_location="cpu")
        tensor = orig if isinstance(orig, torch.Tensor) else next(iter(orig.values()))
        vol_np = tensor.numpy().astype(np.uint8)

        cleaned_np = clean_volume(vol_np, args.min_size, args.closing)
        save_pt(torch.from_numpy(cleaned_np), orig, p)   # strip '_pred' add '_clean'

    print(f"\n✅  Cleaned volumes saved to {out_dir.resolve()}")

if __name__ == "__main__":
    main()
