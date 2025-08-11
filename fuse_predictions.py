#!/usr/bin/env python3
# -----------------------------------------------------------------------------#
#  fuse_to_nii.py  –  axial-priority fusion                                    #
# -----------------------------------------------------------------------------#
"""
Fuse axial, coronal and sagittal prediction volumes into a final BraTS-style
segmentation (240 × 240 × 155) saved as <pid>_seg_fused.nii.gz.

Fusion rule
-----------
• Wherever axial is NON-background (label ≠ 0) → keep axial label.
• Where axial == 0:
      ─ if only coronal or sagittal is non-zero → take that label
      ─ if both are non-zero and identical      → take that label
      ─ if both are non-zero but DIFFER         → coronal wins
      ─ if both zero                            → stay zero
"""

import argparse, json
from pathlib import Path

import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm

PAD = 18            
Z   = 155           
# --------------------------- io helpers ----------------------------------- #
def load_pt(p: Path) -> np.ndarray:
    arr = torch.load(p, map_location="cpu")
    if isinstance(arr, dict):          
        arr = next(iter(arr.values()))
    return arr.numpy().astype(np.uint8)

def reorient_coronal(vol: np.ndarray) -> np.ndarray:
    vol = np.transpose(vol, (0, 2, 1))        # (H, Y, Zpad)
    return vol[:, :, PAD: PAD + Z]            # (240,240,155)

def reorient_sagittal(vol: np.ndarray) -> np.ndarray:
    vol = np.transpose(vol, (2, 0, 1))        # (X, Y, Zpad)
    return vol[:, :, PAD: PAD + Z]            # (240,240,155)


def priority_fuse(ax, co, sa):
    fused = ax.copy()

    bg = (ax == 0)

    # voxels where exactly one of co/sa is non-zero
    take_co = bg & (co != 0) & (sa == 0)
    take_sa = bg & (sa != 0) & (co == 0)
    fused[take_co] = co[take_co]
    fused[take_sa] = sa[take_sa]

    # voxels where both co & sa are non-zero
    both = bg & (co != 0) & (sa != 0)
    agree    = both & (co == sa)
    disagree = both & (co != sa)
    fused[agree]    = co[agree] 
    fused[disagree] = co[disagree]  

    return fused.astype(np.uint8)

# --------------------------- main ----------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Fuse axial / coronal / sagittal predictions with axial "
                    "priority and save <pid>_seg_fused.nii.gz.")
    ap.add_argument("--ax_dir", required=True)
    ap.add_argument("--co_dir", required=True)
    ap.add_argument("--sa_dir", required=True)
    ap.add_argument("--patients_txt", required=True)
    ap.add_argument("--brats_root", required=True)
    ap.add_argument("--out_dir", default="fused_nii")
    args = ap.parse_args()

    ax_dir, co_dir, sa_dir = map(Path, (args.ax_dir, args.co_dir, args.sa_dir))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    pids = [ln.strip() for ln in Path(args.patients_txt).read_text().splitlines()
            if ln.strip()]
    summary = {}

    for pid in tqdm(pids, unit="patient"):
        # ---------- load predictions ------------------------------------ #
        ax = load_pt(ax_dir / f"{pid}_axial_pred.pt")        # (240,240,155)
        co = load_pt(co_dir / f"{pid}_coronal_pred.pt")      # (240,240,240)
        sa = load_pt(sa_dir / f"{pid}_sagittal_pred.pt")     # (240,240,240)

        # ---------- re-orient ------------------------------------------ #
        co_aln = reorient_coronal(co)
        sa_aln = reorient_sagittal(sa)

        # ---------- fuse with axial priority --------------------------- #
        fused = priority_fuse(ax, co_aln, sa_aln)            # (240,240,155)

        # ---------- save NIfTI ---------------------------------------- #
        affine = nib.load(Path(args.brats_root)/pid/f"{pid}_seg.nii.gz").affine
        nib.save(nib.Nifti1Image(fused, affine),
                 out_dir / f"{pid}_seg_fused.nii.gz")

        summary[pid] = {"ax_nonzero": int((ax!=0).sum()),
                        "filled_from_other": int(((ax==0)&(fused!=0)).sum())}

    (out_dir/"fusion_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n✅  Saved fused volumes → {out_dir.resolve()}")

if __name__ == "__main__":
    main()
