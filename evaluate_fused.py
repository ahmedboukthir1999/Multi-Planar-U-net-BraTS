#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#  mean_dice_fused_detail.py                                                  #
# --------------------------------------------------------------------------- #
"""
Detailed Dice report for fused predictions (*.nii.gz) vs. BraTS ground truth.

For each patient:
  GT#   : voxel count of label # in ground truth
  PR#   : voxel count of label # in prediction
  D#    : Dice of label #
  Mean  : (D1 + D2 + D3) / 3
"""

import json, numpy as np, nibabel as nib
from pathlib import Path
from tqdm import tqdm

# ------------------------- 🔧 CONFIG -------------------------------------- #
PRED_DIR     = Path("./fused_nii")             # *_seg_fused.nii.gz
GT_ROOT      = Path("./BraTS2021")             # ground-truth folder
PATIENTS_TXT = Path("./splits/test.txt")       # patient IDs
OUT_JSON     = Path("./mean_dice_fused_detail2.json")
LABELS       = (1, 2, 3)                       # foreground classes
COL_W        = 9                               # console column width
# -------------------------------------------------------------------------- #

def load_nii(p: Path) -> np.ndarray:
    return nib.load(str(p)).get_fdata().astype(np.uint8)

def remap(a: np.ndarray) -> np.ndarray:
    a = a.copy()
    a[a == 4] = 3                               # BraTS label 4 → 3
    return a

def dice(pred_bin, gt_bin):
    inter = (pred_bin & gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    return 1.0 if denom == 0 else 2 * inter / denom

def label_stats(pred, gt):
    stats = {}
    for c in LABELS:
        gt_bin, pr_bin = gt == c, pred == c
        stats[c] = {
            "gt":   int(gt_bin.sum()),
            "pred": int(pr_bin.sum()),
            "dice": round(dice(pr_bin, gt_bin), 4),
        }
    return stats

# ------------------------- main ------------------------------------------ #
def main():
    pids = [ln.strip() for ln in PATIENTS_TXT.read_text().splitlines() if ln.strip()]
    results, sum_dice = {}, np.zeros(len(LABELS), dtype=np.float64)

    # header
    hdr = ["Patient"] + [f"{x}{c}" for c in LABELS for x in ("GT", "PR", "D")] + ["Mean"]
    print(" ".join(f"{h:>{COL_W}}" if i else f"{h:<20}" for i, h in enumerate(hdr)))
    print("-" * (20 + COL_W * (len(hdr) - 1)))

    for pid in tqdm(pids):
        pred = remap(load_nii(PRED_DIR / f"{pid}_seg_fused.nii.gz"))
        gt   = remap(load_nii(GT_ROOT / pid / f"{pid}_seg.nii.gz"))
        assert pred.shape == gt.shape, f"shape mismatch for {pid}"

        st   = label_stats(pred, gt)
        dices = [st[c]["dice"] for c in LABELS]
        mean  = round(float(np.mean(dices)), 4)

        # console row
        row = [pid]
        for c in LABELS:
            row += [st[c]["gt"], st[c]["pred"], st[c]["dice"]]
        row += [mean]
        print(" ".join(
            f"{val:>{COL_W}.4f}" if isinstance(val, float) else
            f"{val:>{COL_W}}"    if isinstance(val, int)   else
            f"{val:<20}"
            for val in row
        ))

        sum_dice += np.array(dices)
        results[pid] = {
            "dice":      {str(c): st[c]["dice"] for c in LABELS},
            "gt_vox":    {str(c): st[c]["gt"]   for c in LABELS},
            "pred_vox":  {str(c): st[c]["pred"] for c in LABELS},
            "mean":      mean,
        }

    avg_dice = (sum_dice / len(pids)).round(4).tolist()
    avg_mean = round(float(np.mean(avg_dice)), 4)
    print("-" * (20 + COL_W * (len(hdr) - 1)))
    avg_row = ([""] * (2 * len(LABELS))) + avg_dice + [avg_mean]
    print(f"{'AVERAGE':<20}" +
      "".join(f"{v:>{COL_W}.4f}" if isinstance(v, float) else f"{v:>{COL_W}}"
              for v in avg_row))


    results["_average"] = {"dice": {str(c): v for c, v in zip(LABELS, avg_dice)},
                           "mean": avg_mean}
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\n✅  Detailed Dice scores saved to {OUT_JSON.resolve()}")

if __name__ == "__main__":
    main()
