#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#  mean_dice_axial_detail.py                                                  #
# --------------------------------------------------------------------------- #
"""
Evaluate axial predictions (.pt) vs GT and print:

GT#   : voxel count per label in ground truth
PR#   : voxel count per label in prediction
D#    : Dice per label
Mean  : (D1 + D2 + D3) / 3
"""

import json, numpy as np, torch, nibabel as nib
from pathlib import Path
from tqdm import tqdm

# ----------------------------- CONFIG ------------------------------------- #
PRED_DIR     = Path("./pred_volumes/axial")     # *_axial_pred.pt
GT_ROOT      = Path("./BraTS2021")              # GT seg.nii.gz
PATIENTS_TXT = Path("./splits/test.txt")        # patient list
OUT_JSON     = Path("./mean_dice_axial_detail2.json")
LABELS       = (1, 2, 3)                        # evaluate foreground classes
# -------------------------------------------------------------------------- #

def load_pred(p: Path) -> np.ndarray:
    arr = torch.load(p, map_location="cpu")
    if isinstance(arr, dict):
        arr = next(iter(arr.values()))
    return arr.numpy().astype(np.uint8)

def load_gt(p: Path) -> np.ndarray:
    return nib.load(str(p)).get_fdata().astype(np.uint8)

def remap(a: np.ndarray) -> np.ndarray:
    a = a.copy()
    a[a == 4] = 3                                # BraTS label 4 → 3
    return a

def dice(pred_bin, gt_bin):
    inter = (pred_bin & gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    return 1.0 if denom == 0 else 2 * inter / denom

def metrics_per_label(pred, gt):
    stats = {}
    for c in LABELS:
        gt_bin   = gt == c
        pred_bin = pred == c
        stats[c] = {
            "gt":   int(gt_bin.sum()),
            "pred": int(pred_bin.sum()),
            "dice": round(dice(pred_bin, gt_bin), 4),
        }
    return stats

# ----------------------------- main --------------------------------------- #
def main():
    pids = [ln.strip() for ln in PATIENTS_TXT.read_text().splitlines() if ln.strip()]
    out  = {}
    total_dice = np.zeros(len(LABELS), dtype=np.float64)

    # header
    hdr = ["Patient"]
    for c in LABELS:
        hdr += [f"GT{c}", f"PR{c}", f"D{c}"]
    hdr += ["Mean"]
    colw = 9
    print(" ".join(f"{h:>{colw}}" if i else f"{h:<20}" for i, h in enumerate(hdr)))
    print("-" * (20 + colw * (len(hdr) - 1)))

    for pid in tqdm(pids):
        pred = remap(load_pred(PRED_DIR / f"{pid}_axial_pred.pt"))
        gt   = remap(load_gt (GT_ROOT / pid / f"{pid}_seg.nii.gz"))
        assert pred.shape == gt.shape, f"shape mismatch for {pid}"

        stats = metrics_per_label(pred, gt)
        dices = [stats[c]["dice"] for c in LABELS]
        mean  = round(float(np.mean(dices)), 4)

        # console row
        row = [pid]
        for c in LABELS:
            row += [stats[c]["gt"], stats[c]["pred"], stats[c]["dice"]]
        row += [mean]
        print(" ".join(
            f"{x:>{colw}.4f}" if isinstance(x, float) else
            f"{x:>{colw}}"    if isinstance(x, int)   else
            f"{x:<20}"
            for x in row
        ))

        # accumulate for average
        total_dice += np.array(dices)
        # store JSON
        out[pid] = {"dice": {str(c): stats[c]["dice"] for c in LABELS},
                    "gt_vox": {str(c): stats[c]["gt"] for c in LABELS},
                    "pred_vox": {str(c): stats[c]["pred"] for c in LABELS},
                    "mean": mean}

    # overall average Dice
    avg_dice = (total_dice / len(pids)).round(4).tolist()
    avg_mean = round(float(np.mean(avg_dice)), 4)
    print("-" * (20 + colw * (len(hdr) - 1)))
    print(f"{'AVERAGE':<20}" + "".join(f"{v:>{colw}.4f}" for v in ([""]*0)))  # placeholder
    print(" " * 20 + " ".join(f"{v:>{colw}.4f}" for v in avg_dice) + f"{avg_mean:>{colw}.4f}")

    out["_average"] = {"dice": {str(c): v for c, v in zip(LABELS, avg_dice)},
                       "mean": avg_mean}
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\n✅  Detailed Dice scores saved to {OUT_JSON.resolve()}")

if __name__ == "__main__":
    main()
