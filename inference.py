#!/usr/bin/env python3
# -----------------------------------------------------------------------------#
#  predict_volume.py                                                           #
# -----------------------------------------------------------------------------#
"""
Generate 3-D prediction masks (one file per patient per plane) from 
trained 2-D U-Net checkpoints.

"""

from pathlib import Path
import argparse, torch
from tqdm import tqdm
import torch.nn.functional as F
from dataset import load_patient_ids, BraTSSliceDataset
from unet2d import UNet2D


PLANE_DEPTH = {"axial": 155, "coronal": 240, "sagittal": 240}
CROP = 204
FULL = 240
PAD_BEF = (FULL - CROP) // 2  # 18
PAD_AFT = FULL - CROP - PAD_BEF


def pad_slice(mask2d: torch.Tensor) -> torch.Tensor:
    """Pad (204,204) → (240,240) with zeros (symmetric)."""
    return F.pad(mask2d, (PAD_BEF, PAD_AFT, PAD_BEF, PAD_AFT), value=0)



@torch.no_grad()
def predict_patient(root: Path, pid: str, plane: str,
                    model: torch.nn.Module, device: str, out_dir: Path,
                    batch_size: int = 16):
    """Run inference for a single patient + plane and save .pt volume."""
    ds = BraTSSliceDataset(root=str(root), plane=plane, patient_ids=[pid])
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                         shuffle=False, num_workers=2,
                                         pin_memory=True, drop_last=False)

    depth = PLANE_DEPTH[plane]
    vol   = torch.zeros((FULL, FULL, depth), dtype=torch.int64)   # (H,W,D)

    global_idx = 0
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        preds  = logits.argmax(1).cpu()      # (B,204,204)

        for i in range(preds.size(0)):
            slice_path = loader.dataset.paths[global_idx + i]
            k = int(Path(slice_path).stem) 
            vol[:, :, k] = pad_slice(preds[i])

        global_idx += preds.size(0)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(vol, out_dir / f"{pid}_{plane}_pred.pt")



def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reconstruct 3-D prediction masks (per plane) from 2-D "
                    "slice inference using your trained checkpoints.")
    ap.add_argument("--root", default="BraTS2D_PT_Slices",)
    ap.add_argument("--patients_txt", required=True,)
    ap.add_argument("--plane", choices=["axial", "coronal", "sagittal"],required=True)
    ap.add_argument("--checkpoint", required=True,)
    ap.add_argument("--out_dir", default="pred_volumes")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = UNet2D(in_channels=4, n_classes=4)
    ckpt   = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    pids = load_patient_ids(args.patients_txt)
    root = Path(args.root)
    out  = Path(args.out_dir)
    print(f"\n🚀  Running inference ({args.plane}) on {len(pids)} patients")
    for pid in tqdm(pids, unit="patient"):
        predict_patient(root, pid, args.plane, model, device, out,
                        batch_size=args.batch_size)

    print(f"\n✅  Predictions saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
