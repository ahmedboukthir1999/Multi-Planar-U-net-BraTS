import os
import torch
import multiprocessing as mp
import numpy as np
import nibabel as nib
from tqdm import tqdm


def normalize(vol: np.ndarray):
    """Foreground‑aware per‑volume min‑max normalisation.

    Parameters
    ----------
    vol : np.ndarray
        3‑D volume, dtype float32.

    Returns
    -------
    normed : np.ndarray
        Normalised volume in [0,1].  Zeros remain zeros.
    fg_mask : np.ndarray[bool]
        True where the volume is non‑zero (brain foreground).
    """
    fg_mask = vol > 0
    rng = np.ptp(vol)
    if rng == 0:
        return np.zeros_like(vol, dtype=np.float32), fg_mask
    normed = (vol - vol.min()) / (rng + 1e-8)
    return normed.astype(np.float32), fg_mask


def crop(img: np.ndarray, msk: np.ndarray, fg_union: np.ndarray, patch_size=(204, 204)):
    """Smart crop centred on foreground; zero‑pads if slice is smaller than patch."""
    ph, pw = patch_size
    C, H, W = img.shape

    pad_h = max(0, ph - H)
    pad_w = max(0, pw - W)
    if pad_h or pad_w:
        pad = (
            (0, 0),
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2),
        )
        img = np.pad(img, pad, mode="constant")
        msk = np.pad(msk, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)), mode="constant")
        fg_union = np.pad(fg_union, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)), mode="constant")
        _, H, W = img.shape

    if fg_union.any():
        ys, xs = np.nonzero(fg_union)
        cy, cx = int(ys.mean()), int(xs.mean())
    else:
        cy, cx = H // 2, W // 2

    sy = np.clip(cy - ph // 2, 0, H - ph)
    sx = np.clip(cx - pw // 2, 0, W - pw)

    return img[:, sy:sy + ph, sx:sx + pw], msk[sy:sy + ph, sx:sx + pw]


# Helpers for saving & checks

def save_slice(save_dir: str, idx: int, img: np.ndarray, msk: np.ndarray):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{idx:03d}.pt")
    if not os.path.isfile(path):
        torch.save({"image": torch.tensor(img), "mask": torch.tensor(msk)}, path)


def is_patient_done(pat_out: str):
    """Return True if patient directory exists and at least one slice per plane already saved."""
    planes = ["axial", "coronal", "sagittal"]
    return all(
        os.path.isdir(os.path.join(pat_out, p)) and len(os.listdir(os.path.join(pat_out, p))) > 0
        for p in planes
    )


def load_nii(path: str):
    return nib.load(path).get_fdata().astype(np.float32)



def is_all_background(msk_slice: np.ndarray, fg_slice: np.ndarray):
    """True if slice contains no brain AND no tumour."""
    return (not fg_slice.any()) and (msk_slice.max() == 0)



# Per‑patient worker


def process_patient(args):
    pid, data_root, out_root = args
    pat_out = os.path.join(out_root, pid)
    if is_patient_done(pat_out):
        return 0 

    # ---- Load modalities
    path = lambda mod: os.path.join(data_root, pid, f"{pid}_{'seg' if mod == 'seg' else mod}.nii.gz")
    vols = {mod: load_nii(path(mod)) for mod in ["flair", "t1", "t1ce", "t2", "seg"]}
    flair, fg1 = normalize(vols["flair"])
    t1,    fg2 = normalize(vols["t1"])
    t1ce,  fg3 = normalize(vols["t1ce"])
    t2,    fg4 = normalize(vols["t2"])
    seg = vols["seg"].astype(np.uint8)
    fg_union = fg1 | fg2 | fg3 | fg4  # brain foreground mask

    H, W, D = flair.shape  # 240×240×155 for BraTS

    # ---- Axial slices (Z‑axis)
    for k in range(D):
        if is_all_background(seg[:, :, k], fg_union[:, :, k]):
            continue
        img = np.stack([flair[:, :, k], t1[:, :, k], t1ce[:, :, k], t2[:, :, k]], axis=0)
        crop_img, crop_msk = crop(img, seg[:, :, k], fg_union[:, :, k])
        save_slice(os.path.join(pat_out, "axial"), k, crop_img, crop_msk)

    # ---- Coronal slices (Y‑axis)
    for k in range(H):
        if is_all_background(seg[:, k, :], fg_union[:, k, :]):
            continue
        img = np.stack([flair[:, k, :], t1[:, k, :], t1ce[:, k, :], t2[:, k, :]], axis=0)
        crop_img, crop_msk = crop(img, seg[:, k, :], fg_union[:, k, :])
        save_slice(os.path.join(pat_out, "coronal"), k, crop_img, crop_msk)

    # ---- Sagittal slices (X‑axis)
    for k in range(W):
        if is_all_background(seg[k, :, :], fg_union[k, :, :]):
            continue
        img = np.stack([flair[k, :, :], t1[k, :, :], t1ce[k, :, :], t2[k, :, :]], axis=0)
        crop_img, crop_msk = crop(img, seg[k, :, :], fg_union[k, :, :])
        save_slice(os.path.join(pat_out, "sagittal"), k, crop_img, crop_msk)

    return 1


# ---------------------------
# Main
# ---------------------------

def main(data_root: str, out_root: str, n_workers: int = 4):
    patients = sorted(d for d in os.listdir(data_root) if d.startswith("BraTS2021_"))
    print(f"Patients to process: {len(patients)}")

    args = [(pid, data_root, out_root) for pid in patients]
    with mp.Pool(processes=n_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_patient, args), total=len(patients)))

    done = sum(results)
    print(f"\n✅ Finished. Processed {done}/{len(patients)} patients.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main("BraTS2021", "BraTS2D_PT_Slices", n_workers=4)
