from __future__ import annotations
import glob, os
from pathlib import Path
from typing import Callable, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

# Helper functions
def _remap_labels(mask: torch.Tensor) -> torch.Tensor:
    """Map original BraTS labels {0,1,2,4} → {0,1,2,3}."""
    new = torch.zeros_like(mask)
    new[mask == 1] = 1          # tumour core (ET + NET)
    new[mask == 2] = 2          # oedema / necrosis
    new[mask == 4] = 3          # enhancing tumour
    return new


def _collect_slice_paths(root: str, plane: str, pids: List[str]) -> List[str]:
    """Return sorted list of *.pt slice files for the given patient IDs."""
    paths: List[str] = []
    for pid in pids:
        paths.extend(sorted(Path(root, pid, plane).glob("*.pt")))
    return [str(p) for p in paths]


def load_patient_ids(txt_file: str) -> List[str]:
    """Read non‑empty lines from TXT file."""
    with open(txt_file) as f:
        return [ln.strip() for ln in f if ln.strip()]



class BraTSSliceDataset(Dataset):
    """
    2‑D slice dataset (single plane) for BraTS 2021 .pt files.

    Parameters
    ----------
    root         : dataset root folder
    plane        : "axial" | "coronal" | "sagittal"
    patient_ids  : list[str] – patients to load
    transform    : callable(img, mask) or None
    filter_empty : if True, drop slices whose *remapped* mask is all‑zero
    """
    PLANES = {"axial", "coronal", "sagittal"}

    def __init__(   
        self,
        root: str,
        plane: str,
        patient_ids: List[str],
        transform: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]] | None = None,
        filter_empty: bool = False,
    ):
        if plane not in self.PLANES:
            raise ValueError(f"plane must be one of {self.PLANES}")

        paths = _collect_slice_paths(root, plane, patient_ids)

        if filter_empty:
            kept = []
            for p in paths:
                msk = torch.load(p, map_location="cpu")["mask"]
                if torch.any(_remap_labels(msk) > 0):
                    kept.append(p)
            paths = kept

        if not paths:
            raise FileNotFoundError(f"No slice files remain for plane={plane} after filtering")

        self.paths     = paths
        self.transform = transform


    def __len__(self): return len(self.paths)

    def __getitem__(self, idx: int):
        d   = torch.load(self.paths[idx], map_location="cpu")
        img = d["image"].float()           # (4,H,W)
        msk = _remap_labels(d["mask"].long())

        if self.transform is not None:
            img_np, msk_np = self.transform(img.numpy(), msk.numpy())
            img, msk = torch.from_numpy(img_np).float(), torch.from_numpy(msk_np).long()

        return img, msk


# ------------------------------------------------------------------ #
# Loader helpers
# ------------------------------------------------------------------ #
def _make_loaders(train_ds, val_ds, batch, workers, pin):
    if workers is None:
        workers = max(os.cpu_count() - 2, 2)

    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True,
        num_workers=workers, pin_memory=pin, drop_last=True,
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=batch, shuffle=False,
        num_workers=workers, pin_memory=pin, drop_last=False,
    )
    return train_loader, val_loader


def create_dataloaders_from_txt(
    root: str,
    plane: str,
    batch_size: int,
    train_txt: str,
    val_txt: str,
    num_workers: int | None = None,
    pin_memory: bool = True,
    transform: Tuple[Callable | None, Callable | None] = (None, None),
):


    train_ids = load_patient_ids(train_txt)
    val_ids   = load_patient_ids(val_txt)
    train_transform, val_transform = transform
    train_ds = BraTSSliceDataset(root, plane, train_ids, transform=train_transform, filter_empty=True)
    val_ds   = BraTSSliceDataset(root, plane, val_ids,   transform=val_transform,   filter_empty=False)
    return _make_loaders(train_ds, val_ds, batch_size, num_workers, pin_memory)


def create_test_loader_from_txt(
    root: str,
    plane: str,
    batch_size: int,
    test_txt: str,
    num_workers: int | None = None,
    pin_memory: bool = True,
):
    test_ids = load_patient_ids(test_txt)
    test_ds  = BraTSSliceDataset(root, plane, test_ids, filter_empty=True)

    if num_workers is None:
        num_workers = max(os.cpu_count() - 2, 2)

    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
