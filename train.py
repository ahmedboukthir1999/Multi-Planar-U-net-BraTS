#!/usr/bin/env python
# train.py
# -----------------------------------------------------------------------------#
# 2‑D U‑Net training on BraTS multiplanar slices                                #
# Combines the modern training loop (cosine LR, SummaryWriter, AMP, checkpoints)
# from the new script with the richer console and CSV logging from the older
# version.                                                                     #
# -----------------------------------------------------------------------------#

import argparse, csv, json, math, os, platform, shutil, time
from datetime import datetime
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from augment2d import augment2d
from dataset import create_dataloaders_from_txt
from loss import CEDiceLoss, dice_coeff
from unet2d import UNet2D


# Utilities                       


def save_checkpoint(state, is_best: bool, out_dir: Path, epoch: int):
    """Save checkpoint files (latest, best, epoch‑specific)."""
    ckpt_path = out_dir / "checkpoint_latest.pt"
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copy2(ckpt_path, out_dir / "checkpoint_best.pt")
    torch.save(state, out_dir / f"checkpoint_{epoch:03d}.pt")


def set_seed(seed: int = 0):
    """Make experiment reproducible as far as possible."""
    import random, numpy as np  
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



# Training / Validation loops                                                  


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    use_amp: bool,
):
    """Single training epoch with fancy tqdm logging."""
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"   🏋️  Training [epoch {epoch:03d}]", leave=False, unit="batch")

    for imgs, masks in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, masks)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validation loop with per‑class Dice."""
    model.eval()
    total_loss = 0.0
    dice_sum = None
    num_pixels = 0  

    pbar = tqdm(loader, desc="   🧪 Validating", leave=False, unit="batch")
    for imgs, masks in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, masks)

        total_loss += loss.item() * imgs.size(0)
        dice_per_class, _ = dice_coeff(logits, masks, include_background=False)
        if dice_sum is None:
            dice_sum = torch.zeros_like(dice_per_class)
        dice_sum += dice_per_class * imgs.size(0)
        num_pixels += imgs.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    val_loss = total_loss / num_pixels
    dice_avg = dice_sum / num_pixels
    mean_dice = dice_avg[1:].mean().item()
    return val_loss, dice_avg.detach().cpu().tolist(), mean_dice



# Main                                                                          


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data
    parser.add_argument("--root", required=True, help="Pre‑processed .pt slice root")
    parser.add_argument("--train_txt", required=True)
    parser.add_argument("--val_txt", required=True)
    parser.add_argument("--plane", required=True, choices=["axial", "coronal", "sagittal"])
    # training
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.4, help="CE weight")
    parser.add_argument("--beta", type=float, default=0.6, help="Dice weight")
    parser.add_argument("--amp", action="store_true", help="Mixed‑precision training")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    # model
    parser.add_argument("--base_filters", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    # misc
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="runs")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()


    # Environment                                     

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🕒 Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        mem_total = torch.cuda.mem_get_info()[1] / 1e9
        print(f"🖥  Using device: CUDA ({gpu_name})")
        print(f"💾 GPU memory available: {mem_total:.2f} GB")
    else:
        print(f"🖥  Using device: CPU ({platform.processor() or 'Generic x86'})")

    # Output / logging dirs                                                 
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) / f"{args.plane}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cfg.json").write_text(json.dumps(vars(args), indent=2))

    # TensorBoard
    tb_writer = SummaryWriter(log_dir=args.log_dir or str(out_dir / "tb"))

    # CSV metrics
    metrics_path = out_dir / "metrics.csv"
    metrics_file = metrics_path.open("w", newline="")
    csv_writer = csv.writer(metrics_file)
    csv_writer.writerow([
        "epoch",
        "train_loss",
        "val_loss",
        "dice_edema",
        "dice_core",
        "dice_enhancing",
        "mean_dice",
        "lr",
        "sec_per_epoch",
    ])

  
    # Dataloaders                                                           
   
    train_loader, val_loader = create_dataloaders_from_txt(
        root=args.root,
        plane=args.plane,
        batch_size=args.batch,
        train_txt=args.train_txt,
        val_txt=args.val_txt,
        num_workers=args.num_workers,
        pin_memory=True,
        transform=(augment2d, None),
    )

    # Model, loss, optimiser                                                

    model = UNet2D(
        in_channels=4,
        n_classes=4,
        base_filters=args.base_filters,
        p_dropout=args.dropout,
    ).to(device)

    criterion = CEDiceLoss(alpha=args.alpha, beta=args.beta, include_background=False)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = args.amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

  
    # Resume                                                                

    start_epoch = 1
    best_dice = 0.0
    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        scheduler.load_state_dict(ckpt.get("sched_state", scheduler.state_dict()))
        best_dice = ckpt.get("best_dice", 0.0)
        start_epoch = ckpt["epoch"] + 1
        print(f"🔄 Resumed from {args.resume} (epoch {start_epoch})")

   
    # Training loop 
 
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n🚀 Starting epoch {epoch}/{args.epochs}")
        t0 = time.time()

        tr_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch,
            use_amp,
        )
        val_loss, dice_per_cls, mean_dice = validate(model, val_loader, criterion, device)

        if device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # in MB
            print(f"💾 Peak GPU memory this epoch: {peak_mem:.2f} MB")
            torch.cuda.reset_peak_memory_stats()

        scheduler.step()

        # Per‑class diced: [Edema, Core, Enhancing].
        d_edema, d_core, d_enh = dice_per_cls[1:]
        lr_now = optimizer.param_groups[0]["lr"]
        sec_per_epoch = time.time() - t0

        # Console summary
        print(
            "📅 Epoch {0:03d} | Train {1:.4f} | Val {2:.4f} | "
            "Dice (E/C/Enh) {3:.3f}/{4:.3f}/{5:.3f} | Mean {6:.4f}".format(
                epoch,
                tr_loss,
                val_loss,
                d_edema,
                d_core,
                d_enh,
                mean_dice,
            )
        )
        print(f"📈 Learning Rate: {lr_now:.6f}")
        print(f"⏱  Epoch time: {sec_per_epoch/60:.2f} min")

        # TensorBoard
        tb_writer.add_scalar("Loss/train", tr_loss, epoch)
        tb_writer.add_scalar("Loss/val", val_loss, epoch)
        tb_writer.add_scalar("Dice/mean", mean_dice, epoch)
        for k, d in enumerate([d_edema, d_core, d_enh]):
            tb_writer.add_scalar(f"Dice/class_{k}", d, epoch)
        tb_writer.add_scalar("LR", lr_now, epoch)

        # CSV
        csv_writer.writerow([epoch, tr_loss, val_loss, d_edema, d_core, d_enh, mean_dice, lr_now, sec_per_epoch])
        metrics_file.flush()

        # Checkpoints
        is_best = mean_dice > best_dice
        best_dice = max(best_dice, mean_dice)

        save_checkpoint(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "sched_state": scheduler.state_dict(),
                "best_dice": best_dice,
            },
            is_best=is_best,
            out_dir=out_dir,
            epoch=epoch,
        )


    tb_writer.close()
    metrics_file.close()

    print(f"\n✅ Training finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🏁 Best Val Dice: {best_dice:.4f}")
    print(f"📌 Best checkpoint saved in: {out_dir}")


if __name__ == "__main__":
    main()
