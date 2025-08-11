# unet2d.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------------- #
# Helper blocks                                                             #
# ------------------------------------------------------------------------- #
class DoubleConv(nn.Module):
    """(Conv → INorm → ReLU) × 2"""
    def __init__(self, in_ch: int, out_ch: int, mid_ch: int | None = None,
                 p_dropout: float = 0.0):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p_dropout),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downscale with MaxPool then DoubleConv"""
    def __init__(self, in_ch, out_ch, p_dropout=0.0):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, p_dropout=p_dropout)
        )

    def forward(self, x):
        return self.mpconv(x)


class Up(nn.Module):
    """Upscale → concat skip → DoubleConv"""
    def __init__(self, in_ch, out_ch, bilinear=True, p_dropout=0.0):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            # in_ch includes skip channels, halve for internal conv
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2, p_dropout)
        else:
            self.up  = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch, p_dropout=p_dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Pad if inexact due to odd input dimensions
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        if diff_y or diff_x:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


# ------------------------------------------------------------------------- #
# Main U-Net                                                                #
# ------------------------------------------------------------------------- #
class UNet2D(nn.Module):
    """
    2-D U-Net for BraTS slice segmentation.

    Args
    ----
    in_channels  : 4   (FLAIR, T1, T1CE, T2)
    n_classes    : 4   (background, edema, non-enhancing, enhancing, necrosis)
    base_filters : 32  (feature maps in first layer)
    bilinear     : True for cheap upsampling, False for transposed conv
    p_dropout    : dropout prob inside DoubleConv blocks (set >0 for reg.)
    """
    def __init__(self,
                 in_channels: int = 4,
                 n_classes:   int = 4,
                 base_filters: int = 32,
                 bilinear: bool = True,
                 p_dropout: float = 0.0):
        super().__init__()
        f = base_filters

        self.inc   = DoubleConv(in_channels, f,            p_dropout=p_dropout)
        self.down1 = Down(f,        f * 2,                 p_dropout=p_dropout)
        self.down2 = Down(f * 2,    f * 4,                 p_dropout=p_dropout)
        self.down3 = Down(f * 4,    f * 8,                 p_dropout=p_dropout)
        factor     = 2 if bilinear else 1
        self.down4 = Down(f * 8,    f * 16 // factor,      p_dropout=p_dropout)

        self.up1   = Up(f * 16,     f * 8 // factor, bilinear, p_dropout)
        self.up2   = Up(f * 8,      f * 4 // factor, bilinear, p_dropout)
        self.up3   = Up(f * 4,      f * 2 // factor, bilinear, p_dropout)
        self.up4   = Up(f * 2,      f,              bilinear, p_dropout)

        self.outc  = OutConv(f, n_classes)

    def forward(self, x):
        x1 = self.inc(x)     # (B,f,H,W)
        x2 = self.down1(x1)  # (B,2f,H/2,W/2)
        x3 = self.down2(x2)  # (B,4f,H/4,W/4)
        x4 = self.down3(x3)  # (B,8f,H/8,W/8)
        x5 = self.down4(x4)  # (B,16f,H/16,W/16)

        x = self.up1(x5, x4) # (B,8f,H/8,W/8)
        x = self.up2(x,  x3) # (B,4f,H/4,W/4)
        x = self.up3(x,  x2) # (B,2f,H/2,W/2)
        x = self.up4(x,  x1) # (B,f,H,W)
        logits = self.outc(x)
        return logits
