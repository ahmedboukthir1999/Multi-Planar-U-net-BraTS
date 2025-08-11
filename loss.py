import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

__all__ = ["_one_hot", "dice_coeff", "CEDiceLoss"]



def _one_hot(
    labels: torch.Tensor,
    num_classes: int,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Convert integer label map (B,H,W) → one-hot (B,C,H,W).

    Parameters
    ----------
    labels       : torch.Tensor, shape (B,H,W), dtype long / int
    num_classes  : int  – number of classes C
    dtype        : torch.dtype or None (defaults to float32)

    Returns
    -------
    one_hot      : torch.Tensor, shape (B,C,H,W), dtype `dtype`
    """
    if dtype is None:
        dtype = torch.float32
    b, h, w = labels.shape
    one_hot = torch.zeros(b, num_classes, h, w, device=labels.device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0)


def dice_coeff(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
    include_background: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute soft Dice per class *and* the mean Dice.

    Parameters
    ----------
    logits             : (B,C,H,W) – raw network outputs
    targets            : (B,H,W)   – integer labels
    include_background : bool      – average over all C classes (True)
                                     or exclude class 0 (False)

    Returns
    -------
    dice_per_class     : (C,)  tensor with Dice for each class
    mean_dice          : ()    scalar with mean Dice
    """
    C = logits.shape[1]
    probs   = F.softmax(logits, dim=1)          
    targets = _one_hot(targets, C, dtype=probs.dtype)

    dims = (0, 2, 3)                            
    inter = torch.sum(probs * targets, dim=dims)
    denom = torch.sum(probs + targets, dim=dims)

    dice  = (2.0 * inter + eps) / (denom + eps) # (C,)

    if include_background:
        mean_dice = dice.mean()
    else:
        mean_dice = dice[1:].mean()             # skip class 0

    return dice, mean_dice


 
class CEDiceLoss(nn.Module):
    """
    L = α · CrossEntropy  +  β · (1 – mean Dice)

    Parameters
    ----------
    alpha               : weight for CE term
    beta                : weight for Dice term
    weight              : optional class-weighting tensor for CE
    ignore_index        : optional label ID to ignore in CE
    include_background  : include class 0 in Dice mean (True / False)
    """
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        weight: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = None,
        include_background: bool = False,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.include_background = include_background
 
        ce_kwargs = {"weight": weight} if weight is not None else {}       
        if ignore_index is not None:            # forward only if user supplied
            ce_kwargs["ignore_index"] = ignore_index
        self.ce = nn.CrossEntropyLoss(**ce_kwargs)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        _, mean_dice = dice_coeff(
            logits, targets, include_background=self.include_background
        )
        dice_loss = 1.0 - mean_dice
        return self.alpha * ce_loss + self.beta * dice_loss
