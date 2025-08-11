import numpy as np
import random
import scipy.ndimage as ndi

def _random_gamma(img: np.ndarray, gamma_range=(0.7, 1.5)):
    gamma = random.uniform(*gamma_range)
    return img ** gamma

def _add_noise(img: np.ndarray, std_range=(0.01, 0.05)):
    std = random.uniform(*std_range)
    noise = np.random.normal(0, std, img.shape)
    return np.clip(img + noise, 0.0, 1.0)

def _random_affine(img: np.ndarray, msk: np.ndarray, max_shift=10):
    tx = random.randint(-max_shift, max_shift)
    ty = random.randint(-max_shift, max_shift)
    shift = (ty, tx)  

    img_aug = np.stack([ndi.shift(ch, shift, order=1, mode="nearest") for ch in img], axis=0)
    msk_aug = ndi.shift(msk, shift, order=0, mode="nearest")
    return img_aug, msk_aug

def _elastic_deform(img: np.ndarray, msk: np.ndarray, alpha=10, sigma=5):
    random_state = np.random.RandomState(None)
    shape = img.shape[1:]  # (H,W)

    dx = ndi.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="reflect") * alpha
    dy = ndi.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="reflect") * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (y + dy).reshape(-1), (x + dx).reshape(-1)

    img_deformed = np.stack([ndi.map_coordinates(ch, indices, order=1, mode="reflect").reshape(shape)
                              for ch in img], axis=0)
    msk_deformed = ndi.map_coordinates(msk, indices, order=0, mode="nearest").reshape(shape)
    return img_deformed, msk_deformed

def augment2d(img: np.ndarray, msk: np.ndarray):
    """
    In‑plane 2‑D augmentation for a slice pair (img: C×H×W, msk: H×W).

    Includes:
    - Flip (horizontal/vertical)
    - 0/90/180/270 rotation
    - Random gamma correction
    - Gaussian noise
    - Random affine shift
    - Elastic deformation

    Returns:
    img_aug : np.ndarray  (C, H, W)
    msk_aug : np.ndarray  (H, W)
    """
    # Flip
    if random.random() < 0.5:
        img = np.flip(img, axis=2)
        msk = np.flip(msk, axis=1)

    if random.random() < 0.5:
        img = np.flip(img, axis=1)
        msk = np.flip(msk, axis=0)

    # Rotation
    k = random.choice([0, 1, 2, 3])
    if k:
        img = np.rot90(img, k=k, axes=(1, 2))
        msk = np.rot90(msk, k=k, axes=(0, 1))

    # Intensity augmentations
    if random.random() < 0.5:
        img = _random_gamma(img)

    if random.random() < 0.5:
        img = _add_noise(img)

    # Geometric augmentations
    if random.random() < 0.5:
        img, msk = _random_affine(img, msk)

    if random.random() < 0.3:
        img, msk = _elastic_deform(img, msk)

    return img.copy(), msk.copy()
