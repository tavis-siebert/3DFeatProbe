
import torch
import torch.nn.functional as F

def compute_padding_hw(images: torch.Tensor, patch_size: int) -> tuple[int, int]:
    """
    Compute the padding needed for height and width to be divisible by patch_size

    Args:
        images (torch.Tensor): Tensor of shape (B, C, H, W)
        patch_size (int): Patch size to make height and width divisible by
    """
    _, _, h, w = images.shape
    diff_h = h % patch_size
    diff_w = w % patch_size
    pad_h = patch_size - diff_h if diff_h > 0 else 0
    pad_w = patch_size - diff_w if diff_w > 0 else 0
    return pad_h, pad_w

def center_pad(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Center pad images so that their height and width are divisible by patch_size

    Args:
        images (torch.Tensor): Tensor of shape (B, C, H, W)
        patch_size (int): Patch size to make height and width divisible by
    Returns:
        Padded image with H, W divisble by patch_size
    """
    pad_h, pad_w = compute_padding_hw(images, patch_size)
    if pad_h == 0 and pad_w == 0:
        return images

    pad_t = pad_h // 2
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l
    pad_b = pad_h - pad_t

    images = F.pad(images, (pad_l, pad_r, pad_t, pad_b))
    return images