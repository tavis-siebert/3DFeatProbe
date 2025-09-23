
import torch
import torch.nn.functional as F

def center_pad(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Center pad images so that their height and width are divisible by patch_size

    Args:
        images (torch.Tensor): Tensor of shape (B, C, H, W)
        patch_size (int): Patch size to make height and width divisible by
    Returns:
        Padded image with H, W divisble by patch_size
    """
    _, _, h, w = images.shape
    diff_h = h % patch_size
    diff_w = w % patch_size

    if diff_h == 0 and diff_w == 0:
        return images

    pad_h = patch_size - diff_h
    pad_w = patch_size - diff_w

    pad_t = pad_h // 2
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l
    pad_b = pad_h - pad_t

    images = F.pad(images, (pad_l, pad_r, pad_t, pad_b))
    return images