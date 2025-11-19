import torch
import torch.nn.functional as F
from typing import Tuple

def interpolate_positional_embeddings(
    positional_embeddings: torch.Tensor,
    original_image_size: Tuple[int, int],
    target_image_size: Tuple[int, int],
    patch_size: int,
    num_additional_tokens: int = 1,
) -> torch.Tensor:
    """
    Function interpolates positional embeddings to a different image size.
    Code adapted form: https://github.com/tum-vision/scenedino/blob/080ff3ea16a81603861f4ad62272b5548d9fd7f8/scenedino/models/backbones/dino/vit.py#L65

    Args:
        positional_embeddings (torch.Tensor): Positional embeddings of the sape [1, N, C].
        original_image_size (Tuple[int, int]): Original image size as a tuple.
        target_image_size (Tuple[int, int]): Target image size as a tuple.
        patch_size (int): Utilize patch size.
        num_additional_tokens (int): Number of additional tokens used. Default 1 (class token).

    Returns:
        positional_embeddings_interpolated (torch.Tensor): Interpolated positional embeddings [1, N_new, C].
    """
    # Get positional embeddings for image
    if num_additional_tokens > 0:
        positional_embeddings_add_tokens = positional_embeddings[:, :num_additional_tokens]
        positional_embeddings_image = positional_embeddings[:, num_additional_tokens:]
    else:
        positional_embeddings_image = positional_embeddings
    # Reshape embeddings to 2D
    positional_embeddings_image = positional_embeddings_image.view(
        1, original_image_size[0] // patch_size, original_image_size[1] // patch_size, -1
    )
    # Interpolate positional embeddings
    positional_embeddings_image = F.interpolate(
        positional_embeddings_image.permute(0, 3, 1, 2),
        size=(target_image_size[0] // patch_size, target_image_size[1] // patch_size),
        mode="bicubic",
        align_corners=False,
        antialias=False,
    ).permute(0, 2, 3, 1)
    # Stack positional embeddings again
    if num_additional_tokens > 0:
        positional_embeddings_interpolated = torch.cat(
            (positional_embeddings_add_tokens, positional_embeddings_image.flatten(1, 2)), dim=1
        )
    else:
        positional_embeddings_interpolated = positional_embeddings_image.flatten(1, 2)
    return positional_embeddings_interpolated