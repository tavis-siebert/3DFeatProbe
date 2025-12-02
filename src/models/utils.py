import logging
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Callable

from src.utils.logging import direct_logger_to_stdout

logger = logging.getLogger(__name__)
direct_logger_to_stdout(logger)

def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    strict: bool=False,
    weights_only: bool=False,
    filter_fn: Optional[Callable]=None,
):
    """Load checkpoint into model.

    Args:
        model: Model to load checkpoint into.
        checkpoint_path: Path to checkpoint file.
        strict: Whether to strictly enforce state_dict keys match.
        weights_only: Whether to load only weights (torch.load parameter).
        filter_fn: Optional function to filter state dict.
    """
    logger.info(f"Loading from checkpoint: {checkpoint_path}")
    
    with open(checkpoint_path, "rb") as f:
        try:
            checkpoint = torch.load(f, map_location="cpu", weights_only=weights_only)
        except TypeError:
            checkpoint = torch.load(f, map_location="cpu")

        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        if filter_fn:
            state_dict = filter_fn(state_dict)

        missing, unexpected = model.load_state_dict(
            state_dict, strict=strict
        )
        if unexpected:
            logger.warning(f"Got unexpected keys: {unexpected}")
        if missing and not "processor.mean" in missing:
            raise ValueError(f"Missing keys: {missing}")

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