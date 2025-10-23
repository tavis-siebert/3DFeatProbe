import torch
import torch.nn as nn
from typing import Optional
from torchvision.transforms import Resize, InterpolationMode

from src.utils.image_utils import center_pad

class BaseProcessor(nn.Module):
    """
    Basic input processor which handles
    1. Resizing
    2. Normalizing
    3. Padding to match patch size
    """
    def __init__(
        self, 
        resize: bool = False,
        resize_size: int | list[int] = 512,
        interpolation_mode: str = 'bicubic',
        patch_size: Optional[int] = None, 
        normalize: bool = False,
        mean: list[int] = [0.485, 0.456, 0.406], 
        std: list[int] = [0.229, 0.224, 0.225]
    ):
        """
        Args:
            resize (bool): Whether to resize the image
            resize_size (int, list[int]): The size to resize the image to. 
                                            Either an int (square) or H,W specified by resize_size
            interpolation_mode (str): The interpolation mode for resizing. Either 'bilinear' or 'bicubic'
            patch_size (int, Optional): The patch size to pad to. By default there is none to support non-ViTs
            normalize (bool): Whether to normalize the images with a std and mean. Defaults to False
            mean (list[int]): The mean to normalize images with. Defaults to ImageNet
            std (list[int]): The std to normalize images with. Defaults to ImageNet
        """
        super().__init__()
        # handle resizing
        self.resize = resize
        if self.resize:
            if interpolation_mode == 'bilinear':
                interp = InterpolationMode.BILINEAR
            elif interpolation_mode == 'bicubic':
                interp = InterpolationMode.BICUBIC
            else:
                raise ValueError(f"Interpolatio mode {interpolation_mode} unrecognized. Must be one of 'bilinear' or 'bicubic'")

            if isinstance(resize_size, int):
                resize_size = (resize_size, resize_size)
            self.resizer = Resize(size=resize_size, interpolation=interp)
        
        self.patch_size = patch_size
        self.normalize = normalize
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Normalize images before feeding them into DINO models.
        
        Args:
            images (torch.Tensor): Batch of images of shape (B, 3, H, W).
                Can be in range [0, 255] or [0, 1].

        Returns:
            torch.Tensor: Normalized images of shape (B, 3, H, W).
        """
        # scale to [0, 1] if necessary
        if images.max() > 1.0:
            images = images / 255.0

        # resize using interpolation
        if self.resize:
            images = self.resizer(images)

        # normalie with ImageNet mean, std
        if self.normalize:
            images = (images - self.mean) / self.std

        # center pad to ensure divisibility w/ patch size
        if self.patch_size is not None:
            images = center_pad(images, self.patch_size)
        
        return images
