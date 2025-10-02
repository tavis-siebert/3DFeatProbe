
import torch
import torch.nn as nn
from typing import Dict
from transformers import Siglip2VisionModel, AutoProcessor

from src.utils.image_utils import center_pad

class SigLIPProcessor(nn.Module):
    def __init__(self, model_id: str, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.processor = AutoProcessor.from_pretrained(model_id)
    
    def forward(self, images: torch.Tensor) -> tuple[Dict, int, int]:
        """
        Normalize images before feeding them into DINO models.
        
        Args:
            images (torch.Tensor): Batch of images of shape (B, 3, H, W).
                Can be in range [0, 255] or [0, 1].

        Returns:
            the BatchFeature dict from SigLIP's image processor
            tuple[int, int] number of patches in each spatial dimension
        """
        # scale to [0, 1] if necessary
        if images.max() > 1.0:
            images = images / 255.0
        
        # use processor to avoid shape mismatches
        images = center_pad(images, self.patch_size)
        _, _, h, w = images.shape
        num_patches_h, num_patches_w = h // self.patch_size, w // self.patch_size
        num_patches = num_patches_h * num_patches_w

        input = self.processor(
            images, 
            return_tensors="pt", 
            do_resize=False, 
            do_rescale=False,
            patch_size=self.patch_size,
            max_num_patches=num_patches
        )
        return input, num_patches_h, num_patches_w


class SigLIP2(nn.Module):
    """
    SigLIP2 (image encoder) model class
    """
    def __init__(self, backbone: str="base"):
        super().__init__()

        possible_backbones = ("base", "so400m")
        assert backbone in possible_backbones, "Backbone must be one of {}".format(possible_backbones)

        model_id = f"google/siglip2-{backbone}-patch16-naflex"
        self.model = Siglip2VisionModel.from_pretrained(model_id)
        self.patch_size = self.model.config.patch_size
        self.feature_dim = self.model.config.hidden_size
        self.processor = SigLIPProcessor(model_id=model_id, patch_size=self.patch_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): Batch of images of shape (B, C, H, W).

        Returns:
            feature map (torch.Tensor) of final hidden layer (B, H_patch, W_patch, D)
                We return the 2D spatial map because it's easier to go from unflattened to flattened than vise versa
        """
        input, num_patches_h, num_patches_w = self.processor(images)
        num_patches = num_patches_h * num_patches_w

        outputs = self.model(**input)
        last_hidden_states = outputs.last_hidden_state
        feature_map = last_hidden_states[:, -num_patches:, :].unflatten(1, (num_patches_h, num_patches_w))
        return feature_map