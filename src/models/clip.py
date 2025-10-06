import torch
import torch.nn as nn
from transformers import CLIPVisionModel

from src.utils.image_utils import center_pad

class CLIPProcessor(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.register_buffer("mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
    
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
        
        # center pad to ensure divisibility w/ patch size
        images = center_pad(images, self.patch_size)

        # normalie with ImageNet mean, std
        return (images - self.mean) / self.std

class CLIP(nn.Module):
    """
    CLIP (image encoder) model class
    """
    def __init__(self, backbone: str="vit-base-patch16"):
        """
        Args:
            backbone (str): Backbone architecture. See model README for details
        """
        super().__init__()

        possible_backbones = ("vit-large-patch14", "vit-base-patch16", "vit-base-patch32")
        assert backbone in possible_backbones, "Backbone must be one of {}".format(possible_backbones)
        
        self.model = CLIPVisionModel.from_pretrained(f"openai/clip-{backbone}")
        self.patch_size = self.model.config.patch_size
        self.feature_dim = self.model.config.hidden_size
        self.processor = CLIPProcessor(self.patch_size)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): Batch of images of shape (B, C, H, W).

        Returns:
            feature map (torch.Tensor) of final hidden layer (B, H_patch, W_patch, D)
                We return the 2D spatial map because it's easier to go from unflattened to flattened than vise versa
        """
        images = self.processor(images)
        _, _, h, w = images.shape
        num_patches_h, num_patches_w = h // self.patch_size, w // self.patch_size
        num_patches = num_patches_h * num_patches_w

        outputs = self.model(pixel_values=images, interpolate_pos_encoding=True)
        last_hidden_states = outputs.last_hidden_state
        feature_map = last_hidden_states[:, -num_patches:, :].unflatten(1, (num_patches_h, num_patches_w))
        return feature_map
