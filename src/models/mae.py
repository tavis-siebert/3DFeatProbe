import torch
import torch.nn as nn
from transformers import ViTMAEModel

from src.utils.image_utils import center_pad

class MAEProcessor(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
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


class MAE(nn.Module):
    """
    Wrapper for ViT-MAE encoder to produce dense patch features
    """
    def __init__(self, backbone: str="base"):
        super().__init__()
        possible_backbones = ("base", "large", "huge")
        assert backbone in possible_backbones, "Backbone must be one of {}".format(possible_backbones)

        model_id = f"facebook/vit-mae-{backbone}"
        self.model = ViTMAEModel.from_pretrained(model_id)
        self.patch_size = self.model.config.patch_size
        self.feature_dim = self.model.config.hidden_size
        self.model.config.mask_ratio = 0.0  # disable masking
        self.processor = MAEProcessor(self.patch_size)

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
