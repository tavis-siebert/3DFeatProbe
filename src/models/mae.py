import torch
import torch.nn as nn
from transformers import ViTMAEModel

from src.models.processor import BaseProcessor

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
        self.processor = BaseProcessor(patch_size=self.patch_size, normalize=True)

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
